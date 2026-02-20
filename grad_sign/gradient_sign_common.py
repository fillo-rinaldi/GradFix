import itertools
import logging
import pickle
from copy import deepcopy
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from grad_sign.dataset_sampler import DatasetSampler, PRECOMPUTED_DIR
from task_vectors.src.task_vectors import TaskVector

logger = logging.getLogger(__name__)


def compare_signs(tb_signs, gradient_signs):
    """Compute overall and per-parameter sign agreement percentage."""
    total_params = 0
    matching_signs = 0
    per_param_sign_agreement = {}

    for name, tb_sign in tb_signs.items():
        if name in gradient_signs:
            grad_sign = gradient_signs[name]
            match = (tb_sign == grad_sign).sum().item()
            total = tb_sign.numel()
            agreement = match / total * 100

            per_param_sign_agreement[name] = agreement
            matching_signs += match
            total_params += total

    overall_sign_agreement = matching_signs / total_params * 100
    return overall_sign_agreement, per_param_sign_agreement


def taskvector_from_gradient_signs(
    gradient_signs,
    reference_taskvector,
    fallback_taskvector=None,
    mask_mode="normal",
    beta=5.0,
    gamma=1.0,
    c=1.0,
):
    """Build a TaskVector from gradient signs using a reference vector."""
    tv_vector = {}

    for key in gradient_signs.keys():
        t = reference_taskvector.vector[key]
        g = gradient_signs[key]

        if mask_mode == "force":
            tv_vector[key] = torch.abs(t) * torch.sign(g)
        elif mask_mode == "normal":
            tv_vector[key] = torch.where(torch.sign(g) == torch.sign(t), t, torch.zeros_like(t))
        elif mask_mode == "random":
            rand_signs = torch.sign(torch.rand_like(g) * 2 - 1)
            tv_vector[key] = torch.abs(t) * rand_signs
        elif mask_mode in ["soft1", "soft1_rowwise"]:
            if mask_mode == "soft1_rowwise" and t.ndim >= 2:
                t_flat = t.view(t.shape[0], -1)
                g_flat = g.view(g.shape[0], -1)

                masks = []
                for i in range(t.shape[0]):
                    t_row = t_flat[i]
                    g_row = g_flat[i]
                    cos_sim = torch.dot(t_row, g_row) / (t_row.norm() * g_row.norm() + 1e-8)
                    masks.append(torch.sigmoid(beta * cos_sim))

                masks = torch.stack(masks)
                shape = [t.shape[0]] + [1] * (t.ndim - 1)
                mask = masks.view(*shape).expand_as(t)
            else:
                denom = (t.norm() * g.norm() + 1e-8)
                mask = torch.sigmoid(beta * (t * g / denom))

            tv_vector[key] = mask * t
        elif mask_mode in ["soft2", "soft2_rowwise"]:
            if mask_mode == "soft2_rowwise" and t.ndim >= 2:
                t_flat = t.view(t.shape[0], -1)
                g_flat = g.view(g.shape[0], -1)
                masks = []
                for i in range(t.shape[0]):
                    alignment = torch.dot(t_flat[i], g_flat[i])
                    masks.append(torch.clamp(torch.tanh(gamma * alignment), min=0.0))
                masks = torch.stack(masks)
                shape = [t.shape[0]] + [1] * (t.ndim - 1)
                mask = masks.view(*shape).expand_as(t)
            else:
                alignment = t * g
                mask = torch.clamp(torch.tanh(gamma * alignment), min=0.0)

            tv_vector[key] = mask * t
        elif mask_mode in ["soft3", "soft3_rowwise"]:
            if mask_mode == "soft3_rowwise" and t.ndim >= 2:
                t_flat = t.view(t.shape[0], -1)
                g_flat = g.view(g.shape[0], -1)
                masks = []
                for i in range(t.shape[0]):
                    same_sign = (torch.sign(t_flat[i]) == torch.sign(g_flat[i])).float()
                    sign_agreement = same_sign.mean()
                    weight = torch.abs(g_flat[i]).mean() / (torch.abs(g_flat[i]).mean() + c)
                    masks.append(sign_agreement * weight)
                masks = torch.stack(masks)
                shape = [t.shape[0]] + [1] * (t.ndim - 1)
                mask = masks.view(*shape).expand_as(t)
            else:
                same_sign = (torch.sign(t) == torch.sign(g)).float()
                mask = same_sign * torch.abs(g) / (torch.abs(g) + c)

            tv_vector[key] = mask * t
        else:
            raise ValueError(
                f"Invalid mode '{mask_mode}'. Use 'normal', 'force', 'random', "
                "'soft1', 'soft1_rowwise', 'soft2', 'soft2_rowwise', "
                "'soft3', 'soft3_rowwise'."
            )

    if fallback_taskvector is not None:
        for key in reference_taskvector.vector.keys():
            if "shortcut" in key:
                tv_vector[key] = fallback_taskvector.vector[key]

    return TaskVector(vector=tv_vector)


def autodetect_indices_file(dataset_name: str):
    """Try to find a precomputed indices pickle matching the dataset name.

    Search path:
    <repo_root>/precomputed_indices/
    """
    dataset_name = dataset_name.lower()
    patterns = [f"*{dataset_name}*indices*.pkl"]

    # Only check the repo-local precomputed_indices folder
    repo_root = Path(__file__).resolve().parent
    repo_precomp = repo_root / "precomputed_indices"
    for patt in patterns:
        candidates = list(repo_precomp.glob(
            patt)) if repo_precomp.exists() else []
        if candidates:
            candidates.sort(key=lambda p: ("medoids" not in p.name,
                            "herding" not in p.name, "coreset" not in p.name))
            return candidates[0]

    return None


class FirstNBatches:
    """Iterable wrapper to yield only the first N batches from an existing dataloader."""

    def __init__(self, dataloader, n):
        self.dataloader = dataloader
        self.n = int(n)

    def __iter__(self):
        return itertools.islice(iter(self.dataloader), self.n)

    def __len__(self):
        try:
            return min(self.n, len(self.dataloader))
        except TypeError:
            return self.n


def build_realgrad_dataloader(train_dataset, base_loader, args, imgs_indices=None):
    """Build a dataloader for real-gradient computation.

    - If imgs_indices is provided (dict[class_id -> list[int]]), build a Subset dataloader over those indices.
    - Else if args.num_batches is provided, wrap base_loader to only yield first N batches.
    - Else return base_loader as-is.
    """
    if imgs_indices is not None:
        flat_idx = []
        for _, idxs in imgs_indices.items():
            flat_idx.extend(list(map(int, idxs)))
        subset = Subset(train_dataset, flat_idx)
        return DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
    if getattr(args, "num_batches", None) is not None:
        return FirstNBatches(base_loader, args.num_batches)
    return base_loader


def compute_real_gradient_signs(
    pretrained_model,
    dataset,
    loss_fn,
    device,
    dataloader,
    optimize=False,
    prompt_ensemble=True,
    vote="mean",
    show_progress=False,
):
    """Compute signed gradients for the visual backbone from a dataloader.

    - vote="mean": sign of the mean gradient
    - vote="max": majority vote of the signs per image
    """

    pretrained_model.eval()
    pretrained_model.zero_grad()
    model = deepcopy(pretrained_model).to(device)

    model.eval()

    if optimize:
        model.train()
        optimizer = torch.optim.AdamW(model.visual.parameters(), lr=1e-5)
        optimizer.zero_grad()

    trainable_parameters = [
        p for _, p in model.visual.named_parameters() if p.requires_grad]
    named_trainable_parameters = [
        (name, p) for name, p in model.visual.named_parameters() if p.requires_grad]

    def build_text_features():
        if prompt_ensemble and hasattr(dataset, "templates"):
            prompts = [[template(c.lower()) for c in dataset.class_names]
                       for template in dataset.templates]
            with torch.no_grad():
                template_embeddings = []
                for template in prompts:
                    texts = open_clip.tokenize(template).to(device)
                    text_feats = F.normalize(model.encode_text(texts), dim=-1)
                    template_embeddings.append(text_feats)
                return torch.mean(torch.stack(template_embeddings), dim=0)
        else:
            prompts = [dataset.single_template(
                c.lower()) for c in dataset.class_names]
            with torch.no_grad():
                texts = open_clip.tokenize(prompts).to(device)
                return F.normalize(model.encode_text(texts), dim=-1)

    text_features = build_text_features()
    total_steps = max(1, len(dataloader))
    scale = 1.0 / float(total_steps)

    if vote == "max":
        sign_sums = {
            name: torch.zeros_like(p, device=device)
            for name, p in model.visual.named_parameters()
            if p.requires_grad
        }

    iterator = tqdm(dataloader) if show_progress else dataloader
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device).long()

        image_features = F.normalize(model.encode_image(images), dim=-1)
        vl_logits = model.logit_scale.exp() * (image_features @ text_features.t())

        if vote == "mean":
            total_loss = loss_fn(vl_logits, labels)
            (total_loss * scale).backward()
        elif vote == "max":
            losses = F.cross_entropy(
                vl_logits, labels, reduction="none") * scale
            for i in range(len(images)):
                grads = torch.autograd.grad(
                    losses[i],
                    trainable_parameters,
                    retain_graph=True,
                    create_graph=False,
                )

                for (name, _), g in zip(named_trainable_parameters, grads):
                    if g is not None:
                        sign_sums[name] += torch.sign(-g.detach())
            if optimize:
                total_loss = losses.mean()
                total_loss.backward()
            del grads
        else:
            raise ValueError(
                f"Invalid vote method '{vote}'. Use 'mean' or 'max'.")

    gradient_signs = {}
    if vote == "mean":
        for name, param in model.visual.named_parameters():
            if param.grad is not None:
                gradient_signs[name] = torch.sign(-param.grad)
    elif vote == "max":
        gradient_signs = {name: torch.sign(acc)
                          for name, acc in sign_sums.items()}

    if optimize:
        optimizer.step()

    return gradient_signs, (model if optimize else None)


def load_precomputed_indices(spec: str, train_dataset, device, feature_extractor=None):
    """
    Load per-class image indices to compute real gradient signs.
    Accepted 'spec' formats:
      - Absolute/relative path to an existing .pkl (dict[int -> list[int]])
      - A method spec 'coreset:K' | 'herding:K' | 'k-medoid:K' (also accepts ',' or '_' as separator)
        When a method spec is provided:
            - It looks for precomputed_indices/{dataset}_{tag}_indices_{K}.pkl
                where tag is 'coreset' | 'herding' | 'medoids' (for 'k-medoid').
            - If not found, it computes indices on-the-fly via DatasetSampler and uses them.
    """
    p = Path(spec)
    # Case 1: direct .pkl path
    if p.suffix == ".pkl":
        if not p.exists():
            raise FileNotFoundError(f"Pickle not found: {p}")
        with open(p, "rb") as f:
            return pickle.load(f)

    # Case 2: method:K (or method,K | method_K)
    for sep in (":", ",", "_"):
        if sep in spec:
            method_raw, k_raw = spec.split(sep, 1)
            method = method_raw.strip().lower()
            if method not in {"coreset", "herding", "k-medoid"}:
                raise ValueError(
                    f"Invalid method '{method}'. Use: coreset, herding, k-medoid.")
            try:
                k = int(k_raw)
            except Exception as e:
                raise ValueError(f"Invalid K in spec '{spec}': {e}")

            ds_name = getattr(train_dataset, "name", "").lower()
            if not ds_name:
                raise RuntimeError(
                    "Dataset does not expose 'name'; cannot build precomputed filename.")
            tag = "medoids" if method == "k-medoid" else method
            std = Path(PRECOMPUTED_DIR) / f"{ds_name}_{tag}_indices_{k}.pkl"
            legacy = Path(PRECOMPUTED_DIR) / f"{ds_name}_{tag}_indices.pkl"
            if std.exists():
                with open(std, "rb") as f:
                    return pickle.load(f)
            if legacy.exists():
                with open(legacy, "rb") as f:
                    return pickle.load(f)

            logger.info(
                f"No precomputed found ({std.name}). Computing '{method}' indices with K={k}...")
            sampler = DatasetSampler(
                train_dataset,
                strategy=method,
                device=device,
                feature_extractor=feature_extractor,
            )
            return sampler.sample_indices_per_class(
                k=k,
                use_precomputed=True,
                save_precomputed=True,
            )

    raise ValueError(
        f"--precomputed_indices='{spec}' is neither an existing .pkl nor a valid 'method:K' spec.")
