import os
import sys
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict

from grad_sign.utils import (
    set_seed,
    get_models,
    load_dataset,
    evaluate_model,
    build_base_parser,
    evaluate_task_vectors,
)
from grad_sign.models import OpenCLIPModel
from task_vectors.src.task_vectors import TaskVector
from grad_sign.dataset_sampler import DatasetSampler
from grad_sign.gradient_sign_common import (
    build_realgrad_dataloader,
    compute_real_gradient_signs,
    load_precomputed_indices,
    taskvector_from_gradient_signs,
)

try:
    import wandb
    os.environ["WANDB__SERVICE_WAIT"] = "800"
except ImportError:
    wandb = None

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_local_args(argv=None):
    """Parse CLI args using the shared base parser."""
    argv = list(sys.argv[1:] if argv is None else argv)
    base_parser = build_base_parser()
    return base_parser.parse_args(argv)


def sign_agreement(grad_signs_a, grad_signs_b, names=("A", "B")):
    """
    Compute the percentage of sign agreement between two sets of gradient signs.
    """
    logger.info(f"Computing sign agreement")
    common_keys = set(grad_signs_a.keys()) & set(grad_signs_b.keys())
    agreement = 0
    total = 0
    for key in common_keys:
        # Compare only signs (1, -1, 0).
        a = torch.sign(grad_signs_a[key])
        b = torch.sign(grad_signs_b[key])
        agreement += (a == b).sum().item()
        total += a.numel()
    final_agreement = agreement / total if total > 0 else 0
    logger.info(
        f"sign_agreement_{names[0]}_wrt_{names[1]}: {agreement}/{total} = {final_agreement*100:.2f}%")
    if wandb is not None:
        wandb.log(
            {f"analysis/sign_agreement/{names[0]}_vs_{names[1]}": final_agreement})
    return final_agreement


def cosine_similarity_taskvectors(vec_a: dict, vec_b: dict, names=("A", "B")) -> float:
    """Compute cosine similarity between two task vectors (dict[name->tensor]).

    - Concatenate all parameters shared by both vectors in a fixed order.
    - Skip keys with mismatched shapes; log once if any skipped.
    """
    common = sorted(set(vec_a.keys()) & set(vec_b.keys()))
    a_chunks = []
    b_chunks = []
    skipped = 0
    for k in common:
        ta = vec_a[k]
        tb = vec_b[k]
        if ta.shape != tb.shape:
            skipped += 1
            continue
        a_chunks.append(ta.detach().float().reshape(-1))
        b_chunks.append(tb.detach().float().reshape(-1))
    if not a_chunks:
        logger.warning(
            f"cosine_similarity: no comparable keys between {names[0]} and {names[1]}")
        return float("nan")
    if skipped:
        logger.debug(
            f"cosine_similarity: skipped {skipped} mismatched tensors between {names[0]} and {names[1]}")
    a = torch.cat(a_chunks)
    b = torch.cat(b_chunks)
    # Avoid zero-norm vectors.
    a_norm = a.norm(p=2)
    b_norm = b.norm(p=2)
    if a_norm.item() == 0 or b_norm.item() == 0:
        logger.warning(f"cosine_similarity: zero-norm vector for {names}")
        return float("nan")
    cos = torch.dot(a, b) / (a_norm * b_norm)
    val = cos.item()
    logger.info(f"cosine_{names[0]}_{names[1]}: {val:.6f}")
    if wandb is not None:
        wandb.log({f"analysis/cosine/{names[0]}_vs_{names[1]}": val})
    return val


def run_eval_mode(args, device):
    """Execute full evaluation workflow using already parsed args and selected device."""
    model_a, model_b, model_a_ft, model_b_ft, preprocess_A, preprocess_B = get_models(
        args, device)

    mod_openclip_a = OpenCLIPModel(model_a).clip_model
    mod_openclip_a_ft = OpenCLIPModel(model_a_ft).clip_model
    mod_openclip_b = OpenCLIPModel(model_b).clip_model
    mod_openclip_b_ft = OpenCLIPModel(model_b_ft).clip_model

    train_loader_A, test_loader_A, val_loader_A, train_dataset_A, test_dataset_A, val_dataset_A, support_loader_A, support_dataset_A = load_dataset(
        args, preprocess_A, validation=True, support=True)
    train_loader_B, test_loader_B, val_loader_B, train_dataset_B, test_dataset_B, val_dataset_B, support_loader_B, support_dataset_B = load_dataset(
        args, preprocess_B, validation=True, support=True)
    logger.debug(f"Train dataset classes: {train_dataset_A.class_names}")
    logger.debug(f"Test dataset classes: {test_dataset_A.class_names}")

    # Evaluate zero-shot baselines.

    loss_task, acc_task = evaluate_model(
        mod_openclip_b, test_loader_B, test_dataset_B, device, prompt_ensemble=True)
    loss_supp, acc_sup = evaluate_model(
        mod_openclip_b, support_loader_B, support_dataset_B, device, prompt_ensemble=True)
    logger.info(f"Model B ZERO SHOT | TASK : {acc_task}, loss {loss_task}")
    logger.info(f"Model B ZERO SHOT | SUPPORT : {acc_sup}, loss {loss_supp}")
    if wandb is not None:
        wandb.log({
            "baseline/zero_shot/task/accuracy": acc_task,
            "baseline/zero_shot/support/accuracy": acc_sup,
            "baseline/zero_shot/task/loss": loss_task,
            "baseline/zero_shot/support/loss": loss_supp,
        })

    # Build task vectors.
    taskvector_a = TaskVector(mod_openclip_a.visual, mod_openclip_a_ft.visual)
    taskvector_b = TaskVector(mod_openclip_b.visual, mod_openclip_b_ft.visual)

    # Evaluate finetuned model A.
    loss_task_aft, acc_task_aft = evaluate_model(
        mod_openclip_a_ft, test_loader_A, test_dataset_A, device, prompt_ensemble=True)
    loss_supp_aft, acc_sup_aft = evaluate_model(
        mod_openclip_a_ft, support_loader_A, support_dataset_A, device, prompt_ensemble=True)
    logger.info(f"Model A ft | TASK : {acc_task_aft}, loss {loss_task_aft}")
    logger.info(f"Model A ft | SUPPORT : {acc_sup_aft}, loss {loss_supp_aft}")
    if wandb is not None:
        wandb.log({
            "model/A_finetuned/task/accuracy": acc_task_aft,
            "model/A_finetuned/task/loss": loss_task_aft,
            "model/A_finetuned/support/accuracy": acc_sup_aft,
            "model/A_finetuned/support/loss": loss_supp_aft,
        })

    # Evaluate finetuned model B.
    loss_task_bft, acc_task_bft = evaluate_model(
        mod_openclip_b_ft, test_loader_B, test_dataset_B, device, prompt_ensemble=True)
    loss_supp_bft, acc_sup_bft = evaluate_model(
        mod_openclip_b_ft, support_loader_B, support_dataset_B, device, prompt_ensemble=True)
    logger.info(f"Model B ft | TASK : {acc_task_bft}, loss {loss_task_bft}")
    logger.info(f"Model B ft | SUPPORT : {acc_sup_bft}, loss {loss_supp_bft}")
    if wandb is not None:
        wandb.log({
            "model/B_finetuned/task/accuracy": acc_task_bft,
            "model/B_finetuned/task/loss": loss_task_bft,
            "model/B_finetuned/support/accuracy": acc_sup_bft,
            "model/B_finetuned/support/loss": loss_supp_bft,
        })

    tb_signs = {k: torch.sign(v) for k, v in taskvector_b.vector.items()}

    # Determine indices for real-gradient computation.
    imgs_indices = None
    if args.precomputed_indices is not None:
        imgs_indices = load_precomputed_indices(
            args.precomputed_indices, train_dataset_B, device, feature_extractor=model_b.encode_image)
        logger.info(
            f"Using precomputed indices spec '{args.precomputed_indices}'.")
    else:
        # Fallback: random-per-class sampling or dataloader/num_batches mode
        if args.real_imgs_per_class is not None:
            sampler = DatasetSampler(
                train_dataset_B,
                strategy="random",
                device=device,
                seed=args.seed,
            )
            imgs_indices = sampler.sample_indices_per_class(
                k=args.real_imgs_per_class,
                use_precomputed=False,
                save_precomputed=False,
            )
            logger.info(
                f"Sampled {args.real_imgs_per_class} images per class for real gradient signs")
        else:
            imgs_indices = None

    # Build the dedicated dataloader for real-gradient computation.
    realgrad_loader = build_realgrad_dataloader(
        train_dataset_B, train_loader_B, args, imgs_indices)

    # Compute real gradient signs using the prepared dataloader.
    real_gradient_signs, optimized_b = compute_real_gradient_signs(
        pretrained_model=mod_openclip_b,
        dataset=train_dataset_B,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        dataloader=realgrad_loader,
        optimize=getattr(args, "optimize_during_realgrad", False),
        prompt_ensemble=True,
        vote=args.sign_mode
    )

    # Evaluate optimized model B.
    if optimized_b is not None:
        loss_task_opt, acc_task_opt = evaluate_model(
            optimized_b, train_loader_B, train_dataset_B, device, prompt_ensemble=True)
        loss_supp_opt, acc_sup_opt = evaluate_model(
            optimized_b, support_loader_B, support_dataset_B, device, prompt_ensemble=True)
        logger.info(
            f"Model B optimized | TASK : {acc_task_opt}, loss {loss_task_opt}")
        logger.info(
            f"Model B optimized | SUPPORT : {acc_sup_opt}, loss {loss_supp_opt}")
        if wandb is not None:
            wandb.log({
                "model/B_optimized/task/accuracy": acc_task_opt,
                "model/B_optimized/support/accuracy": acc_sup_opt,
                "model/B_optimized/task/loss": loss_task_opt,
                "model/B_optimized/support/loss": loss_supp_opt,
            })

    # Build GradFix task vectors.
    taskvector_gradfix_oracle = taskvector_from_gradient_signs(
        tb_signs, taskvector_a, fallback_taskvector=taskvector_a, mask_mode=args.mask_mode)
    taskvector_gradfix_realgrad = taskvector_from_gradient_signs(
        real_gradient_signs, taskvector_a, fallback_taskvector=taskvector_a, mask_mode=args.mask_mode)

    task_vectors_to_evaluate = [
        ("tau_A_source", taskvector_a),
        ("tau_A_GradFix_oracle", taskvector_gradfix_oracle),
        ("tau_A_GradFix_realgrad", taskvector_gradfix_realgrad),
    ]

    # Evaluate on the test set.
    results = evaluate_task_vectors(
        mod_openclip_b=mod_openclip_b,
        test_dataloader=test_loader_B,
        test_dataset=test_dataset_B,
        device=device,
        alphas=np.linspace(0.1, 1, args.eval_alphas),
        task_vectors_list=task_vectors_to_evaluate,
        logger=logger,
        layers_to_skip=None,
        support_dataloader=support_loader_B,
        support_dataset=support_dataset_B,
    )

    for task, (acc, alpha) in results.items():
        logger.info(f"Test best {task}: {acc} @ alpha={alpha}")


def main():
    args = parse_local_args()
    set_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    run_eval_mode(args, device)


if __name__ == "__main__":
    main()
