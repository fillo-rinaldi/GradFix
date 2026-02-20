import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from grad_sign.dataset_sampler import DatasetSampler
from grad_sign.gradient_sign_common import (
    load_precomputed_indices,
    build_realgrad_dataloader,
    compute_real_gradient_signs,
)
from grad_sign.utils import (
    evaluate_task_vectors,
    build_merge_mask_strategies,
    load_taskvector_from_checkpoint,
    merge_taskvector_list,
    build_base_parser,
    get_models,
    load_dataset,
)
from grad_sign.models import OpenCLIPModel

DEFAULT_PRETRAINING_BACKBONES = [
    "laion400m_e32",
    "datacomp_xl_s13b_b90k",
    "datacomp_l_s1b_b8k",
    "commonpool_l_laion_s1b_b8k",
    "metaclip_fullcc",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_local_args(argv=None):
    """Parse merge-mix local args and merge them into base parser args."""
    argv = list(sys.argv[1:] if argv is None else argv)

    local = argparse.ArgumentParser(add_help=False)
    local.add_argument("--merge_taskvectors", action="store_true")
    local.add_argument(
        "--merge_method", type=str, default="mean", choices=["mean", "ties"]
    )
    local.add_argument("--merge_ties_fraction", type=float, default=0.2)
    local.add_argument("--merge_output", type=str, default=None)
    local.add_argument("--real_grad_only", action="store_true")

    local_args, remaining = local.parse_known_args(argv)
    base_args = build_base_parser().parse_args(remaining)

    for key, value in vars(local_args).items():
        setattr(base_args, key.replace("-", "_"), value)
    return base_args


def load_pretraining_taskvector_entries(
    args,
    dataset_name: str,
    pretraining_backbones: Sequence[str] | None = None,
    map_location: str = "cpu",
):
    """Load taskvector entries for the same dataset across different pretraining backbones."""
    backbones = list(
        DEFAULT_PRETRAINING_BACKBONES
        if pretraining_backbones is None
        else pretraining_backbones
    )

    entries = []
    for backbone in backbones:
        checkpoint_path = Path(
            args.base_folder,
            "clip-finetuned-weights",
            dataset_name,
            args.arch,
            backbone,
            "best.pt",
        )
        if not checkpoint_path.exists():
            logger.warning(
                "Skipping backbone '%s' because checkpoint %s is missing.",
                backbone,
                checkpoint_path,
            )
            continue

        taskvector = load_taskvector_from_checkpoint(
            checkpoint_path=checkpoint_path,
            args=args,
            map_location=map_location,
            pretraining_backbone_a=backbone,
        )
        entries.append(
            {
                "name": backbone,
                "taskvector_path": checkpoint_path,
                "taskvector": taskvector,
            }
        )

    if not entries:
        raise RuntimeError(
            f"No finetuned checkpoints found for dataset '{dataset_name}' and arch '{args.arch}'."
        )

    return entries


def evaluate_merged_vector_across_datasets(
    merged_taskvector,
    dataset_names,
    args,
    preprocess,
    base_model,
    device,
    logger,
    vector_label: str | None = None,
):
    """Evaluate one merged vector across a list of datasets."""
    if not dataset_names:
        return {}

    results = {}
    effective_label = vector_label or f"tau_merge_{args.merge_method}"

    for ds_name in dataset_names:
        tmp_args = deepcopy(args)
        tmp_args.dataset = ds_name

        _, test_loader, _, _, test_dataset, _, _, _ = load_dataset(
            tmp_args,
            preprocess,
            validation=False,
            support=False,
        )

        eval_res = evaluate_task_vectors(
            mod_openclip_b=base_model,
            test_dataloader=test_loader,
            test_dataset=test_dataset,
            device=device,
            alphas=np.linspace(0.1, 1, tmp_args.eval_alphas),
            task_vectors_list=[(effective_label, merged_taskvector)],
            logger=logger,
        )

        acc, best_alpha = eval_res[effective_label]
        results[ds_name] = {"accuracy": acc, "best_alpha": best_alpha}

    return results


def build_merge_mix_outputs(args, device):
    """Build merge outputs for same dataset across different pretraining backbones."""
    target_dataset = getattr(args, "dataset", None)
    if not target_dataset:
        raise ValueError("A target dataset must be provided via --dataset.")

    should_merge = (
        getattr(args, "merge_taskvectors", False) or args.merge_output is not None
    )

    entries = load_pretraining_taskvector_entries(
        args,
        dataset_name=target_dataset,
        map_location=device,
    )
    taskvectors_by_source = {entry["name"]: entry["taskvector"] for entry in entries}
    source_names = list(taskvectors_by_source.keys())

    logger.info(
        "Evaluating dataset '%s' across %d pretraining backbones: %s",
        target_dataset,
        len(source_names),
        ", ".join(source_names),
    )

    base_args = deepcopy(args)
    base_args.pretraining_backbone_A = source_names[0]
    _, model_b, _, _, _, preprocess_B = get_models(base_args, device)
    mod_openclip_b = OpenCLIPModel(model_b).clip_model

    merged_taskvector = None
    if should_merge:
        merged_taskvector = merge_taskvector_list(
            [taskvectors_by_source[name] for name in source_names],
            method=args.merge_method,
            ties_fraction=args.merge_ties_fraction,
        )
        logger.info(
            "Merged %d task vectors using '%s'.",
            len(source_names),
            args.merge_method,
        )

        if args.merge_output is not None:
            output_path = Path(args.merge_output)
            torch.save({"task_vector": merged_taskvector.vector}, output_path)
            logger.info("Saved merged task vector to %s", output_path)

    real_gradient_signs = {}
    for source_name in source_names:
        logger.info(
            "Processing dataset '%s' with pretraining backbone A '%s'",
            target_dataset,
            source_name,
        )
        pair_args = deepcopy(args)
        pair_args.dataset = target_dataset
        pair_args.pretraining_backbone_A = source_name

        _, model_b_pair, _, _, _, preprocess_B_pair = get_models(pair_args, device)
        mod_openclip_b_pair = OpenCLIPModel(model_b_pair).clip_model

        train_loader, _, _, train_dataset, *_ = load_dataset(
            pair_args,
            preprocess_B_pair,
            validation=True,
            support=True,
        )

        imgs_indices = None
        if pair_args.precomputed_indices is not None:
            imgs_indices = load_precomputed_indices(
                pair_args.precomputed_indices,
                train_dataset,
                device,
                feature_extractor=model_b_pair.encode_image,
            )
            logger.info(
                "Using precomputed indices spec '%s' for backbone '%s'.",
                pair_args.precomputed_indices,
                source_name,
            )
        elif pair_args.real_imgs_per_class is not None:
            sampler = DatasetSampler(
                train_dataset,
                strategy="random",
                device=device,
                seed=pair_args.seed,
            )
            imgs_indices = sampler.sample_indices_per_class(
                k=pair_args.real_imgs_per_class,
                use_precomputed=False,
                save_precomputed=False,
            )
            logger.info(
                "Sampled %d images per class for backbone '%s'",
                pair_args.real_imgs_per_class,
                source_name,
            )

        realgrad_loader = build_realgrad_dataloader(
            train_dataset,
            train_loader,
            pair_args,
            imgs_indices,
        )

        real_gradient_signs[source_name], _ = compute_real_gradient_signs(
            pretrained_model=mod_openclip_b_pair,
            dataset=train_dataset,
            loss_fn=nn.CrossEntropyLoss(),
            device=device,
            dataloader=realgrad_loader,
            optimize=getattr(pair_args, "optimize_during_realgrad", False),
            prompt_ensemble=True,
            vote=pair_args.sign_mode,
            show_progress=True,
        )

    strategy_outputs = build_merge_mask_strategies(
        taskvectors_by_source=taskvectors_by_source,
        gradient_signs_by_source=real_gradient_signs,
        mask_mode=args.mask_mode,
        merge_method=args.merge_method,
        ties_fraction=args.merge_ties_fraction,
        merged_taskvector=merged_taskvector,
    )

    return {
        "merged_taskvector": merged_taskvector,
        "merge_then_mask_tv": strategy_outputs["merge_then_mask_tv"],
        "mask_then_merge_tv": strategy_outputs["mask_then_merge_tv"],
        "datasets": [target_dataset],
        "preprocess_B": preprocess_B,
        "base_model": mod_openclip_b,
        "real_gradient_signs": real_gradient_signs,
    }


def run_merge_mix_mode(args, device, logger_instance=None):
    """Execute full merge-mix workflow (same dataset, different pretrainings)."""
    active_logger = logger if logger_instance is None else logger_instance

    if not getattr(args, "merge_taskvectors", False) and args.merge_output is None:
        active_logger.warning(
            "--merge_taskvectors not set; enabling it so merge-only and merge-then-mask run."
        )
        args.merge_taskvectors = True

    outputs = build_merge_mix_outputs(args, device)

    if outputs["merged_taskvector"] is not None and not args.real_grad_only:
        evaluate_merged_vector_across_datasets(
            merged_taskvector=outputs["merged_taskvector"],
            dataset_names=outputs["datasets"],
            args=args,
            preprocess=outputs["preprocess_B"],
            base_model=outputs["base_model"],
            device=device,
            logger=active_logger,
            vector_label="tau_merge_task_arithmetic",
        )

    if outputs["merge_then_mask_tv"] is not None:
        evaluate_merged_vector_across_datasets(
            merged_taskvector=outputs["merge_then_mask_tv"],
            dataset_names=outputs["datasets"],
            args=args,
            preprocess=outputs["preprocess_B"],
            base_model=outputs["base_model"],
            device=device,
            logger=active_logger,
            vector_label="tau_merge_then_GradFix",
        )

    if outputs["mask_then_merge_tv"] is not None:
        evaluate_merged_vector_across_datasets(
            merged_taskvector=outputs["mask_then_merge_tv"],
            dataset_names=outputs["datasets"],
            args=args,
            preprocess=outputs["preprocess_B"],
            base_model=outputs["base_model"],
            device=device,
            logger=active_logger,
            vector_label="tau_GradFix_then_merge",
        )
