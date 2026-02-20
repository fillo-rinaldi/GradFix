import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

from grad_sign.gradient_sign_common import (
    load_precomputed_indices,
    build_realgrad_dataloader,
    compute_real_gradient_signs,
)
from grad_sign.dataset_sampler import DatasetSampler
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

DEFAULT_MERGE_DATASETS = [
    "eurosat",
    "gtsrb",
    "svhn",
    "resisc45",
    "dtd",
    "sun397",
    "mnist",
    "cars",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_local_args(argv=None):
    """Parse merge-local args and merge them into the shared base parser."""
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


def load_dataset_taskvector_entries(
    args,
    dataset_names: Sequence[str] | None = None,
    map_location: str = "cpu",
):
    """Load taskvector entries for the same pretraining across different datasets."""
    datasets = list(DEFAULT_MERGE_DATASETS if dataset_names is None else dataset_names)
    entries = []

    for dataset_name in datasets:
        checkpoint_path = Path(
            args.base_folder,
            "clip-finetuned-weights",
            dataset_name,
            args.arch,
            args.pretraining_backbone_A,
            "best.pt",
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Task vector checkpoint not found: {checkpoint_path}")

        taskvector = load_taskvector_from_checkpoint(
            checkpoint_path=checkpoint_path,
            args=args,
            map_location=map_location,
            pretraining_backbone_a=args.pretraining_backbone_A,
        )
        entries.append(
            {
                "name": dataset_name,
                "taskvector_path": checkpoint_path,
                "taskvector": taskvector,
            }
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


def build_merge_outputs(args, device):
    """Build merge outputs for different datasets with same pretraining."""
    datasets_to_merge = list(DEFAULT_MERGE_DATASETS)
    should_merge = (
        getattr(args, "merge_taskvectors", False) or args.merge_output is not None
    )

    entries = load_dataset_taskvector_entries(
        args,
        dataset_names=datasets_to_merge,
        map_location=device,
    )
    taskvectors_by_source = {entry["name"]: entry["taskvector"] for entry in entries}

    model_a, model_b, model_a_ft, model_b_ft, preprocess_A, preprocess_B = get_models(
        args, device
    )
    mod_openclip_b = OpenCLIPModel(model_b).clip_model

    merged_taskvector = None
    if should_merge:
        merged_taskvector = merge_taskvector_list(
            list(taskvectors_by_source.values()),
            method=args.merge_method,
            ties_fraction=args.merge_ties_fraction,
        )
        logger.info(
            "Merged %d task vectors using '%s'.",
            len(taskvectors_by_source),
            args.merge_method,
        )

        if args.merge_output is not None:
            output_path = Path(args.merge_output)
            torch.save({"task_vector": merged_taskvector.vector}, output_path)
            logger.info("Saved merged task vector to %s", output_path)

    real_gradient_signs = {}
    for ds_name in datasets_to_merge:
        logger.info("Processing dataset '%s' for real gradient signs...", ds_name)
        tmp_args = deepcopy(args)
        tmp_args.dataset = ds_name

        train_loader, _, _, train_dataset, *_ = load_dataset(
            tmp_args,
            preprocess_B,
            validation=True,
            support=True,
        )

        imgs_indices = None
        if args.precomputed_indices is not None:
            imgs_indices = load_precomputed_indices(
                args.precomputed_indices,
                train_dataset,
                device,
                feature_extractor=model_b.encode_image,
            )
            logger.info("Using precomputed indices spec '%s'.", args.precomputed_indices)
        elif args.real_imgs_per_class is not None:
            sampler = DatasetSampler(
                train_dataset,
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
                "Sampled %d images per class for real gradient signs",
                args.real_imgs_per_class,
            )

        realgrad_loader = build_realgrad_dataloader(
            train_dataset,
            train_loader,
            tmp_args,
            imgs_indices,
        )

        real_gradient_signs[ds_name], _ = compute_real_gradient_signs(
            pretrained_model=mod_openclip_b,
            dataset=train_dataset,
            loss_fn=nn.CrossEntropyLoss(),
            device=device,
            dataloader=realgrad_loader,
            optimize=getattr(tmp_args, "optimize_during_realgrad", False),
            prompt_ensemble=True,
            vote=tmp_args.sign_mode,
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
        "datasets": datasets_to_merge,
        "preprocess_B": preprocess_B,
        "base_model": mod_openclip_b,
        "real_gradient_signs": real_gradient_signs,
    }


def run_merge_mode(args, device, logger_instance=None):
    """Execute full merge workflow (merge-only, merge-then-mask, mask-then-merge)."""
    active_logger = logger if logger_instance is None else logger_instance

    if not getattr(args, "merge_taskvectors", False) and args.merge_output is None:
        active_logger.warning(
            "--merge_taskvectors not set; enabling it so both merge strategies run."
        )
        args.merge_taskvectors = True

    outputs = build_merge_outputs(args, device)

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
