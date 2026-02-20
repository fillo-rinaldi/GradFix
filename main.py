import argparse
import logging
import os
import sys

import torch

from grad_sign import gradient_sign_eval
from grad_sign import gradient_sign_merging
from grad_sign import gradient_sign_merging_mix
from grad_sign.utils import set_seed

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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _parse_precomputed(spec: str):
    for sep in (":", ",", "_"):
        if spec and sep in spec:
            method, count = spec.split(sep, 1)
            method = method.strip().lower()
            try:
                return method, int(count)
            except Exception:
                return None, None
    return None, None


def _init_wandb(args, mode_label: str):
    if wandb is None:
        return

    config = {
        "Architecture": getattr(args, "arch", None),
        "Pretraining_A": getattr(args, "pretraining_backbone_A", None),
        "Pretraining_B": getattr(args, "pretraining_backbone_B", None),
        "Base_Folder": getattr(args, "base_folder", None),
        "Dataset": getattr(args, "dataset", None),
        "Real_Imgs_Per_Class": getattr(args, "real_imgs_per_class", None),
        "Num_Batches": getattr(args, "num_batches", None),
        "Seed": getattr(args, "seed", None),
        "Sign_Mode": getattr(args, "sign_mode", None),
        "Mask_Mode": getattr(args, "mask_mode", None),
        "Optimize_During_RealGrad": getattr(args, "optimize_during_realgrad", None),
        "Soft_Beta": getattr(args, "soft_beta", None),
        "Soft_Gamma": getattr(args, "soft_gamma", None),
        "Soft_C": getattr(args, "soft_c", None),
        "Merge_Method": getattr(args, "merge_method", None),
        "Merge_Ties_Fraction": getattr(args, "merge_ties_fraction", None),
        "Mode": mode_label,
    }

    if getattr(args, "precomputed_indices", None):
        method, count = _parse_precomputed(args.precomputed_indices)
        if method is not None and count is not None:
            config["Method"] = method
            config["Real_Imgs_Per_Class"] = count

    wandb.init(
        project="RealGradientEval_grid",
        entity="fillo_rinaldi-unimore",
        config=config,
        mode=getattr(args, "wandb_mode", "disabled"),
        dir=getattr(args, "base_folder", None),
        group=getattr(args, "wandb_group", None),
    )


def main():
    mode_parser = argparse.ArgumentParser(add_help=False)
    mode_parser.add_argument(
        "--mode",
        choices=["eval", "merge", "merge-mix"],
        default="eval",
        help="Run evaluation or merge modes.",
    )
    mode_args, remaining = mode_parser.parse_known_args(sys.argv[1:])

    merge_flags = {
        "--merge_method",
        "--merge_taskvectors",
        "--merge_task_vectors",
        "--merge_ties_fraction",
        "--merge_output",
        "--real_grad_only",
    }
    if mode_args.mode == "eval" and any(flag in remaining for flag in merge_flags):
        logger.error(
            "Merge flags were provided without --mode merge or --mode merge-mix. "
            "Use --mode merge to run merge strategies."
        )
        raise SystemExit(2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if mode_args.mode == "eval":
        args = gradient_sign_eval.parse_local_args(remaining)
        set_seed(args.seed)
        _init_wandb(args, "eval")
        gradient_sign_eval.run_eval_mode(args, device)
        return

    if mode_args.mode == "merge":
        args = gradient_sign_merging.parse_local_args(remaining)
        set_seed(args.seed)
        _init_wandb(args, "merge")
        gradient_sign_merging.run_merge_mode(args, device, logger)
        return

    if mode_args.mode == "merge-mix":
        args = gradient_sign_merging_mix.parse_local_args(remaining)
        set_seed(args.seed)
        _init_wandb(args, "merge-mix")
        gradient_sign_merging_mix.run_merge_mix_mode(args, device, logger)


if __name__ == "__main__":
    main()