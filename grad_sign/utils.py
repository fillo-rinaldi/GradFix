from collections import defaultdict
from copy import deepcopy
import gc
import math
import torch
import numpy as np
import random
import open_clip
from torch.utils.data import Subset
from grad_sign.dataset.eurosat import EuroSat
from grad_sign.dataset.cifar100 import CIFAR100
from grad_sign.dataset.sun397 import SUN397
from grad_sign.dataset.cars import Cars
from grad_sign.dataset.dtd import DTD
from grad_sign.dataset.svhn import SVHN
from grad_sign.dataset.gtsrb import GTSRB
from grad_sign.dataset.resisc45 import RESISC45
from grad_sign.dataset.imagenet_r import IMAGENETR
from grad_sign.dataset.mnist import MNIST
from torch.utils.data import DataLoader
import argparse
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import wandb
import torchvision
from torchvision.transforms import Normalize
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
from sklearn.model_selection import train_test_split
from grad_sign.models import OpenCLIPModel
from grad_sign.gradient_sign_common import taskvector_from_gradient_signs
from task_vectors.src.task_vectors import TaskVector
logger = logging.getLogger(__name__)


class SubsetCustom(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)

    @property
    def targets(self):
        if hasattr(self.dataset, "labels"):   # SVHN uses "labels"
            return np.array(self.dataset.labels)[self.indices]
        elif hasattr(self.dataset, "targets"):  # Other datasets use "targets" (e.g. CIFAR)
            return np.array(self.dataset.targets)[self.indices]
        else:
            return None

    @property
    def class_names(self):
        return getattr(self.dataset, 'class_names', None)

    @property
    def templates(self):
        return getattr(self.dataset, 'templates', None)

    def single_template(self, *args, **kwargs):
        if hasattr(self.dataset, 'single_template'):
            return self.dataset.single_template(*args, **kwargs)
        return None

    def __getattr__(self, attr):
        # Prevent recursion by checking local attributes first.
        if attr in ['dataset', 'indices']:
            return super().__getattribute__(attr)
        # Delegate only selected attributes to the wrapped dataset.
        if attr in ['class_names', 'templates', 'single_template']:
            return getattr(self.dataset, attr)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'")


def create_few_shot_subset(dataset, images_per_class, dataset_name, save_dir="few_shot_indices"):
    """Create a few-shot subset of a dataset.

    Args:
        dataset: PyTorch dataset with (image, label) format
        images_per_class (int): Number of images per class to sample
        dataset_name (str): Name of the dataset for file naming
        save_dir (str): Directory to save/load indices files

    Returns:
        Subset: Subset of the original dataset with few-shot samples
    """

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create filename based on dataset name and images per class
    indices_file = os.path.join(
        save_dir, f"{dataset_name}_indices_{images_per_class}_per_class.pkl")

    # Check if indices file already exists
    if os.path.exists(indices_file):
        logger.info(f"Loading existing indices from {indices_file}")
        with open(indices_file, 'rb') as f:
            sampled_indices = pickle.load(f)
    else:
        logger.info(
            f"Creating new indices for {dataset_name} with {images_per_class} images per class")
        class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)

        sampled_indices = []
        for indices in class_indices.values():
            sampled_indices.extend(random.sample(
                indices, min(images_per_class, len(indices))))

        # Save the indices for future use
        with open(indices_file, 'wb') as f:
            pickle.dump(sampled_indices, f)
        logger.info(f"Saved indices to {indices_file}")

    return SubsetCustom(dataset, sampled_indices)


def build_base_parser():
    """Build the base argument parser without parsing argv.

    This is safe to use as a parent parser (add_help must remain False here).
    """

    parser = argparse.ArgumentParser(
        description='CLIP task vector analysis', add_help=False)
    parser.add_argument('--seed', default=33, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--arch', default='ViT-B-16',
                        type=str, help='Model architecture.')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for evaluation.')
    parser.add_argument('--workers', default=2, type=int,
                        help='Number of data loading workers.')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='Compute device to use.')
    parser.add_argument('--pretraining_backbone_A', default='datacomp_xl_s13b_b90k',
                        type=str, help='Pretraining model A for backbone1.')
    parser.add_argument('--pretraining_backbone_B', default='laion2b_s34b_b88k',
                        type=str, help='Pretraining model B for backbone2.')
    parser.add_argument('--base_folder', default="/work/debiasing/gradientSignData",
                        type=str, help='Path of base folder')
    parser.add_argument('--wandb_mode', default='online', type=str,
                        choices=['online', 'offline', 'disabled'], help='Wandb mode')
    parser.add_argument('--dataset', default='eurosat',
                        type=str, help='Dataset to use')
    parser.add_argument(
        '--wandb_group', default='dataset distillation', type=str, help='Wandb group name')
    # gradient-sign evaluation (common)
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--real_imgs_per_class',
        type=int,
        default=None,
        help=(
            'Randomly sample this many images per class from the training set to compute real '
            'gradient signs (mutually exclusive with --precomputed_indices and --num_batches).'
        ),
    )
    group.add_argument(
        '--precomputed_indices',
        type=str,
        default=None,
        help=(
            "Either a .pkl path or a spec 'coreset:K' | 'herding:K' | 'k-medoid:K'. If given, only "
            'these indices are used (mutually exclusive with --real_imgs_per_class and --num_batches).'
        ),
    )
    group.add_argument(
        '--num_batches',
        type=int,
        default=None,
        help=(
            'Use dataloader mode and backprop this many batches for gradient-sign computation '
            '(mutually exclusive with --real_imgs_per_class and --precomputed_indices).'
        ),
    )
    parser.add_argument(
        '--optimize_during_realgrad',
        action='store_true',
        help='Optimize model B during gradient sign computation.',
    )
    parser.add_argument(
        '--sign_mode',
        type=str,
        default='max',
        choices=['mean', 'max'],
        help='Method to compute gradient signs: mean sign or majority vote.',
    )
    parser.add_argument(
        '--mask_mode',
        type=str,
        default='normal',
        help='How to combine signs with magnitudes when building the task vector.',
    )
    parser.add_argument('--eval_alphas', default=10, type=int,
                        help='Number of alpha values to evaluate')
    parser.add_argument('--finetuned_checkpoint_A', default=None,
                        type=str, help='Path to finetuned model A. If not set, defaults to base_folder/clip-finetuned-weights/<dataset>/<arch>/<pretraining_backbone_A>/best.pt')
    parser.add_argument('--finetuned_checkpoint_B', default=None,
                        type=str, help='Path to finetuned model B. If not set, defaults to base_folder/clip-finetuned-weights/<dataset>/<arch>/<pretraining_backbone_B>/best.pt')

    parser.add_argument('--images_per_class', default=None, type=int,
                        help='Number of images per class for few-shot learning (None=use full dataset)')

    return parser


def parse_arguments():
    """Parse and return command-line arguments for the project using the base parser.

    Returns:
        Tuple[argparse.ArgumentParser, argparse.Namespace]: (parser, args)
    """
    parser = build_base_parser()
    args = parser.parse_args()
    return parser, args


def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set for random number generation.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_with_task_vector(
    base_model,
    task_vector,
    alpha,
    test_dataloader,
    test_dataset,
    device,
    vector_name,
    logger,
    results_list=[],
    best_acc=0,
    best_alpha=0,
    display_name=None,
    layers_to_skip=None
    
):
    """
    Evaluates a model with a given task vector applied at a specific scaling coefficient (alpha).

    Args:
        base_model: Base model to apply the task vector to
        task_vector: TaskVector to apply to the model
        alpha: Scaling coefficient for the task vector
        test_dataloader: DataLoader for test data
        test_dataset: Test dataset
        device: Device to run evaluation on
        vector_name: Name of the task vector (for logging)
        best_acc: Current best accuracy for this vector type
        best_alpha: Current best alpha for this vector type
        results_list: List to append results to
        logger: Logger object for info logging
        display_name: Optional display name (defaults to vector_name if None)

    Returns:
        tuple: (updated best accuracy, updated best alpha)
    """
    if display_name is None:
        display_name = vector_name

    # Save original state dict to restore later (avoid deepcopy)
    original_state_dict = {k: v.clone() for k, v in base_model.visual.state_dict().items()}
    
    # Apply task vector in-place
    task_vector.apply_to(base_model.visual, scaling_coef=alpha, layers_to_skip=layers_to_skip)
    
    # Evaluate
    loss, acc = evaluate_model(
        base_model, test_dataloader, test_dataset, device, prompt_ensemble=True)
    logger.info(
        f"Model B + {display_name:>6} | TASK : {acc:.2%}, loss {loss:.4f} at alpha {alpha:.2f}")

    # Restore original weights
    base_model.visual.load_state_dict(original_state_dict)
    del original_state_dict

    # Track best result
    new_best_acc = best_acc
    new_best_alpha = best_alpha
    if acc > best_acc:
        new_best_acc = acc
        new_best_alpha = alpha

    # Store results
    results_list.append({
        "alpha": alpha,
        "loss": loss,
        "accuracy": acc
    })

    # Log results to wandb
    wandb.log({
        "sweep/alpha": alpha,
        f"sweep/test/{vector_name}/loss": loss,
        f"sweep/test/{vector_name}/accuracy": acc,
    })
    
    # Aggressive cleanup
    gc.collect()
    torch.cuda.empty_cache()
    return new_best_acc, new_best_alpha


def accuracy(output, target, topk=(1, 5)):
    """Compute top-k accuracy.

    Args:
        output (np.ndarray): Model outputs as probabilities or logits (N x C) or binary vector.
        target (np.ndarray): Ground truth labels (N,).
        topk (Tuple[int, ...]): Values of k to compute accuracy for.

    Returns:
        float: Top-1 accuracy (or binary accuracy if output is 1-D).
    """
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = torch.from_numpy(output).topk(maxk, dim=1)
        target = torch.from_numpy(target).view(batch_size, 1).repeat(1, maxk)

        correct = (pred == target)

        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item()
            accuracy /= batch_size  # [0, 1.]
            topk_accuracy.append(accuracy)

        return topk_accuracy[0]


def setup_environment(args):
    """Setup random seeds and compute device from arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.

    Returns:
        str: Device string (e.g., 'cuda:0' or 'cpu').
    """
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Device configuration
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    return device


def get_models(args, device):
    """Create OpenCLIP backbones and load finetuned checkpoints when available.

    Args:
        args (argparse.Namespace): CLI arguments with architecture and checkpoint info.
        device (str): Device to instantiate models on.

    Returns:
        Tuple[open_clip.CLIP, open_clip.CLIP, open_clip.CLIP, open_clip.CLIP, torchvision.Transform]:
        Backbone A, Backbone B, finetuned A, finetuned B, and preprocess transform.
    """
    # Handle HuggingFace models (format: 'hf-hub:timm/model_name')
    if args.pretraining_backbone_A.startswith('hf-hub:'):
        logger.info(f"Loading HuggingFace model A: {args.pretraining_backbone_A}")
        backbone_A, _, preprocess_A = open_clip.create_model_and_transforms(
            args.pretraining_backbone_A,
            cache_dir=f'{args.base_folder}/open_clip',
            device=device
        )
    else:
        logger.info(f"Loading OpenCLIP model A: {args.pretraining_backbone_A}")
        backbone_A, _, preprocess_A = open_clip.create_model_and_transforms(
            args.arch,
            pretrained=args.pretraining_backbone_A,
            cache_dir=f'{args.base_folder}/open_clip',
            device=device
        )
    
    if args.pretraining_backbone_B.startswith('hf-hub:'):
        logger.info(f"Loading HuggingFace model B: {args.pretraining_backbone_B}") 
        backbone_B, _, preprocess_B = open_clip.create_model_and_transforms(
            args.pretraining_backbone_B,
            cache_dir=f'{args.base_folder}/open_clip',
            device=device
        )
    else:
        logger.info(f"Loading OpenCLIP model B: {args.pretraining_backbone_B}")
        backbone_B, _, preprocess_B = open_clip.create_model_and_transforms(
            args.arch,
            pretrained=args.pretraining_backbone_B,
            cache_dir=f'{args.base_folder}/open_clip',
            device=device
        )
    if args.finetuned_checkpoint_A is None:
        if args.pretraining_backbone_A.startswith('hf-hub:'):
            pretraining_backbone_A_sanitized = args.pretraining_backbone_A.split('/')[1]
            args.finetuned_checkpoint_A = f"{args.base_folder}/clip-finetuned-weights/{args.dataset}/{args.arch}/{pretraining_backbone_A_sanitized}/best.pt"
        else:
            args.finetuned_checkpoint_A = f"{args.base_folder}/clip-finetuned-weights/{args.dataset}/{args.arch}/{args.pretraining_backbone_A}/best.pt"
    if args.finetuned_checkpoint_B is None:
        if args.pretraining_backbone_B.startswith('hf-hub:'):
            pretraining_backbone_B_sanitized = args.pretraining_backbone_B.split('/')[1]
            args.finetuned_checkpoint_B = f"{args.base_folder}/clip-finetuned-weights/{args.dataset}/{args.arch}/{pretraining_backbone_B_sanitized}/best.pt"
        else:
            args.finetuned_checkpoint_B = f"{args.base_folder}/clip-finetuned-weights/{args.dataset}/{args.arch}/{args.pretraining_backbone_B}/best.pt"
    try:
        state_dict = torch.load(args.finetuned_checkpoint_A)[
            'model_state_dict']
        model_A_ft = deepcopy(backbone_A)
        model_A_ft.load_state_dict(state_dict)
    except FileNotFoundError:
        logger.warning(
            f"Finetuned checkpoint {args.finetuned_checkpoint_A} not found. Using backbone A as model A ft.")
        model_A_ft = deepcopy(backbone_A)
    try:
        state_dict = torch.load(args.finetuned_checkpoint_B)[
            'model_state_dict']
        model_B_ft = deepcopy(backbone_B)
        model_B_ft.load_state_dict(state_dict)
    except FileNotFoundError:
        logger.warning(
            f"Finetuned checkpoint {args.finetuned_checkpoint_B} not found. Using backbone B as model B ft.")
        model_B_ft = deepcopy(backbone_B)

    return backbone_A, backbone_B, model_A_ft, model_B_ft, preprocess_A, preprocess_B


def get_normalize_mean_std(preprocess):
    """Extract mean and std from a Normalize transform inside preprocess.

    Args:
        preprocess: A torchvision transform pipeline containing Normalize.

    Returns:
        Tuple[Tuple[float, ...], Tuple[float, ...]]: (mean, std)
    """
    for t in preprocess.transforms:
        if isinstance(t, Normalize):
            return (t.mean, t.std)
    raise ValueError("Normalize not found in preprocess")


class SubsetWithAttrs(Subset):
    @property
    def targets(self):
        if hasattr(self.dataset, "labels"):   # SVHN uses "labels"
            return np.array(self.dataset.labels)[self.indices]
        elif hasattr(self.dataset, "targets"):  # Other datasets use "targets" (e.g. CIFAR)
            return np.array(self.dataset.targets)[self.indices]
        else:
            return None

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)


def get_validation_split(dataset, val_ratio=0.2, val_seed=42):
    """Split a dataset into training and validation subsets.

    Args:
        dataset: Dataset to split.
        val_ratio (float): Proportion of data to use for validation.
        val_seed (int): Random seed for reproducibility.

    Returns:
        Tuple[Subset, Subset]: (train_subset, val_subset)
    """
    targets = dataset.targets
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        stratify=targets,
        random_state=val_seed
    )
    train_dataset = SubsetWithAttrs(dataset, train_idx)
    val_dataset = SubsetWithAttrs(dataset, val_idx)
    return train_dataset, val_dataset


def load_dataset(args, preprocess, support=False, validation=False, images_per_class=None, prefix_dataset='datasets'):
    """Load datasets and dataloaders for training/testing or support/target.

    Args:
        args (argparse.Namespace): CLI arguments controlling dataset choice and paths.
        preprocess: Transform to apply to images.
        support (bool): If True, return (target, support) dataloaders/datasets.
        validation (bool): If True, return validation dataloaders/datasets.
        images_per_class (int, optional): If provided, create few-shot subset with this many images per class.
    """
    val_ratio = 0.2
    val_seed = 42
    print("Loading dataset...")
    if args.dataset == 'cifar100':
        train_dataset = CIFAR100(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_dataset = CIFAR100(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            train_dataset, val_dataset = get_validation_split(
                train_dataset, val_ratio=val_ratio, val_seed=val_seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'eurosat':
        test_dataset = EuroSat(
            root=f"{args.base_folder}/{prefix_dataset}/eurosat", split='test', transform=preprocess)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, drop_last=False)
        train_dataset = EuroSat(
            root=f'{args.base_folder}/{prefix_dataset}/eurosat', split='train', transform=preprocess)

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)

        train_loader = DataLoader(
            train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)
        if validation:
            val_dataset = EuroSat(
                root=f"{args.base_folder}/{prefix_dataset}/eurosat", split='val', transform=preprocess)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.workers, pin_memory=True, drop_last=False)
    elif args.dataset == 'sun397':
        train_dataset = SUN397(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_dataset = SUN397(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            train_dataset, val_dataset = get_validation_split(
                train_dataset, val_ratio=val_ratio, val_seed=val_seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'cars':
        train_dataset = Cars(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_dataset = Cars(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            train_dataset, val_dataset = get_validation_split(
                train_dataset, val_ratio=val_ratio, val_seed=val_seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'dtd':
        train_dataset = DTD(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_dataset = DTD(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            val_dataset = DTD(preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}',
                              num_workers=args.workers, batch_size=args.batch_size, validation=True)
            val_loader = val_dataset.val_loader
    elif args.dataset == 'svhn':
        train_dataset = SVHN(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_dataset = SVHN(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            train_dataset, val_dataset = get_validation_split(
                train_dataset, val_ratio=val_ratio, val_seed=val_seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'gtsrb':
        train_dataset = GTSRB(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_dataset = GTSRB(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            train_dataset, val_dataset = get_validation_split(
                train_dataset, val_ratio=val_ratio, val_seed=val_seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'resisc45':
        train_dataset = RESISC45(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_dataset = RESISC45(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            train_dataset, val_dataset = get_validation_split(
                train_dataset, val_ratio=val_ratio, val_seed=val_seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    elif args.dataset == 'mnist':
        train_dataset = MNIST(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        train_loader = train_dataset.train_loader
        test_dataset = MNIST(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers, batch_size=args.batch_size)
        test_loader = test_dataset.test_loader

        # Apply few-shot subset if requested
        if images_per_class is not None:
            train_dataset = create_few_shot_subset(
                train_dataset, images_per_class, args.dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

        if validation:
            train_dataset, val_dataset = get_validation_split(
                train_dataset, val_ratio=val_ratio, val_seed=val_seed)
            train_loader = DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")

    if support:
        support_dataset = IMAGENETR(
            preprocess=preprocess, location=f'{args.base_folder}/{prefix_dataset}', num_workers=args.workers)
        support_loader = support_dataset.test_loader
        print(f'Number of support samples: {len(support_loader.dataset)}')
        print(f'Number of train samples: {len(train_loader.dataset)}')
        print(f'Number of test samples: {len(test_loader.dataset)}')
        if validation:
            print(f'Number of val samples: {len(val_loader.dataset)}')
            return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset, support_loader, support_dataset
        else:
            return train_loader, test_loader, None,  train_dataset, test_dataset, None, support_loader, support_dataset
    else:
        print(f'Number of train samples: {len(train_loader.dataset)}')
        print(f'Number of test samples: {len(test_loader.dataset)}')
        if validation:
            print(f'Number of val samples: {len(val_loader.dataset)}')
            return train_loader, test_loader, val_loader, train_dataset, test_dataset, val_dataset, None, None
        else:
            return train_loader, test_loader, None, train_dataset, test_dataset, None, None, None


def evaluate_model(model, dataloader, dataset, device='cuda:0', prompt_ensemble=True, first_n_batches=None, disable_bar=True):
    """Evaluate a CLIP-like model on a dataset.

    Args:
        model: Model exposing encode_image/encode_text or get_logits.
        dataloader (DataLoader): Dataloader for evaluation.
        dataset: Dataset object with templates and class_names.
        device (str): Device string.
        prompt_ensemble (bool): If True, average over all templates; else use single template.
        first_n_batches (int | None): If set, evaluate only first N batches.
        disable_bar (bool): Disable tqdm progress bar.

    Returns:
        Tuple[float, float]: (average loss, top-1 accuracy).
    """

    eval_avg_loss = 0
    all_probs = []
    all_labels = []
    ce_loss = nn.CrossEntropyLoss()

    model.eval()

    if prompt_ensemble:
        prompts = [[template(c.lower()) for c in dataset.class_names]
                   for template in dataset.templates]
        with torch.no_grad():
            template_embeddings = []
            for template in prompts:
                test_texts = open_clip.tokenize(template)
                test_texts = test_texts.to(device)
                test_text_features = F.normalize(
                    model.encode_text(test_texts), dim=-1)
                template_embeddings.append(test_text_features)

            text_features = torch.mean(torch.stack(template_embeddings), dim=0)
    else:
        prompts = [dataset.single_template(c.lower())
                   for c in dataset.class_names]

        with torch.no_grad():
            test_texts = open_clip.tokenize(prompts)
            test_texts = test_texts.to(device)
            text_features = F.normalize(model.encode_text(test_texts), dim=-1)
    for id, batch in tqdm(enumerate(dataloader), disable=disable_bar):
        if first_n_batches is not None:
            if id == first_n_batches:
                break
        images, targets = batch

        images = images.to(device)

        targets = targets.to(device)

        targets = targets.long()  # Ensure label dtype is compatible (e.g. RESISC45).

        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            image_features = F.normalize(model.encode_image(images), dim=-1)
            vl_logits = model.logit_scale.exp() * (torch.einsum('ij,cj->ic',
                                                                image_features, text_features))

        vl_prob = torch.softmax(vl_logits.float(), dim=-1)

        all_probs.append(vl_prob.cpu().numpy())
        all_labels.append(targets.cpu().numpy())
        loss = ce_loss(vl_logits, targets)

        eval_avg_loss += loss.item()

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    eval_avg_loss /= len(dataloader)

    overall_acc = accuracy(all_probs, all_labels, topk=(1,))
    return eval_avg_loss, overall_acc


def evaluate_target_and_support(model, dataloaders: list, datasets: list, device, prompt_ensemble=True) -> dict:
    results = {}
    for dataloader, dataset in zip(dataloaders, datasets):
        loss, accuracy = evaluate_model(
            model, dataloader, dataset, device, prompt_ensemble)
        results[dataset.__class__.__name__] = (loss, accuracy)
    return results


def evaluate_task_vectors(
    mod_openclip_b,
    test_dataloader,
    test_dataset,
    device,
    alphas,
    task_vectors_list: List[Tuple[str, TaskVector]],
    logger,
    metric_prefix="",
    layers_to_skip=None,
    support_dataloader=None,
    support_dataset=None,
):
    """Evaluate a list of task vectors and log results to W&B."""
    best_accs = {name: 0 for name, _ in task_vectors_list}
    best_alphas = {name: None for name, _ in task_vectors_list}
    results_lists = {name: [] for name, _ in task_vectors_list}
    support_best_accs = {name: 0 for name, _ in task_vectors_list}
    support_best_alphas = {name: None for name, _ in task_vectors_list}
    support_results_lists = {name: [] for name, _ in task_vectors_list}

    for alpha in alphas:
        logger.info(f"{alpha=}")

        for display_name, task_vector in task_vectors_list:
            current_best_acc = best_accs[display_name]
            current_best_alpha = best_alphas[display_name]

            new_best_acc, new_best_alpha = evaluate_with_task_vector(
                base_model=mod_openclip_b,
                task_vector=task_vector,
                alpha=alpha,
                test_dataloader=test_dataloader,
                test_dataset=test_dataset,
                device=device,
                vector_name=display_name,
                best_acc=current_best_acc,
                best_alpha=current_best_alpha,
                results_list=results_lists[display_name],
                logger=logger,
                display_name=display_name,
                layers_to_skip=layers_to_skip,
            )

            best_accs[display_name] = new_best_acc
            best_alphas[display_name] = new_best_alpha

            if support_dataloader is not None and support_dataset is not None:
                support_best_acc, support_best_alpha = evaluate_with_task_vector(
                    base_model=mod_openclip_b,
                    task_vector=task_vector,
                    alpha=alpha,
                    test_dataloader=support_dataloader,
                    test_dataset=support_dataset,
                    device=device,
                    vector_name=f"{metric_prefix}support_{display_name}",
                    best_acc=support_best_accs[display_name],
                    best_alpha=support_best_alphas[display_name],
                    results_list=support_results_lists[display_name],
                    logger=logger,
                    display_name=f"Support {display_name}",
                    layers_to_skip=layers_to_skip,
                )
                support_best_accs[display_name] = support_best_acc
                support_best_alphas[display_name] = support_best_alpha

    all_best_accs = [(acc, best_alphas[name], name) for name, acc in best_accs.items()]
    best_overall = max(all_best_accs, key=lambda x: x[0])
    best_overall_acc, best_overall_alpha, best_method = best_overall

    if wandb is not None:
        summary_log = {}
        for name, acc in best_accs.items():
            summary_log[f"summary/best/{name}/accuracy"] = acc
            summary_log[f"summary/best/{name}/alpha"] = best_alphas[name]

        summary_log.update(
            {
                "summary/best_overall/alpha": best_overall_alpha,
                "summary/best_overall/accuracy": best_overall_acc,
                "summary/best_overall/vector": best_method,
            }
        )
        wandb.log(summary_log)

    return {name: (acc, best_alphas[name]) for name, acc in best_accs.items()}


def load_taskvector_from_checkpoint(
    checkpoint_path: str | Path,
    args,
    map_location: str = "cpu",
    pretraining_backbone_a: str | None = None,
) -> TaskVector:
    """Load a TaskVector from a finetuned checkpoint containing model_state_dict."""
    checkpoint = torch.load(str(checkpoint_path), map_location=map_location)

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} does not contain 'model_state_dict'."
        )

    model_state = checkpoint["model_state_dict"]
    local_args = deepcopy(args)
    if pretraining_backbone_a is not None:
        local_args.pretraining_backbone_A = pretraining_backbone_a

    reference_model, _, _ = open_clip.create_model_and_transforms(
        local_args.arch,
        pretrained=local_args.pretraining_backbone_A,
        cache_dir=f"{local_args.base_folder}/open_clip",
        device=map_location,
    )

    finetuned_model = deepcopy(reference_model)
    finetuned_model.load_state_dict(model_state, strict=True)

    reference_model = OpenCLIPModel(reference_model).clip_model
    finetuned_model = OpenCLIPModel(finetuned_model).clip_model
    return TaskVector(reference_model.visual, finetuned_model.visual)


def _trim_tensor_by_fraction(tensor: torch.Tensor, fraction: float) -> torch.Tensor:
    """Keep only the top-|.| fraction entries in magnitude."""
    if tensor.numel() == 0 or fraction <= 0:
        return torch.zeros_like(tensor)
    if fraction >= 1.0:
        return tensor

    flat = tensor.reshape(-1)
    abs_flat = flat.abs()
    k = max(1, int(math.ceil(abs_flat.numel() * fraction)))
    if k >= abs_flat.numel():
        return tensor

    threshold = torch.topk(abs_flat, k).values.min()
    mask = abs_flat >= threshold
    trimmed = torch.zeros_like(flat)
    trimmed[mask] = flat[mask]
    return trimmed.view_as(tensor)


def merge_taskvector_list(
    task_vectors: Sequence[TaskVector],
    method: str = "mean",
    ties_fraction: float = 0.2,
) -> TaskVector:
    """Merge a list of TaskVectors with mean or TIES."""
    if not task_vectors:
        raise ValueError("No task vectors provided for merging.")

    common_keys = set(task_vectors[0].vector.keys())
    for tv in task_vectors[1:]:
        common_keys &= set(tv.vector.keys())
    if not common_keys:
        raise RuntimeError("Task vectors do not share any common parameter keys.")

    merged_vector = {}
    merge_method = method.lower()

    for key in common_keys:
        reference_tensor = task_vectors[0].vector[key]
        stacked = torch.stack(
            [tv.vector[key].to(torch.float32) for tv in task_vectors], dim=0
        )

        if merge_method == "mean":
            merged = stacked.mean(dim=0)
        elif merge_method == "ties":
            trimmed = torch.stack(
                [
                    _trim_tensor_by_fraction(
                        tv.vector[key].to(torch.float32), ties_fraction
                    )
                    for tv in task_vectors
                ],
                dim=0,
            )

            sign_votes = torch.sign(trimmed)
            pos_votes = (sign_votes > 0).sum(dim=0)
            neg_votes = (sign_votes < 0).sum(dim=0)
            total_votes = pos_votes + neg_votes

            elected_sign = torch.zeros_like(trimmed[0])
            elected_sign[(pos_votes >= neg_votes) & (total_votes > 0)] = 1.0
            elected_sign[(neg_votes > pos_votes) & (total_votes > 0)] = -1.0

            aligned_mask = (
                (torch.sign(trimmed) == elected_sign.unsqueeze(0)) & (trimmed != 0)
            )
            aligned_values = torch.where(aligned_mask, trimmed, torch.zeros_like(trimmed))
            aligned_counts = aligned_mask.sum(dim=0)
            safe_counts = torch.clamp(aligned_counts, min=1)
            mean_aligned = aligned_values.sum(dim=0) / safe_counts
            mean_aligned[aligned_counts == 0] = 0.0
            merged = mean_aligned
        else:
            raise ValueError("Unsupported merge method. Use 'mean' or 'ties'.")

        merged_vector[key] = merged.to(dtype=reference_tensor.dtype)

    return TaskVector(vector=merged_vector)


def sum_gradient_signs(
    gradient_signs_by_source: Dict[str, Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Sum gradient-sign tensors across sources using common keys only."""
    if not gradient_signs_by_source:
        raise ValueError("No gradient signs were provided.")

    names = list(gradient_signs_by_source.keys())
    common_keys = set(gradient_signs_by_source[names[0]].keys())
    for name in names[1:]:
        common_keys &= set(gradient_signs_by_source[name].keys())
    if not common_keys:
        raise RuntimeError("Gradient sign dictionaries do not share common keys.")

    merged = {}
    for key in common_keys:
        base = gradient_signs_by_source[names[0]][key]
        total = torch.zeros_like(base)
        for name in names:
            total = total + gradient_signs_by_source[name][key]
        merged[key] = total
    return merged


def build_merge_mask_strategies(
    taskvectors_by_source: Dict[str, TaskVector],
    gradient_signs_by_source: Dict[str, Dict[str, torch.Tensor]],
    mask_mode: str,
    merge_method: str,
    ties_fraction: float,
    merged_taskvector: TaskVector | None,
):
    """Build merge-then-mask and mask-then-merge task vectors."""
    source_names = [
        name for name in taskvectors_by_source.keys() if name in gradient_signs_by_source
    ]
    if not source_names:
        raise ValueError("No overlapping sources between task vectors and gradient signs.")

    realgrad_taskvectors = {}
    for name in source_names:
        base_tv = taskvectors_by_source[name]
        realgrad_taskvectors[name] = taskvector_from_gradient_signs(
            gradient_signs_by_source[name],
            base_tv,
            fallback_taskvector=base_tv,
            mask_mode=mask_mode,
        )

    mask_then_merge_tv = merge_taskvector_list(
        [realgrad_taskvectors[name] for name in source_names],
        method=merge_method,
        ties_fraction=ties_fraction,
    )

    merged_gradient_signs = sum_gradient_signs(
        {name: gradient_signs_by_source[name] for name in source_names}
    )

    merge_then_mask_tv = None
    if merged_taskvector is not None:
        merge_then_mask_tv = taskvector_from_gradient_signs(
            merged_gradient_signs,
            merged_taskvector,
            fallback_taskvector=merged_taskvector,
            mask_mode=mask_mode,
        )

    return {
        "merge_then_mask_tv": merge_then_mask_tv,
        "mask_then_merge_tv": mask_then_merge_tv,
        "realgrad_taskvectors": realgrad_taskvectors,
        "merged_gradient_signs": merged_gradient_signs,
    }
