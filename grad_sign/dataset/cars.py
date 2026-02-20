"""Stanford Cars dataset implementation using ImageFolder format.

Expected directory structure:
    {location}/stanford_cars/
        train/<class_name>/*.jpg
        test/<class_name>/*.jpg
"""

import os
import torch
import torchvision.datasets as datasets
import pathlib
from collections import defaultdict
import random

from torch.utils.data import DataLoader, Subset

from .templates import get_templates


class Cars:
    """Stanford Cars dataset wrapper for CLIP finetuning and activation alignment.
    
    Exposes the same interface as EuroSat, DTD, GTSRB, RESISC45, etc.
    """

    @property
    def targets(self):
        """Return labels of all samples in train_dataset."""
        if hasattr(self.train_dataset, 'targets'):
            return list(self.train_dataset.targets)
        if hasattr(self.train_dataset, 'samples'):
            return [label for _, label in self.train_dataset.samples]
        return None

    @property
    def name(self):
        return "cars"

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def __len__(self):
        if hasattr(self, 'train_dataset_subset'):
            return len(self.train_dataset_subset)
        return len(self.train_dataset)

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 few_shot=False,
                 download=False):
        """Initialize Stanford Cars dataset.
        
        Args:
            preprocess: Transform to apply to images
            location: Root directory for data storage
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            few_shot: If True, sample up to 10 examples per class
            download: Ignored, kept for API compatibility
        """
        train_folder = pathlib.Path(location) / "stanford_cars" / "train"
        test_folder = pathlib.Path(location) / "stanford_cars" / "test"
        
        if not train_folder.is_dir() or not test_folder.is_dir():
            raise RuntimeError(
                f"Stanford Cars dataset not found at {location}/stanford_cars/. "
                "Expected structure: stanford_cars/train/<class>/*.jpg and stanford_cars/test/<class>/*.jpg"
            )

        self.train_dataset = datasets.ImageFolder(str(train_folder), transform=preprocess)
        
        if few_shot:
            class_indices = defaultdict(list)
            for idx, (_, label) in enumerate(self.train_dataset.samples):
                class_indices[label].append(idx)

            sampled_indices = []
            for indices in class_indices.values():
                sampled_indices.extend(random.sample(indices, min(10, len(indices))))

            self.train_dataset_subset = Subset(self.train_dataset, sampled_indices)
            self.train_loader = DataLoader(
                self.train_dataset_subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )
        else:
            self.train_loader = DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=num_workers
            )

        self.test_dataset = datasets.ImageFolder(str(test_folder), transform=preprocess)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # Build class names
        idx_to_class = {v: k for k, v in self.train_dataset.class_to_idx.items()}
        self.class_names = [idx_to_class[i].replace('_', ' ') for i in range(len(idx_to_class))]
        self.classnames = self.class_names  # backward compatibility

        # Templates for CLIP text encoder
        self.templates = get_templates('cars')
        self.single_template = self.test_dataset.single_template = self.train_dataset.single_template = lambda c: f'a photo of a {c}'
