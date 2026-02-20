import os
import torch
import torchvision.datasets as datasets
from collections import defaultdict
import random
from torch.utils.data import DataLoader, Subset

from .templates import get_templates


class DTD:
    @property
    def targets(self):
        # Return labels for all samples in the training dataset
        if hasattr(self.train_dataset, 'targets'):
            return list(self.train_dataset.targets)
        if hasattr(self.train_dataset, 'labels'):
            return list(self.train_dataset.labels)
        if hasattr(self.train_dataset, 'samples'):
            return [label for _, label in self.train_dataset.samples]
        
        # Fallback: check private _labels which DTD mostly likely has
        if hasattr(self.train_dataset, '_labels'):
             return list(self.train_dataset._labels)
             
        # Fallback: iteration
        return [self.train_dataset[i][1] for i in range(len(self.train_dataset))]

    @property
    def name(self):
        return "dtd"

    def __getitem__(self, idx):
        return self.train_dataset[idx]

    def __len__(self):
        # If few-shot is active and a subset exists, return its length
        if hasattr(self, 'train_dataset_subset'):
            return len(self.train_dataset_subset)
        return len(self.train_dataset)

    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=16,
                 few_shot=False,
                 download=True,
                 validation=False
                 ):
        root = location

        self.train_dataset = datasets.DTD(
            root=root,
            split='train',
            transform=preprocess,
            download=download
        )

        if few_shot:
            # Sample up to 10 examples per class (avoid I/O by using targets/labels)
            if hasattr(self.train_dataset, 'targets'):
                labels = list(self.train_dataset.targets)
            elif hasattr(self.train_dataset, 'labels'):
                labels = list(self.train_dataset.labels)
            else:
                labels = [self.train_dataset[i][1]
                          for i in range(len(self.train_dataset))]

            class_indices = defaultdict(list)
            for idx, label in enumerate(labels):
                class_indices[label].append(idx)

            sampled_indices = []
            for indices in class_indices.values():
                sampled_indices.extend(random.sample(
                    indices, min(10, len(indices))))

            self.train_dataset_subset = Subset(
                self.train_dataset, sampled_indices)
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

        self.test_dataset = datasets.DTD(
            root=root,
            split='test',
            transform=preprocess,
            download=download
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        if validation:
            self.val_dataset = datasets.DTD(
                root=root,
                split='val',
                transform=preprocess,
                download=download
            )
            self.val_loader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                num_workers=num_workers
            )

        if hasattr(self.train_dataset, 'class_to_idx'):
            class_to_idx = self.train_dataset.class_to_idx
        elif hasattr(self.train_dataset, 'classes'):
            class_to_idx = {c: i for i, c in enumerate(
                self.train_dataset.classes)}
        else:
            unique_labels = sorted(
                set(self.targets)) if self.targets is not None else []
            class_to_idx = {str(c): i for i, c in enumerate(unique_labels)}

        idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.class_names = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]

        # Keep templates and single_template aligned with the original implementation.
        self.templates = get_templates('dtd')
        self.single_template = self.test_dataset.single_template = self.train_dataset.single_template = lambda c: f'A photo of a {c}'
