import os
import torch
import torchvision.datasets as datasets
from collections import defaultdict
import random
from torch.utils.data import DataLoader, Subset

from .templates import get_templates


class MNIST:
    """MNIST dataset wrapper compatible with CLIP finetuning and activation alignment.
    
    Exposes the same interface as EuroSat, DTD, GTSRB, RESISC45, etc.
    """
    
    @property
    def targets(self):
        """Return labels of all samples in train_dataset."""
        if hasattr(self.train_dataset, 'targets'):
            return list(self.train_dataset.targets)
        return [self.train_dataset[i][1] for i in range(len(self.train_dataset))]

    @property
    def name(self):
        return "mnist"

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
                 download=True):
        """Initialize MNIST dataset.
        
        Args:
            preprocess: Transform to apply to images (should handle grayscale to RGB conversion if needed)
            location: Root directory for data storage
            batch_size: Batch size for dataloaders
            num_workers: Number of dataloader workers
            few_shot: If True, sample up to 10 examples per class
            download: If True, download dataset if not present
        """
        root = location
        
        # MNIST is grayscale, we need to convert to RGB for CLIP
        # Create a wrapper transform that converts grayscale to RGB before applying preprocess
        import torchvision.transforms as transforms
        
        # Convert grayscale to RGB by repeating channels
        grayscale_to_rgb = transforms.Lambda(lambda x: x.convert('RGB'))
        
        # Combine: first convert to RGB, then apply the preprocess
        if preprocess is not None:
            full_transform = transforms.Compose([grayscale_to_rgb, preprocess])
        else:
            full_transform = grayscale_to_rgb

        self.train_dataset = datasets.MNIST(
            root=root,
            train=True,
            transform=full_transform,
            download=download
        )

        if few_shot:
            # Sample up to 10 examples per class
            labels = list(self.train_dataset.targets)
            if isinstance(labels[0], torch.Tensor):
                labels = [label.item() for label in labels]
            
            class_indices = defaultdict(list)
            for idx, label in enumerate(labels):
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

        self.test_dataset = datasets.MNIST(
            root=root,
            train=False,
            transform=full_transform,
            download=download
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )

        # MNIST classes are digits 0-9
        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        # Templates for CLIP text encoder
        self.templates = get_templates('mnist')
        self.single_template = self.test_dataset.single_template = self.train_dataset.single_template = lambda c: f'a photo of the number: "{c}"'
