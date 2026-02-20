"""
DatasetSampler: Unified class for dataset sampling strategies.

Supports: random, coreset, k-medoid, herding sampling methods.
Extracted from synth_image.py and gradient_sign_eval.py for reusability.
"""

import logging
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import pairwise_distances

logger = logging.getLogger(__name__)

# Base path for precomputed indices (.pkl)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRECOMPUTED_DIR = os.path.join(BASE_DIR, 'precomputed_indices')


class DatasetSampler:
    """
    Unified class for sampling indices from a dataset using various strategies.
    
    Supported strategies:
        - 'random': Random sampling per class
        - 'coreset': Coreset selection using greedy medoid approach
        - 'k-medoid': K-medoids clustering (exact or fast MiniBatchKMeans)
        - 'herding': Herding selection (closest to class mean)
    
    Usage:
        sampler = DatasetSampler(dataset, strategy='random', device='cuda')
        indices_dict = sampler.sample_indices_per_class(k=10)
        # indices_dict: {class_id -> list[int]}
    """
    
    def __init__(
        self,
        dataset,
        strategy: str = 'random',
        device: str = 'cuda',
        feature_extractor: Optional[nn.Module] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the DatasetSampler.
        
        Args:
            dataset: PyTorch dataset with (image, label) items.
                     Must have `class_names` attribute for num_classes detection,
                     or `targets` / labels accessible via iteration.
            strategy: Sampling strategy. One of 'random', 'coreset', 'k-medoid', 'herding'.
            device: Device for feature extraction ('cuda' or 'cpu').
            feature_extractor: Pre-trained model for extracting features.
                               If None and strategy requires features, ResNet18 is used.
            seed: Optional random seed for reproducibility.
        """
        self.dataset = dataset
        self.strategy = strategy.lower()
        self.device = device
        self.seed = seed
        
        # Validate strategy
        valid_strategies = ['random', 'coreset', 'k-medoid', 'herding']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'. Must be one of {valid_strategies}")
        
        # Build class indices mapping: {class_id -> list[dataset_idx]}
        self.class_indices = self._build_class_indices()
        self.num_classes = len(self.class_indices)
        
        # Initialize feature extractor for strategies that need it
        self.feature_extractor = None
        if self.strategy in ['coreset', 'k-medoid', 'herding']:
            if feature_extractor is None:
                logger.info("Initializing default ResNet18 feature extractor...")
                self.feature_extractor = models.resnet18(pretrained=True)
                self.feature_extractor.fc = nn.Identity()  # Remove final classification layer
                self.feature_extractor.to(device)
                self.feature_extractor.eval()
            else:
                self.feature_extractor = feature_extractor
    
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """Build mapping from class_id to list of dataset indices."""
        class_indices = defaultdict(list)
        
        # Try to use .targets if available (faster)
        if hasattr(self.dataset, 'targets'):
            for idx, target in enumerate(self.dataset.targets):
                class_indices[int(target)].append(idx)
        else:
            # Fall back to iterating through dataset
            for idx in range(len(self.dataset)):
                _, label = self.dataset[idx]
                class_indices[int(label)].append(idx)
        
        return dict(class_indices)
    
    def sample_indices_per_class(
        self,
        k: int,
        use_precomputed: bool = True,
        save_precomputed: bool = True,
        fast_kmedoid: bool = True,
        kmedoid_threshold: int = 2000,
    ) -> Dict[int, List[int]]:
        """
        Sample k indices per class using the configured strategy.
        
        Args:
            k: Number of samples per class.
            use_precomputed: Whether to load from precomputed pickle if available.
            save_precomputed: Whether to save computed indices to pickle.
            fast_kmedoid: Use fast MiniBatchKMeans for k-medoid (only for 'k-medoid').
            kmedoid_threshold: Threshold for switching to fast k-medoid.
        
        Returns:
            Dict[int, List[int]]: Mapping from class_id to list of selected indices.
        """
        # Special-case: k == -1 -> return all indices for each class
        if int(k) == -1:
            logger.info("Requested k=-1: returning full dataset indices for each class")
            return {cls: idxs.copy() for cls, idxs in self.class_indices.items()}

        if self.strategy == 'random':
            return self._sample_random(k)
        elif self.strategy == 'coreset':
            return self._sample_coreset(k, use_precomputed, save_precomputed)
        elif self.strategy == 'k-medoid':
            return self._sample_kmedoid(k, use_precomputed, save_precomputed, fast_kmedoid, kmedoid_threshold)
        elif self.strategy == 'herding':
            return self._sample_herding(k, use_precomputed, save_precomputed)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_flat_indices(self, indices_dict: Dict[int, List[int]]) -> List[int]:
        """Flatten indices dict to a single list of indices."""
        flat = []
        for cls in sorted(indices_dict.keys()):
            flat.extend(indices_dict[cls])
        return flat
    
    def _sample_random(self, k: int) -> Dict[int, List[int]]:
        """Random sampling of k indices per class."""
        if self.seed is not None:
            random.seed(self.seed)
        
        sampled = {}
        for cls, idxs in self.class_indices.items():
            if len(idxs) < k:
                logger.warning(f"Class {cls} has only {len(idxs)} samples, requested {k}. Using all.")
                sampled[cls] = idxs.copy()
            else:
                sampled[cls] = random.sample(idxs, k)
        return sampled
    
    def _get_precomputed_path(self, method: str, k: int) -> str:
        """Get path for precomputed indices pickle."""
        dataset_name = getattr(self.dataset, 'name', 'unknown_dataset')
        filename = f'{dataset_name}_{method}_indices_{k}.pkl'
        return os.path.join(PRECOMPUTED_DIR, filename)
    
    def _load_precomputed(self, path: str) -> Optional[Dict[int, List[int]]]:
        """Load precomputed indices from pickle file."""
        if os.path.exists(path):
            logger.info(f"Loading precomputed indices from {path}")
            with open(path, 'rb') as f:
                data = pickle.load(f)
            # Normalize keys to int
            return {int(k): list(map(int, v)) for k, v in data.items()}
        return None
    
    def _save_precomputed(self, path: str, indices: Dict[int, List[int]]):
        """Save computed indices to pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(indices, f)
        logger.info(f"Saved precomputed indices to {path}")
    
    def _extract_features_for_class(self, class_id: int) -> np.ndarray:
        """Extract normalized features for all images in a class."""
        class_idxs = self.class_indices[class_id]
        features = []
        
        with torch.no_grad():
            batch_size = 64
            for i in range(0, len(class_idxs), batch_size):
                batch_indices = class_idxs[i:i + batch_size]
                batch_images = torch.stack([self.dataset[idx][0] for idx in batch_indices])
                batch_images = batch_images.to(self.device)
                
                batch_features = self.feature_extractor(batch_images)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                features.append(batch_features.cpu().numpy())
        
        return np.concatenate(features, axis=0)
    
    def _sample_coreset(
        self, k: int, use_precomputed: bool, save_precomputed: bool
    ) -> Dict[int, List[int]]:
        """Coreset selection using greedy medoid approach."""
        precomputed_path = self._get_precomputed_path('coreset', k)
        
        if use_precomputed:
            loaded = self._load_precomputed(precomputed_path)
            if loaded is not None:
                return loaded
        
        sampled = {}
        for class_id in range(self.num_classes):
            indices = self._coreset_for_class(k, class_id)
            sampled[class_id] = indices
        
        if save_precomputed:
            self._save_precomputed(precomputed_path, sampled)
        
        return sampled
    
    def _coreset_for_class(self, n: int, class_id: int) -> List[int]:
        """Coreset selection for a single class."""
        class_indices = self.class_indices[class_id]
        
        if n >= len(class_indices):
            return class_indices.copy()
        
        features = self._extract_features_for_class(class_id)
        distances = pairwise_distances(features, metric='euclidean')
        
        selected_indices = []
        remaining_indices = list(range(len(class_indices)))
        
        # First, select the point that minimizes the sum of distances to all other points
        sum_distances = distances.sum(axis=1)
        first_medoid = int(np.argmin(sum_distances))
        selected_indices.append(first_medoid)
        remaining_indices.remove(first_medoid)
        
        # Iteratively select the next medoid
        for _ in range(n - 1):
            if not remaining_indices:
                break
            
            best_candidate = None
            best_score = float('inf')
            
            for candidate in remaining_indices:
                min_dist_to_selected = min([distances[candidate][s] for s in selected_indices])
                if min_dist_to_selected < best_score:
                    best_score = min_dist_to_selected
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
        
        return [class_indices[i] for i in selected_indices]
    
    def _sample_kmedoid(
        self, k: int, use_precomputed: bool, save_precomputed: bool,
        fast: bool, threshold: int
    ) -> Dict[int, List[int]]:
        """K-medoid clustering selection."""
        precomputed_path = self._get_precomputed_path('medoids', k)
        
        if use_precomputed:
            loaded = self._load_precomputed(precomputed_path)
            if loaded is not None:
                return loaded
        
        sampled = {}
        for class_id in range(self.num_classes):
            indices = self._kmedoid_for_class(k, class_id, fast, threshold)
            sampled[class_id] = indices
        
        if save_precomputed:
            self._save_precomputed(precomputed_path, sampled)
        
        return sampled
    
    def _kmedoid_for_class(
        self, n: int, class_id: int, fast: bool, threshold: int
    ) -> List[int]:
        """K-medoid selection for a single class."""
        class_indices = self.class_indices[class_id]
        
        if n >= len(class_indices):
            return class_indices.copy()
        
        features = self._extract_features_for_class(class_id).astype(np.float32)
        num_samples = features.shape[0]
        use_fast = fast or (num_samples > threshold)
        
        if use_fast:
            try:
                from sklearn.cluster import MiniBatchKMeans
                kmeans = MiniBatchKMeans(
                    n_clusters=n,
                    batch_size=2048,
                    max_iter=100,
                    n_init="auto",
                    random_state=42
                )
                kmeans.fit(features)
                centers = kmeans.cluster_centers_.astype(np.float32)
                centers /= (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-12)
                sims = centers @ features.T
                
                selected = []
                used = set()
                for row in sims:
                    order = np.argsort(-row)
                    for idx in order:
                        idx = int(idx)
                        if idx not in used:
                            used.add(idx)
                            selected.append(idx)
                            break
                
                if len(selected) < n:
                    flat_order = np.argsort(-sims.ravel())
                    for flat_idx in flat_order:
                        cand = int(flat_idx % num_samples)
                        if cand not in used:
                            used.add(cand)
                            selected.append(cand)
                        if len(selected) == n:
                            break
                
                return [class_indices[i] for i in selected[:n]]
            except Exception as e:
                logger.warning(f"Fast k-medoids fallback to exact due to: {e}")
        
        # Exact k-medoids
        if self.seed is not None:
            np.random.seed(self.seed)
        else:
            np.random.seed(42)
        
        medoid_indices = np.random.choice(num_samples, size=n, replace=False)
        max_iterations = 100
        tolerance = 1e-6
        best_cost = None
        
        for _ in range(max_iterations):
            medoid_feats = features[medoid_indices]
            distances_to_medoids = np.linalg.norm(
                features[:, None, :] - medoid_feats[None, :, :], axis=2
            )
            assignments = np.argmin(distances_to_medoids, axis=1)
            current_cost = np.sum(distances_to_medoids[np.arange(num_samples), assignments])
            
            new_medoid_indices = medoid_indices.copy()
            improved = False
            
            for cluster_id in range(n):
                cluster_points = np.where(assignments == cluster_id)[0]
                if len(cluster_points) == 0:
                    continue
                sub = features[cluster_points]
                dists = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=2)
                costs = dists.sum(axis=1)
                best_idx_local = int(np.argmin(costs))
                best_medoid_local = cluster_points[best_idx_local]
                if best_medoid_local != medoid_indices[cluster_id]:
                    new_medoid_indices[cluster_id] = best_medoid_local
                    improved = True
            
            if not improved or (best_cost is not None and abs(current_cost - best_cost) < tolerance):
                break
            medoid_indices = new_medoid_indices
            best_cost = current_cost
        
        return [class_indices[i] for i in medoid_indices]
    
    def _sample_herding(
        self, k: int, use_precomputed: bool, save_precomputed: bool
    ) -> Dict[int, List[int]]:
        """Herding selection (closest to class mean)."""
        precomputed_path = self._get_precomputed_path('herding', k)
        
        if use_precomputed:
            loaded = self._load_precomputed(precomputed_path)
            if loaded is not None:
                return loaded
        
        sampled = {}
        for class_id in range(self.num_classes):
            indices = self._herding_for_class(k, class_id)
            sampled[class_id] = indices
        
        if save_precomputed:
            self._save_precomputed(precomputed_path, sampled)
        
        return sampled
    
    def _herding_for_class(self, n: int, class_id: int) -> List[int]:
        """Herding selection for a single class."""
        class_indices = self.class_indices[class_id]
        
        if n >= len(class_indices):
            return class_indices.copy()
        
        features = self._extract_features_for_class(class_id)
        mean_feature = features.mean(axis=0)
        mean_feature /= np.linalg.norm(mean_feature)
        
        selected_indices = []
        remaining_indices = list(range(len(class_indices)))
        selected_sum = np.zeros_like(mean_feature)
        
        for i in range(n):
            if not remaining_indices:
                break
            
            best_candidate = None
            best_distance = float('inf')
            
            for candidate_idx in remaining_indices:
                candidate_feature = features[candidate_idx]
                new_selected_sum = selected_sum + candidate_feature
                new_selected_mean = new_selected_sum / (i + 1)
                distance = np.linalg.norm(new_selected_mean - mean_feature)
                
                if distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
                selected_sum += features[best_candidate]
        
        return [class_indices[i] for i in selected_indices]
