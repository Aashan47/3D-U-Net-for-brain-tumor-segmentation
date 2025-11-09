import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import Compose
from sklearn.model_selection import KFold

from .preprocessing import BraTSPreprocessor
from .transforms import get_training_transforms, get_validation_transforms


class BraTSDataset(Dataset):
    """BraTS dataset for brain tumor segmentation."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        subjects: List[str],
        transforms: Optional[Compose] = None,
        cache_rate: float = 0.0,
        num_workers: int = 4,
    ):
        """
        Initialize BraTS dataset.
        
        Args:
            data_dir: Path to BraTS data directory
            subjects: List of subject IDs to include
            transforms: MONAI transforms to apply
            cache_rate: Fraction of data to cache in memory
            num_workers: Number of workers for data loading
        """
        self.data_dir = Path(data_dir)
        self.subjects = subjects
        self.transforms = transforms
        
        # Build data dictionaries
        self.data_dicts = self._build_data_dicts()
        
        # Create MONAI dataset
        if cache_rate > 0:
            self.dataset = CacheDataset(
                data=self.data_dicts,
                transform=transforms,
                cache_rate=cache_rate,
                num_workers=num_workers,
            )
        else:
            from monai.data import Dataset as MonaiDataset
            self.dataset = MonaiDataset(
                data=self.data_dicts,
                transform=transforms,
            )
    
    def _build_data_dicts(self) -> List[Dict[str, str]]:
        """Build list of data dictionaries for each subject."""
        data_dicts = []
        
        for subject_id in self.subjects:
            subject_dir = self.data_dir / subject_id
            
            if not subject_dir.exists():
                continue
                
            # Define modality file paths
            modalities = {
                "t1": subject_dir / f"{subject_id}_t1.nii.gz",
                "t1ce": subject_dir / f"{subject_id}_t1ce.nii.gz", 
                "t2": subject_dir / f"{subject_id}_t2.nii.gz",
                "flair": subject_dir / f"{subject_id}_flair.nii.gz",
            }
            
            seg_path = subject_dir / f"{subject_id}_seg.nii.gz"
            
            # Verify all files exist
            if all(p.exists() for p in modalities.values()) and seg_path.exists():
                data_dict = {
                    "image": [str(p) for p in modalities.values()],  # List of modality paths
                    "label": str(seg_path),
                    "subject_id": subject_id,
                }
                data_dicts.append(data_dict)
        
        return data_dicts
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.dataset[idx]


class BraTSDataModule:
    """Data module for handling BraTS dataset splits and data loading."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 2,
        num_workers: int = 4,
        cache_rate: float = 0.1,
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        samples_per_epoch: int = 1000,
        foreground_ratio: float = 0.7,
        val_split: float = 0.2,
        n_folds: int = 5,
        seed: int = 42,
    ):
        """
        Initialize BraTS data module.
        
        Args:
            data_dir: Path to BraTS data directory
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            cache_rate: Fraction of data to cache in memory
            patch_size: Size of patches for training
            samples_per_epoch: Number of samples per epoch
            foreground_ratio: Ratio of foreground patches during training
            val_split: Validation split ratio
            n_folds: Number of folds for cross-validation
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_rate = cache_rate
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.foreground_ratio = foreground_ratio
        self.val_split = val_split
        self.n_folds = n_folds
        self.seed = seed
        
        # Get all subjects
        self.all_subjects = self._get_all_subjects()
        
        # Create cross-validation splits
        self.cv_splits = self._create_cv_splits()
        
    def _get_all_subjects(self) -> List[str]:
        """Get list of all available subjects."""
        subject_dirs = [
            d.name for d in self.data_dir.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ]
        
        # Filter subjects that have all required files
        valid_subjects = []
        for subject_id in subject_dirs:
            subject_dir = self.data_dir / subject_id
            
            required_files = [
                f"{subject_id}_t1.nii.gz",
                f"{subject_id}_t1ce.nii.gz", 
                f"{subject_id}_t2.nii.gz",
                f"{subject_id}_flair.nii.gz",
                f"{subject_id}_seg.nii.gz",
            ]
            
            if all((subject_dir / f).exists() for f in required_files):
                valid_subjects.append(subject_id)
        
        return sorted(valid_subjects)
    
    def _create_cv_splits(self) -> List[Tuple[List[str], List[str]]]:
        """Create cross-validation splits."""
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        
        splits = []
        for train_idx, val_idx in kfold.split(self.all_subjects):
            train_subjects = [self.all_subjects[i] for i in train_idx]
            val_subjects = [self.all_subjects[i] for i in val_idx]
            splits.append((train_subjects, val_subjects))
        
        return splits
    
    def get_fold_dataloaders(self, fold: int) -> Tuple[DataLoader, DataLoader]:
        """Get train and validation dataloaders for specific fold."""
        if fold >= self.n_folds:
            raise ValueError(f"Fold {fold} exceeds number of folds {self.n_folds}")
        
        train_subjects, val_subjects = self.cv_splits[fold]
        
        # Get transforms
        train_transforms = get_training_transforms(
            patch_size=self.patch_size,
            samples_per_epoch=self.samples_per_epoch,
            foreground_ratio=self.foreground_ratio,
        )
        
        val_transforms = get_validation_transforms()
        
        # Create datasets
        train_dataset = BraTSDataset(
            data_dir=self.data_dir,
            subjects=train_subjects,
            transforms=train_transforms,
            cache_rate=self.cache_rate,
            num_workers=self.num_workers,
        )
        
        val_dataset = BraTSDataset(
            data_dir=self.data_dir,
            subjects=val_subjects,
            transforms=val_transforms,
            cache_rate=0.0,  # Don't cache validation data
            num_workers=self.num_workers,
        )
        
        # Create data loaders
        train_loader = MonaiDataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        val_loader = MonaiDataLoader(
            val_dataset,
            batch_size=1,  # Validation with batch size 1
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader
    
    def get_subject_info(self) -> pd.DataFrame:
        """Get information about all subjects."""
        info = []
        for subject_id in self.all_subjects:
            info.append({
                "subject_id": subject_id,
                "data_dir": str(self.data_dir / subject_id),
            })
        
        return pd.DataFrame(info)
    
    def print_dataset_info(self):
        """Print dataset information."""
        print(f"Total subjects: {len(self.all_subjects)}")
        print(f"Cross-validation folds: {self.n_folds}")
        
        for fold, (train_subjects, val_subjects) in enumerate(self.cv_splits):
            print(f"Fold {fold}: Train={len(train_subjects)}, Val={len(val_subjects)}")