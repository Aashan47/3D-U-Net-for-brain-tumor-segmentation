import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd

from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    Orientation,
    Spacing,
    ScaleIntensityRange,
    NormalizeIntensity,
    CropForeground,
    SaveImage,
)


class BraTSPreprocessor:
    """Preprocessor for BraTS dataset with comprehensive validation and preprocessing."""
    
    def __init__(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        target_orientation: str = "RAS",
        intensity_range: Tuple[float, float] = (-175, 250),
        num_workers: int = 4,
    ):
        """
        Initialize BraTS preprocessor.
        
        Args:
            input_dir: Path to raw BraTS data directory
            output_dir: Path to output preprocessed data
            target_spacing: Target voxel spacing in mm
            target_orientation: Target image orientation
            intensity_range: Intensity clipping range
            num_workers: Number of parallel workers
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.target_spacing = target_spacing
        self.target_orientation = target_orientation
        self.intensity_range = intensity_range
        self.num_workers = num_workers
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def validate_dataset(self) -> pd.DataFrame:
        """
        Validate BraTS dataset structure and file integrity.
        
        Returns:
            DataFrame with validation results for each subject
        """
        self.logger.info("Validating BraTS dataset...")
        
        # Get all subject directories
        subject_dirs = [
            d for d in self.input_dir.iterdir() 
            if d.is_dir() and not d.name.startswith(".")
        ]
        
        validation_results = []
        
        for subject_dir in tqdm(subject_dirs, desc="Validating subjects"):
            subject_id = subject_dir.name
            result = {
                "subject_id": subject_id,
                "valid": True,
                "missing_files": [],
                "file_errors": [],
                "shape_mismatches": [],
            }
            
            # Define expected files
            modalities = ["t1", "t1ce", "t2", "flair"]
            expected_files = {
                mod: subject_dir / f"{subject_id}_{mod}.nii.gz"
                for mod in modalities
            }
            expected_files["seg"] = subject_dir / f"{subject_id}_seg.nii.gz"
            
            # Check file existence
            missing_files = []
            for mod, filepath in expected_files.items():
                if not filepath.exists():
                    missing_files.append(f"{mod}.nii.gz")
            
            if missing_files:
                result["missing_files"] = missing_files
                result["valid"] = False
            
            # Check file integrity and shapes
            if result["valid"]:
                try:
                    shapes = {}
                    for mod, filepath in expected_files.items():
                        img = nib.load(str(filepath))
                        shapes[mod] = img.shape
                    
                    # Check if all modalities have same shape
                    reference_shape = shapes[modalities[0]]
                    for mod in modalities[1:]:
                        if shapes[mod] != reference_shape:
                            result["shape_mismatches"].append(f"{mod}: {shapes[mod]}")
                    
                    if result["shape_mismatches"]:
                        result["valid"] = False
                        
                except Exception as e:
                    result["file_errors"].append(str(e))
                    result["valid"] = False
            
            validation_results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame(validation_results)
        
        # Save validation report
        report_path = self.output_dir / "validation_report.csv"
        df.to_csv(report_path, index=False)
        
        # Print summary
        valid_count = df["valid"].sum()
        total_count = len(df)
        self.logger.info(f"Validation complete: {valid_count}/{total_count} subjects valid")
        
        if valid_count < total_count:
            invalid_subjects = df[~df["valid"]]["subject_id"].tolist()
            self.logger.warning(f"Invalid subjects: {invalid_subjects}")
        
        return df
    
    def preprocess_subject(self, subject_id: str) -> bool:
        """
        Preprocess a single subject.
        
        Args:
            subject_id: Subject identifier
            
        Returns:
            Success flag
        """
        try:
            subject_input_dir = self.input_dir / subject_id
            subject_output_dir = self.output_dir / subject_id
            subject_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Define file paths
            modalities = ["t1", "t1ce", "t2", "flair"]
            input_files = {
                mod: subject_input_dir / f"{subject_id}_{mod}.nii.gz"
                for mod in modalities
            }
            input_files["seg"] = subject_input_dir / f"{subject_id}_seg.nii.gz"
            
            output_files = {
                mod: subject_output_dir / f"{subject_id}_{mod}_preprocessed.nii.gz"
                for mod in modalities
            }
            output_files["seg"] = subject_output_dir / f"{subject_id}_seg_preprocessed.nii.gz"
            
            # Load and preprocess each modality
            processed_images = {}
            reference_image = None
            
            for mod in modalities:
                # Load image
                loader = LoadImage(image_only=True)
                img = loader(str(input_files[mod]))
                
                # Ensure channel first
                img = EnsureChannelFirst()(img)
                
                # Set reference image for consistent processing
                if reference_image is None:
                    reference_image = img
                
                # Reorient to target orientation
                orient = Orientation(axcodes=self.target_orientation)
                img = orient(img)
                
                # Resample to target spacing
                spacing = Spacing(
                    pixdim=self.target_spacing,
                    mode="bilinear",
                )
                img = spacing(img)
                
                # Intensity preprocessing
                # Clip intensities
                img = ScaleIntensityRange(
                    a_min=self.intensity_range[0],
                    a_max=self.intensity_range[1],
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                )(img)
                
                # Normalize intensity
                img = NormalizeIntensity(nonzero=True)(img)
                
                processed_images[mod] = img
            
            # Process segmentation
            loader = LoadImage(image_only=True)
            seg = loader(str(input_files["seg"]))
            seg = EnsureChannelFirst()(seg)
            
            # Reorient and resample segmentation (nearest neighbor)
            seg = Orientation(axcodes=self.target_orientation)(seg)
            seg = Spacing(
                pixdim=self.target_spacing,
                mode="nearest",
            )(seg)
            
            # Crop all images to foreground (based on combined modalities)
            # Create brain mask from all modalities
            combined_mask = None
            for mod in modalities:
                mask = processed_images[mod][0] > 0  # Remove channel dimension for mask
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = combined_mask | mask
            
            # Add channel dimension back to mask
            combined_mask = combined_mask[None, ...]
            
            # Crop to foreground
            cropper = CropForeground(source_key="image", margin=10)
            
            # Apply cropping to all images using the combined mask
            for mod in modalities:
                processed_images[mod] = cropper(
                    {"image": combined_mask, mod: processed_images[mod]}
                )[mod]
            
            seg = cropper({"image": combined_mask, "seg": seg})["seg"]
            
            # Save preprocessed images
            saver = SaveImage(
                output_dir=str(subject_output_dir),
                output_postfix="",
                separate_folder=False,
                print_log=False,
            )
            
            for mod in modalities:
                saver(
                    processed_images[mod],
                    meta_dict={"filename_or_obj": str(output_files[mod])},
                )
            
            # Save segmentation
            saver(
                seg,
                meta_dict={"filename_or_obj": str(output_files["seg"])},
            )
            
            # Save preprocessing metadata
            metadata = {
                "subject_id": subject_id,
                "target_spacing": self.target_spacing,
                "target_orientation": self.target_orientation,
                "intensity_range": self.intensity_range,
                "original_shape": reference_image.shape,
                "processed_shape": processed_images[modalities[0]].shape,
            }
            
            import json
            metadata_path = subject_output_dir / "preprocessing_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preprocessing subject {subject_id}: {str(e)}")
            return False
    
    def preprocess_dataset(self, max_subjects: Optional[int] = None) -> Dict[str, int]:
        """
        Preprocess entire BraTS dataset.
        
        Args:
            max_subjects: Maximum number of subjects to process (for testing)
            
        Returns:
            Dictionary with processing statistics
        """
        self.logger.info("Starting dataset preprocessing...")
        
        # Get valid subjects from validation
        validation_df = self.validate_dataset()
        valid_subjects = validation_df[validation_df["valid"]]["subject_id"].tolist()
        
        if max_subjects:
            valid_subjects = valid_subjects[:max_subjects]
        
        # Process subjects
        successful = 0
        failed = 0
        
        for subject_id in tqdm(valid_subjects, desc="Preprocessing subjects"):
            if self.preprocess_subject(subject_id):
                successful += 1
            else:
                failed += 1
        
        # Save processing summary
        summary = {
            "total_subjects": len(valid_subjects),
            "successful": successful,
            "failed": failed,
            "success_rate": successful / len(valid_subjects) if valid_subjects else 0,
        }
        
        summary_path = self.output_dir / "preprocessing_summary.json"
        import json
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Preprocessing complete: {successful}/{len(valid_subjects)} subjects successful")
        
        return summary
    
    def create_dataset_split(
        self,
        n_folds: int = 5,
        seed: int = 42,
        stratify_by_tumor_presence: bool = True,
    ) -> pd.DataFrame:
        """
        Create cross-validation splits for the dataset.
        
        Args:
            n_folds: Number of cross-validation folds
            seed: Random seed for reproducibility
            stratify_by_tumor_presence: Whether to stratify by tumor presence
            
        Returns:
            DataFrame with fold assignments
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        
        # Get preprocessed subjects
        preprocessed_subjects = [
            d.name for d in self.output_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        
        subjects_df = pd.DataFrame({"subject_id": preprocessed_subjects})
        
        if stratify_by_tumor_presence:
            # Analyze tumor presence for stratification
            tumor_labels = []
            for subject_id in preprocessed_subjects:
                seg_path = self.output_dir / subject_id / f"{subject_id}_seg_preprocessed.nii.gz"
                try:
                    seg = nib.load(str(seg_path)).get_fdata()
                    has_enhancing_tumor = np.any(seg == 4)  # ET label
                    tumor_labels.append(int(has_enhancing_tumor))
                except:
                    tumor_labels.append(0)
            
            kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_iterator = kfold.split(preprocessed_subjects, tumor_labels)
        else:
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            fold_iterator = kfold.split(preprocessed_subjects)
        
        # Assign fold numbers
        subjects_df["fold"] = -1
        for fold, (_, val_idx) in enumerate(fold_iterator):
            subjects_df.iloc[val_idx, subjects_df.columns.get_loc("fold")] = fold
        
        # Save splits
        splits_path = self.output_dir / "dataset_splits.csv"
        subjects_df.to_csv(splits_path, index=False)
        
        self.logger.info(f"Created {n_folds}-fold cross-validation splits")
        self.logger.info(f"Subjects per fold: {subjects_df['fold'].value_counts().sort_index().tolist()}")
        
        return subjects_df