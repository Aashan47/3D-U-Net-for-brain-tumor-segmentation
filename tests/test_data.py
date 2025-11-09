import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import nibabel as nib

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.transforms import get_training_transforms, get_validation_transforms
from data.dataset import BraTSDataset, BraTSDataModule
from data.preprocessing import BraTSPreprocessor


class TestBraTSTransforms:
    """Test BraTS data transforms."""
    
    def test_training_transforms_shape(self):
        """Test that training transforms produce correct output shapes."""
        transforms = get_training_transforms(
            patch_size=(64, 64, 64),
            samples_per_epoch=2,
        )
        
        # Create mock data dict
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock NIfTI files
            for mod in ["t1", "t1ce", "t2", "flair"]:
                img_data = np.random.randn(100, 100, 100).astype(np.float32)
                img = nib.Nifti1Image(img_data, affine=np.eye(4))
                nib.save(img, Path(temp_dir) / f"test_{mod}.nii.gz")
            
            # Create mock segmentation
            seg_data = np.random.randint(0, 5, (100, 100, 100)).astype(np.uint8)
            seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
            nib.save(seg_img, Path(temp_dir) / "test_seg.nii.gz")
            
            data_dict = {
                "image": [str(Path(temp_dir) / f"test_{mod}.nii.gz") for mod in ["t1", "t1ce", "t2", "flair"]],
                "label": str(Path(temp_dir) / "test_seg.nii.gz"),
                "subject_id": "test_subject",
            }
            
            # Apply transforms
            transformed = transforms(data_dict)
            
            # Check shapes
            assert transformed["image"].shape[0] == 4  # 4 modalities
            assert len(transformed["image"].shape) == 4  # (C, D, H, W)
            assert transformed["label"].shape[0] == 1   # Single channel
            assert all(s == 64 for s in transformed["image"].shape[1:])  # Patch size
    
    def test_validation_transforms_consistency(self):
        """Test that validation transforms are consistent (deterministic)."""
        transforms = get_validation_transforms()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock data
            img_data = np.random.randn(80, 80, 80).astype(np.float32)
            for mod in ["t1", "t1ce", "t2", "flair"]:
                img = nib.Nifti1Image(img_data, affine=np.eye(4))
                nib.save(img, Path(temp_dir) / f"test_{mod}.nii.gz")
            
            seg_data = np.random.randint(0, 5, (80, 80, 80)).astype(np.uint8)
            seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
            nib.save(seg_img, Path(temp_dir) / "test_seg.nii.gz")
            
            data_dict = {
                "image": [str(Path(temp_dir) / f"test_{mod}.nii.gz") for mod in ["t1", "t1ce", "t2", "flair"]],
                "label": str(Path(temp_dir) / "test_seg.nii.gz"),
                "subject_id": "test_subject",
            }
            
            # Apply transforms twice
            result1 = transforms(data_dict.copy())
            result2 = transforms(data_dict.copy())
            
            # Results should be identical
            assert torch.allclose(result1["image"], result2["image"], atol=1e-6)
            assert torch.equal(result1["label"], result2["label"])


class TestBraTSDataset:
    """Test BraTS dataset class."""
    
    def test_dataset_creation(self):
        """Test dataset creation with mock data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock subject directory
            subject_dir = Path(temp_dir) / "test_subject"
            subject_dir.mkdir()
            
            # Create mock files
            for mod in ["t1", "t1ce", "t2", "flair"]:
                img_data = np.random.randn(50, 50, 50).astype(np.float32)
                img = nib.Nifti1Image(img_data, affine=np.eye(4))
                nib.save(img, subject_dir / f"test_subject_{mod}.nii.gz")
            
            seg_data = np.random.randint(0, 5, (50, 50, 50)).astype(np.uint8)
            seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
            nib.save(seg_img, subject_dir / "test_subject_seg.nii.gz")
            
            # Create dataset
            dataset = BraTSDataset(
                data_dir=temp_dir,
                subjects=["test_subject"],
                transforms=get_validation_transforms(),
                cache_rate=0.0,
            )
            
            assert len(dataset) == 1
            
            # Test data loading
            sample = dataset[0]
            assert "image" in sample
            assert "label" in sample
            assert sample["image"].shape[0] == 4  # 4 modalities


class TestBraTSDataModule:
    """Test BraTS data module."""
    
    def test_datamodule_creation(self):
        """Test data module creation and split generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create mock subjects
            for i in range(5):
                subject_id = f"subject_{i:03d}"
                subject_dir = Path(temp_dir) / subject_id
                subject_dir.mkdir()
                
                # Create mock files
                for mod in ["t1", "t1ce", "t2", "flair"]:
                    img_data = np.random.randn(40, 40, 40).astype(np.float32)
                    img = nib.Nifti1Image(img_data, affine=np.eye(4))
                    nib.save(img, subject_dir / f"{subject_id}_{mod}.nii.gz")
                
                seg_data = np.random.randint(0, 5, (40, 40, 40)).astype(np.uint8)
                seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
                nib.save(seg_img, subject_dir / f"{subject_id}_seg.nii.gz")
            
            # Create data module
            data_module = BraTSDataModule(
                data_dir=temp_dir,
                batch_size=1,
                n_folds=3,
                cache_rate=0.0,
            )
            
            assert len(data_module.all_subjects) == 5
            assert len(data_module.cv_splits) == 3
            
            # Test fold data loaders
            train_loader, val_loader = data_module.get_fold_dataloaders(0)
            assert len(train_loader.dataset) > 0
            assert len(val_loader.dataset) > 0


class TestBraTSPreprocessor:
    """Test BraTS preprocessing functionality."""
    
    def test_validation(self):
        """Test dataset validation functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid subject
            valid_subject = Path(temp_dir) / "valid_subject"
            valid_subject.mkdir()
            
            for mod in ["t1", "t1ce", "t2", "flair"]:
                img_data = np.random.randn(50, 50, 50).astype(np.float32)
                img = nib.Nifti1Image(img_data, affine=np.eye(4))
                nib.save(img, valid_subject / f"valid_subject_{mod}.nii.gz")
            
            seg_data = np.random.randint(0, 5, (50, 50, 50)).astype(np.uint8)
            seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
            nib.save(seg_img, valid_subject / "valid_subject_seg.nii.gz")
            
            # Create invalid subject (missing files)
            invalid_subject = Path(temp_dir) / "invalid_subject" 
            invalid_subject.mkdir()
            
            # Only create some files
            img_data = np.random.randn(50, 50, 50).astype(np.float32)
            img = nib.Nifti1Image(img_data, affine=np.eye(4))
            nib.save(img, invalid_subject / "invalid_subject_t1.nii.gz")
            
            # Test preprocessor validation
            with tempfile.TemporaryDirectory() as output_dir:
                preprocessor = BraTSPreprocessor(
                    input_dir=temp_dir,
                    output_dir=output_dir,
                    num_workers=1,
                )
                
                validation_df = preprocessor.validate_dataset()
                
                assert len(validation_df) == 2
                assert validation_df.loc[validation_df["subject_id"] == "valid_subject", "valid"].iloc[0] == True
                assert validation_df.loc[validation_df["subject_id"] == "invalid_subject", "valid"].iloc[0] == False


def test_patch_extraction_shapes():
    """Test patch extraction produces correct shapes."""
    # Test different patch sizes
    patch_sizes = [(64, 64, 64), (96, 96, 96), (128, 128, 128)]
    
    for patch_size in patch_sizes:
        transforms = get_training_transforms(
            patch_size=patch_size,
            samples_per_epoch=1,
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create larger mock data
            img_data = np.random.randn(150, 150, 150).astype(np.float32)
            for mod in ["t1", "t1ce", "t2", "flair"]:
                img = nib.Nifti1Image(img_data, affine=np.eye(4))
                nib.save(img, Path(temp_dir) / f"test_{mod}.nii.gz")
            
            seg_data = np.random.randint(0, 5, (150, 150, 150)).astype(np.uint8)
            # Ensure some foreground voxels
            seg_data[70:80, 70:80, 70:80] = 1
            seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
            nib.save(seg_img, Path(temp_dir) / "test_seg.nii.gz")
            
            data_dict = {
                "image": [str(Path(temp_dir) / f"test_{mod}.nii.gz") for mod in ["t1", "t1ce", "t2", "flair"]],
                "label": str(Path(temp_dir) / "test_seg.nii.gz"),
                "subject_id": "test_subject",
            }
            
            transformed = transforms(data_dict)
            
            # Check patch dimensions
            assert transformed["image"].shape[1:] == patch_size
            assert transformed["label"].shape[1:] == patch_size


def test_foreground_background_ratio():
    """Test that patch sampling respects foreground/background ratio."""
    transforms = get_training_transforms(
        patch_size=(32, 32, 32),
        samples_per_epoch=10,
        foreground_ratio=1.0,  # Only foreground patches
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create data with specific tumor location
        img_data = np.random.randn(100, 100, 100).astype(np.float32)
        for mod in ["t1", "t1ce", "t2", "flair"]:
            img = nib.Nifti1Image(img_data, affine=np.eye(4))
            nib.save(img, Path(temp_dir) / f"test_{mod}.nii.gz")
        
        # Create segmentation with tumor only in center
        seg_data = np.zeros((100, 100, 100), dtype=np.uint8)
        seg_data[40:60, 40:60, 40:60] = 1  # Tumor in center
        seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
        nib.save(seg_img, Path(temp_dir) / "test_seg.nii.gz")
        
        data_dict = {
            "image": [str(Path(temp_dir) / f"test_{mod}.nii.gz") for mod in ["t1", "t1ce", "t2", "flair"]],
            "label": str(Path(temp_dir) / "test_seg.nii.gz"),
            "subject_id": "test_subject",
        }
        
        transformed = transforms(data_dict)
        
        # With 100% foreground ratio, all patches should contain some tumor
        assert torch.sum(transformed["label"]) > 0


if __name__ == "__main__":
    pytest.main([__file__])