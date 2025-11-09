"""
Pytest configuration and fixtures for BraTS 3D segmentation tests.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
import nibabel as nib


@pytest.fixture
def device():
    """Fixture providing appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_batch():
    """Fixture providing a sample batch of data."""
    batch_size = 2
    channels = 4
    depth, height, width = 32, 32, 32
    
    images = torch.randn(batch_size, channels, depth, height, width)
    labels = torch.randint(0, 4, (batch_size, depth, height, width))
    
    return {
        "images": images,
        "labels": labels,
        "batch_size": batch_size,
        "shape": (depth, height, width)
    }


@pytest.fixture
def small_batch():
    """Fixture providing a smaller batch for quick tests."""
    batch_size = 1
    channels = 4
    depth, height, width = 16, 16, 16
    
    images = torch.randn(batch_size, channels, depth, height, width)
    labels = torch.randint(0, 4, (batch_size, depth, height, width))
    
    return {
        "images": images,
        "labels": labels,
        "batch_size": batch_size,
        "shape": (depth, height, width)
    }


@pytest.fixture
def mock_brats_data():
    """Fixture providing mock BraTS data."""
    # Create mock image data
    shape = (80, 80, 80)
    
    # Different modalities with slight variations
    t1 = np.random.randn(*shape).astype(np.float32) * 100 + 500
    t1ce = np.random.randn(*shape).astype(np.float32) * 120 + 600
    t2 = np.random.randn(*shape).astype(np.float32) * 80 + 400
    flair = np.random.randn(*shape).astype(np.float32) * 90 + 450
    
    # Create realistic segmentation with tumor regions
    segmentation = np.zeros(shape, dtype=np.uint8)
    
    # Add tumor core (NCR)
    segmentation[30:50, 30:50, 30:50] = 1
    
    # Add edema around tumor
    segmentation[25:55, 25:55, 25:55] = 2
    segmentation[30:50, 30:50, 30:50] = 1  # Keep tumor core as NCR
    
    # Add enhancing tumor within core
    segmentation[35:45, 35:45, 35:45] = 4
    
    return {
        "t1": t1,
        "t1ce": t1ce, 
        "t2": t2,
        "flair": flair,
        "segmentation": segmentation,
        "shape": shape
    }


@pytest.fixture
def temp_brats_directory(mock_brats_data):
    """Fixture creating a temporary BraTS-style directory structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple subjects
        subject_ids = ["subject_001", "subject_002", "subject_003"]
        
        for subject_id in subject_ids:
            subject_dir = temp_path / subject_id
            subject_dir.mkdir()
            
            # Save modalities as NIfTI files
            for modality in ["t1", "t1ce", "t2", "flair"]:
                data = mock_brats_data[modality]
                # Add some subject-specific variation
                data = data + np.random.randn(*data.shape) * 10
                
                img = nib.Nifti1Image(data, affine=np.eye(4))
                nib.save(img, subject_dir / f"{subject_id}_{modality}.nii.gz")
            
            # Save segmentation
            seg_data = mock_brats_data["segmentation"].copy()
            if subject_id == "subject_002":
                # Make subject_002 have no enhancing tumor
                seg_data[seg_data == 4] = 1
            elif subject_id == "subject_003":
                # Make subject_003 have only edema
                seg_data[seg_data == 1] = 0
                seg_data[seg_data == 4] = 0
            
            seg_img = nib.Nifti1Image(seg_data, affine=np.eye(4))
            nib.save(seg_img, subject_dir / f"{subject_id}_seg.nii.gz")
        
        yield {
            "path": temp_path,
            "subjects": subject_ids,
            "data": mock_brats_data
        }


@pytest.fixture
def binary_masks():
    """Fixture providing binary masks for testing."""
    shape = (20, 20, 20)
    
    # Perfect overlap
    mask1 = np.zeros(shape, dtype=bool)
    mask1[5:15, 5:15, 5:15] = True
    
    mask2 = mask1.copy()
    
    # Partial overlap
    mask3 = np.zeros(shape, dtype=bool)
    mask3[8:18, 8:18, 8:18] = True
    
    # No overlap
    mask4 = np.zeros(shape, dtype=bool)
    mask4[0:5, 0:5, 0:5] = True
    
    return {
        "perfect_overlap": (mask1, mask2),
        "partial_overlap": (mask1, mask3),
        "no_overlap": (mask1, mask4),
        "shape": shape
    }


@pytest.fixture
def multiclass_segmentations():
    """Fixture providing multi-class segmentation examples."""
    shape = (40, 40, 40)
    
    # Ground truth with all BraTS classes
    target = np.zeros(shape, dtype=np.uint8)
    target[15:25, 15:25, 15:25] = 1  # NCR
    target[20:30, 15:25, 15:25] = 2  # ED
    target[17:20, 17:20, 17:20] = 4  # ET
    
    # Perfect prediction
    pred_perfect = target.copy()
    
    # Slightly offset prediction
    pred_offset = np.zeros(shape, dtype=np.uint8)
    pred_offset[16:26, 16:26, 16:26] = 1  # NCR shifted by 1
    pred_offset[21:31, 16:26, 16:26] = 2  # ED shifted by 1
    pred_offset[18:21, 18:21, 18:21] = 4  # ET shifted by 1
    
    # Over-segmentation
    pred_over = target.copy()
    pred_over[10:35, 10:35, 10:35] = 2  # Much larger ED region
    
    # Under-segmentation
    pred_under = np.zeros(shape, dtype=np.uint8)
    pred_under[17:23, 17:23, 17:23] = 1  # Smaller NCR
    
    return {
        "target": target,
        "perfect": pred_perfect,
        "offset": pred_offset,
        "over_segmentation": pred_over,
        "under_segmentation": pred_under,
        "shape": shape
    }


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Automatically set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def model_configs():
    """Fixture providing various model configurations for testing."""
    from src.models.unet3d import UNet3DConfig
    
    configs = {
        "minimal": UNet3DConfig(
            input_channels=4,
            output_channels=4,
            feature_maps=[8, 16, 32],
            deep_supervision=False,
        ),
        "standard": UNet3DConfig(
            input_channels=4,
            output_channels=4,
            feature_maps=[32, 64, 128, 256],
            deep_supervision=False,
        ),
        "deep_supervision": UNet3DConfig(
            input_channels=4,
            output_channels=4,
            feature_maps=[32, 64, 128, 256],
            deep_supervision=True,
        ),
        "attention": UNet3DConfig(
            input_channels=4,
            output_channels=4,
            feature_maps=[32, 64, 128, 256],
            use_attention=True,
        ),
    }
    
    return configs


@pytest.fixture
def loss_test_data():
    """Fixture providing data for testing loss functions."""
    batch_size = 2
    num_classes = 4
    spatial_shape = (16, 16, 16)
    
    # Predictions (logits)
    predictions = torch.randn(batch_size, num_classes, *spatial_shape)
    
    # Ground truth (class indices)
    targets = torch.randint(0, num_classes, (batch_size, *spatial_shape))
    
    # One-hot encoded targets
    targets_onehot = torch.nn.functional.one_hot(targets, num_classes)
    targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()
    
    return {
        "predictions": predictions,
        "targets": targets,
        "targets_onehot": targets_onehot,
        "num_classes": num_classes,
        "spatial_shape": spatial_shape,
    }


# Test utilities
def assert_tensor_properties(tensor, expected_shape=None, expected_dtype=None, 
                           expected_device=None, check_finite=True):
    """Utility function to assert tensor properties."""
    if expected_shape is not None:
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    if expected_dtype is not None:
        assert tensor.dtype == expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    
    if expected_device is not None:
        assert tensor.device.type == expected_device, f"Expected device {expected_device}, got {tensor.device}"
    
    if check_finite:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"


def create_mock_checkpoint(model, optimizer=None, epoch=0, metrics=None):
    """Utility to create mock checkpoints for testing."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics or {},
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    return checkpoint