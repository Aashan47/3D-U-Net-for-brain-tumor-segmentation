from typing import Tuple, Optional
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd, 
    Spacingd,
    ScaleIntensityRanged,
    NormalizeIntensityd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandRotated,
    RandFlipd,
    Rand3DElasticd,
    RandScaleIntensityd,
    RandAdjustContrastd,
    RandGaussianNoised,
    RandBiasFieldd,
    RandSpatialCropSamplesd,
    ToTensord,
    EnsureTyped,
    ConcatItemsd,
)


def get_training_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    samples_per_epoch: int = 1000,
    foreground_ratio: float = 0.7,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Get training transforms for BraTS dataset.
    
    Args:
        patch_size: Size of patches for training
        samples_per_epoch: Number of samples per epoch
        foreground_ratio: Ratio of foreground patches
        spacing: Target spacing for resampling
        
    Returns:
        Composed transforms for training
    """
    
    train_transforms = Compose([
        # Load images and labels
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Concatenate modalities into single 4-channel image
        ConcatItemsd(keys=["image"], name="image"),
        
        # Spatial preprocessing
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        
        # Intensity preprocessing
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
        # Per-channel normalization (for each modality)
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        
        # Crop to foreground
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10,
        ),
        
        # Random patch sampling
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=foreground_ratio,
            neg=1 - foreground_ratio,
            num_samples=samples_per_epoch,
        ),
        
        # Spatial augmentations
        RandRotate90d(
            keys=["image", "label"],
            prob=0.5,
            spatial_axes=[0, 1],
        ),
        
        RandRotated(
            keys=["image", "label"],
            prob=0.3,
            range_x=0.3,
            range_y=0.3, 
            range_z=0.3,
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=0,
        ),
        
        RandFlipd(
            keys=["image", "label"], 
            prob=0.5,
            spatial_axis=1,
        ),
        
        RandFlipd(
            keys=["image", "label"],
            prob=0.5,
            spatial_axis=2,
        ),
        
        Rand3DElasticd(
            keys=["image", "label"],
            prob=0.2,
            sigma_range=(5, 7),
            magnitude_range=(50, 150),
            mode=("bilinear", "nearest"),
            padding_mode="border",
        ),
        
        # Intensity augmentations (only on image)
        RandScaleIntensityd(
            keys=["image"],
            factors=0.1,
            prob=0.3,
        ),
        
        RandAdjustContrastd(
            keys=["image"],
            prob=0.3,
            gamma=(0.7, 1.5),
        ),
        
        RandGaussianNoised(
            keys=["image"],
            prob=0.2,
            std=0.1,
        ),
        
        RandBiasFieldd(
            keys=["image"],
            prob=0.2,
            degree=3,
            coeff_range=(0.0, 0.1),
        ),
        
        # Convert to tensors
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    return train_transforms


def get_validation_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Get validation transforms for BraTS dataset.
    
    Args:
        spacing: Target spacing for resampling
        
    Returns:
        Composed transforms for validation
    """
    
    val_transforms = Compose([
        # Load images and labels
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Concatenate modalities into single 4-channel image
        ConcatItemsd(keys=["image"], name="image"),
        
        # Spatial preprocessing
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        
        # Intensity preprocessing
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
        # Per-channel normalization
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        
        # Crop to foreground
        CropForegroundd(
            keys=["image", "label"],
            source_key="image",
            margin=10,
        ),
        
        # Convert to tensors
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"]),
    ])
    
    return val_transforms


def get_inference_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Compose:
    """
    Get inference transforms for BraTS dataset (no labels).
    
    Args:
        spacing: Target spacing for resampling
        
    Returns:
        Composed transforms for inference
    """
    
    inference_transforms = Compose([
        # Load images
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        
        # Concatenate modalities into single 4-channel image
        ConcatItemsd(keys=["image"], name="image"),
        
        # Spatial preprocessing
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=spacing,
            mode="bilinear",
        ),
        
        # Intensity preprocessing
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        
        # Per-channel normalization
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        
        # Crop to foreground
        CropForegroundd(
            keys=["image"],
            source_key="image",
            margin=10,
        ),
        
        # Convert to tensors
        ToTensord(keys=["image"]),
        EnsureTyped(keys=["image"]),
    ])
    
    return inference_transforms