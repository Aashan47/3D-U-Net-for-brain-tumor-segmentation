import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Union


class DiceMetric(nn.Module):
    """Dice coefficient metric for binary segmentation."""
    
    def __init__(
        self,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize Dice metric.
        
        Args:
            include_background: Whether to include background class
            smooth: Smoothing factor to avoid division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Dice coefficient.
        
        Args:
            pred: Predicted probabilities (B, C, ...) or binary mask (B, ...)
            target: Ground truth labels (B, C, ...) or (B, ...)
            
        Returns:
            Dice coefficient(s)
        """
        # Handle binary case
        if pred.dim() == target.dim() and pred.shape[1:] == target.shape[1:]:
            # Binary prediction
            pred = (pred > 0.5).float()
            target = target.float()
            
            # Add channel dimension
            if pred.dim() == 4:  # 3D case: (B, D, H, W) -> (B, 1, D, H, W)
                pred = pred.unsqueeze(1)
                target = target.unsqueeze(1)
        
        # Handle multi-class case
        else:
            # Convert logits to probabilities if needed
            if pred.dim() == 5 and pred.shape[1] > 1:  # Multi-class
                pred = torch.softmax(pred, dim=1)
                
                # Convert target to one-hot if needed
                if target.dim() == 4:  # (B, D, H, W)
                    target = torch.nn.functional.one_hot(
                        target.long(), num_classes=pred.shape[1]
                    )
                    target = target.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Select classes
        if not self.include_background and pred.shape[1] > 1:
            pred = pred[:, 1:]
            target = target[:, 1:]
        
        # Flatten spatial dimensions
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)  # (B, C, N)
        target_flat = target.view(target.shape[0], target.shape[1], -1)  # (B, C, N)
        
        # Compute Dice coefficient
        intersection = torch.sum(pred_flat * target_flat, dim=2)  # (B, C)
        union = torch.sum(pred_flat, dim=2) + torch.sum(target_flat, dim=2)  # (B, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        
        # Apply reduction
        if self.reduction == "mean":
            return dice.mean()
        elif self.reduction == "sum":
            return dice.sum()
        else:
            return dice


class MultiClassDiceMetric(nn.Module):
    """Multi-class Dice metric with per-class computation."""
    
    def __init__(
        self,
        num_classes: int,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean_batch",
    ):
        """
        Initialize multi-class Dice metric.
        
        Args:
            num_classes: Number of classes
            include_background: Whether to include background class
            smooth: Smoothing factor
            reduction: Reduction method ('mean_batch', 'mean_class', 'none')
        """
        super().__init__()
        self.num_classes = num_classes
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute per-class Dice coefficients.
        
        Args:
            pred: Predicted probabilities (B, C, D, H, W) or logits
            target: Ground truth labels (B, D, H, W) or (B, C, D, H, W)
            
        Returns:
            Dice coefficients per class or aggregated
        """
        # Convert logits to probabilities
        if not torch.allclose(pred.sum(dim=1), torch.ones_like(pred.sum(dim=1))):
            pred = torch.softmax(pred, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 4:  # (B, D, H, W)
            target = torch.nn.functional.one_hot(target.long(), num_classes=self.num_classes)
            target = target.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Select classes
        start_idx = 0 if self.include_background else 1
        pred_classes = pred[:, start_idx:]
        target_classes = target[:, start_idx:]
        
        # Flatten spatial dimensions
        pred_flat = pred_classes.view(pred_classes.shape[0], pred_classes.shape[1], -1)
        target_flat = target_classes.view(target_classes.shape[0], target_classes.shape[1], -1)
        
        # Compute per-class Dice
        intersection = torch.sum(pred_flat * target_flat, dim=2)  # (B, C)
        union = torch.sum(pred_flat, dim=2) + torch.sum(target_flat, dim=2)  # (B, C)
        
        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        
        # Apply reduction
        if self.reduction == "mean_batch":
            return dice_per_class.mean(dim=0)  # (C,)
        elif self.reduction == "mean_class":
            return dice_per_class.mean(dim=1)  # (B,)
        elif self.reduction == "mean":
            return dice_per_class.mean()
        else:
            return dice_per_class  # (B, C)


def compute_dice_score(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> float:
    """
    Compute Dice score for numpy arrays.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor
        
    Returns:
        Dice score
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    
    if pred.sum() == 0 and target.sum() == 0:
        return 1.0
    
    intersection = np.sum(pred & target)
    union = np.sum(pred) + np.sum(target)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def compute_multiclass_dice(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    include_background: bool = False,
    smooth: float = 1e-6,
) -> Dict[int, float]:
    """
    Compute per-class Dice scores for numpy arrays.
    
    Args:
        pred: Predicted class labels
        target: Ground truth class labels
        num_classes: Number of classes
        include_background: Whether to include background class
        smooth: Smoothing factor
        
    Returns:
        Dictionary of class -> dice score
    """
    dice_scores = {}
    
    start_class = 0 if include_background else 1
    
    for class_id in range(start_class, num_classes):
        pred_binary = (pred == class_id)
        target_binary = (target == class_id)
        
        dice = compute_dice_score(pred_binary, target_binary, smooth)
        dice_scores[class_id] = dice
    
    return dice_scores


def compute_brats_dice(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Compute BraTS-specific Dice scores for tumor regions.
    
    Args:
        pred: Predicted segmentation (with BraTS labels)
        target: Ground truth segmentation (with BraTS labels)
        smooth: Smoothing factor
        
    Returns:
        Dictionary with WT, TC, ET Dice scores
    """
    # BraTS label mapping
    # 0: Background
    # 1: NCR (Non-enhancing tumor core)
    # 2: ED (Edema)
    # 4: ET (Enhancing tumor)
    
    # Define tumor regions
    regions = {
        "WT": [1, 2, 4],  # Whole tumor
        "TC": [1, 4],     # Tumor core
        "ET": [4],        # Enhancing tumor
    }
    
    dice_scores = {}
    
    for region_name, labels in regions.items():
        pred_region = np.isin(pred, labels)
        target_region = np.isin(target, labels)
        
        dice = compute_dice_score(pred_region, target_region, smooth)
        dice_scores[region_name] = dice
    
    return dice_scores


class BraTSDiceMetric(nn.Module):
    """BraTS-specific Dice metric for tumor regions."""
    
    def __init__(self, smooth: float = 1e-6):
        """
        Initialize BraTS Dice metric.
        
        Args:
            smooth: Smoothing factor
        """
        super().__init__()
        self.smooth = smooth
        
        # BraTS regions
        self.regions = {
            "WT": [1, 2, 4],  # Whole tumor
            "TC": [1, 4],     # Tumor core  
            "ET": [4],        # Enhancing tumor
        }
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BraTS Dice scores.
        
        Args:
            pred: Predicted segmentation (B, C, D, H, W) or (B, D, H, W)
            target: Ground truth segmentation (B, D, H, W)
            
        Returns:
            Dictionary with region Dice scores
        """
        # Convert predictions to class labels
        if pred.dim() == 5:  # Multi-class output
            pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
        
        dice_scores = {}
        
        for region_name, labels in self.regions.items():
            # Create binary masks for region
            pred_region = torch.zeros_like(pred, dtype=torch.bool)
            target_region = torch.zeros_like(target, dtype=torch.bool)
            
            for label in labels:
                pred_region |= (pred == label)
                target_region |= (target == label)
            
            # Compute Dice
            pred_region = pred_region.float()
            target_region = target_region.float()
            
            # Flatten
            pred_flat = pred_region.view(pred_region.shape[0], -1)
            target_flat = target_region.view(target_region.shape[0], -1)
            
            # Compute Dice coefficient
            intersection = torch.sum(pred_flat * target_flat, dim=1)  # (B,)
            union = torch.sum(pred_flat, dim=1) + torch.sum(target_flat, dim=1)  # (B,)
            
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B,)
            dice_scores[region_name] = dice.mean()  # Average over batch
        
        return dice_scores