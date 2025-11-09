import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .dice import DiceLoss
from .focal_tversky import FocalTverskyLoss


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross Entropy loss."""
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize combined Dice + BCE loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            bce_weight: Weight for BCE loss component
            class_weights: Class weights for BCE loss
            include_background: Whether to include background class
            smooth: Smoothing factor for Dice loss
            reduction: Loss reduction method
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.class_weights = class_weights
        self.include_background = include_background
        self.reduction = reduction
        
        # Initialize component losses
        self.dice_loss = DiceLoss(
            include_background=include_background,
            smooth=smooth,
            reduction=reduction,
        )
        
        self.bce_loss = nn.CrossEntropyLoss(
            weight=class_weights,
            reduction=reduction,
        )
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined Dice + BCE loss.
        
        Args:
            input: Predicted logits (B, C, D, H, W)
            target: Ground truth labels (B, D, H, W)
            
        Returns:
            Combined loss
        """
        # Compute Dice loss (expects probabilities)
        input_softmax = F.softmax(input, dim=1)
        dice_loss_val = self.dice_loss(input_softmax, target)
        
        # Compute BCE loss (expects logits)
        bce_loss_val = self.bce_loss(input, target.long())
        
        # Combine losses
        total_loss = self.dice_weight * dice_loss_val + self.bce_weight * bce_loss_val
        
        return total_loss


class DiceFocalLoss(nn.Module):
    """Combined Dice and Focal loss."""
    
    def __init__(
        self,
        dice_weight: float = 1.0,
        focal_weight: float = 1.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize combined Dice + Focal loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            focal_weight: Weight for Focal loss component
            alpha: Focal loss alpha parameter
            gamma: Focal loss gamma parameter
            include_background: Whether to include background class
            smooth: Smoothing factor for Dice loss
            reduction: Loss reduction method
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.alpha = alpha
        self.gamma = gamma
        self.include_background = include_background
        self.reduction = reduction
        
        # Initialize Dice loss
        self.dice_loss = DiceLoss(
            include_background=include_background,
            smooth=smooth,
            reduction=reduction,
        )
    
    def focal_loss(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute multi-class focal loss.
        
        Args:
            input: Predicted logits (B, C, D, H, W)
            target: Ground truth labels (B, D, H, W)
            
        Returns:
            Focal loss
        """
        # Convert to probabilities
        p = F.softmax(input, dim=1)
        
        # Convert target to one-hot
        target_one_hot = F.one_hot(target.long(), num_classes=input.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Select classes
        if not self.include_background:
            p = p[:, 1:]
            target_one_hot = target_one_hot[:, 1:]
        
        # Compute focal weight
        pt = torch.sum(p * target_one_hot, dim=1)  # (B, D, H, W)
        alpha_t = self.alpha
        focal_weight = alpha_t * torch.pow(1 - pt, self.gamma)
        
        # Compute cross-entropy
        log_pt = torch.sum(target_one_hot * torch.log(p + 1e-8), dim=1)  # (B, D, H, W)
        
        # Focal loss
        focal_loss = -focal_weight * log_pt
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined Dice + Focal loss.
        
        Args:
            input: Predicted logits (B, C, D, H, W)
            target: Ground truth labels (B, D, H, W)
            
        Returns:
            Combined loss
        """
        # Compute Dice loss (expects probabilities)
        input_softmax = F.softmax(input, dim=1)
        dice_loss_val = self.dice_loss(input_softmax, target)
        
        # Compute Focal loss
        focal_loss_val = self.focal_loss(input, target)
        
        # Combine losses
        total_loss = self.dice_weight * dice_loss_val + self.focal_weight * focal_loss_val
        
        return total_loss


class DiceTverskyLoss(nn.Module):
    """Combined Dice and Tversky loss."""
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        tversky_weight: float = 0.5,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize combined Dice + Focal Tversky loss.
        
        Args:
            dice_weight: Weight for Dice loss component
            tversky_weight: Weight for Focal Tversky loss component
            alpha: Tversky alpha parameter
            beta: Tversky beta parameter
            gamma: Focal gamma parameter
            include_background: Whether to include background class
            smooth: Smoothing factor
            reduction: Loss reduction method
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.tversky_weight = tversky_weight
        
        # Initialize component losses
        self.dice_loss = DiceLoss(
            include_background=include_background,
            smooth=smooth,
            reduction=reduction,
        )
        
        self.focal_tversky_loss = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            include_background=include_background,
            smooth=smooth,
            reduction=reduction,
        )
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined Dice + Focal Tversky loss.
        
        Args:
            input: Predicted probabilities/logits (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            
        Returns:
            Combined loss
        """
        # Compute Dice loss
        dice_loss_val = self.dice_loss(input, target)
        
        # Compute Focal Tversky loss
        tversky_loss_val = self.focal_tversky_loss(input, target)
        
        # Combine losses
        total_loss = self.dice_weight * dice_loss_val + self.tversky_weight * tversky_loss_val
        
        return total_loss


class WeightedLoss(nn.Module):
    """Wrapper for applying different weights to different loss functions."""
    
    def __init__(self, losses_and_weights: dict):
        """
        Initialize weighted combination of losses.
        
        Args:
            losses_and_weights: Dictionary of {loss_function: weight}
        """
        super().__init__()
        self.losses_and_weights = losses_and_weights
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted combination of losses.
        
        Args:
            input: Model predictions
            target: Ground truth
            
        Returns:
            Weighted loss
        """
        total_loss = 0.0
        
        for loss_fn, weight in self.losses_and_weights.items():
            loss_val = loss_fn(input, target)
            total_loss += weight * loss_val
        
        return total_loss


class DeepSupervisionLoss(nn.Module):
    """Loss function for deep supervision training."""
    
    def __init__(
        self,
        loss_function: nn.Module,
        weights: Optional[list] = None,
        reduction: str = "mean",
    ):
        """
        Initialize deep supervision loss.
        
        Args:
            loss_function: Base loss function to apply at each scale
            weights: Weights for each supervision level (final output gets highest weight)
            reduction: Loss reduction method
        """
        super().__init__()
        self.loss_function = loss_function
        self.weights = weights
        self.reduction = reduction
    
    def forward(
        self, 
        predictions: list, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute deep supervision loss.
        
        Args:
            predictions: List of predictions at different scales (finest to coarsest)
            target: Ground truth labels
            
        Returns:
            Weighted deep supervision loss
        """
        if self.weights is None:
            # Default weights: highest for final output, decreasing for auxiliary outputs
            num_outputs = len(predictions)
            self.weights = [0.5 ** (num_outputs - i - 1) for i in range(num_outputs)]
        
        total_loss = 0.0
        
        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            # Resize target to match prediction if needed
            if pred.shape[2:] != target.shape[1:]:
                target_resized = F.interpolate(
                    target.float().unsqueeze(1),
                    size=pred.shape[2:],
                    mode="nearest"
                ).squeeze(1).long()
            else:
                target_resized = target
            
            # Compute loss at this scale
            loss_val = self.loss_function(pred, target_resized)
            total_loss += weight * loss_val
        
        return total_loss