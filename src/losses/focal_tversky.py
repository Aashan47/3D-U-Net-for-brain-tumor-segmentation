import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TverskyLoss(nn.Module):
    """Tversky loss for handling class imbalance in segmentation."""
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize Tversky loss.
        
        Args:
            alpha: Weight for false positives (higher alpha penalizes FP more)
            beta: Weight for false negatives (higher beta penalizes FN more)
            include_background: Whether to include background class
            smooth: Smoothing factor
            reduction: Loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
        
        # Ensure alpha + beta = 1 for proper interpretation
        if abs(alpha + beta - 1.0) > 1e-6:
            print(f"Warning: alpha + beta = {alpha + beta} != 1.0")
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Tversky loss.
        
        Args:
            input: Predicted probabilities/logits (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            
        Returns:
            Tversky loss
        """
        # Apply softmax if input contains logits
        if not torch.allclose(input.sum(dim=1), torch.ones_like(input.sum(dim=1))):
            input = F.softmax(input, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 4:  # (B, D, H, W)
            target = F.one_hot(target.long(), num_classes=input.shape[1])
            target = target.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Select classes to compute loss for
        if not self.include_background:
            input = input[:, 1:]
            target = target[:, 1:]
        
        # Flatten spatial dimensions
        input_flat = input.view(input.shape[0], input.shape[1], -1)  # (B, C, N)
        target_flat = target.view(target.shape[0], target.shape[1], -1)  # (B, C, N)
        
        # Compute Tversky terms
        true_positives = torch.sum(input_flat * target_flat, dim=2)  # (B, C)
        false_negatives = torch.sum(target_flat * (1 - input_flat), dim=2)  # (B, C)
        false_positives = torch.sum(input_flat * (1 - target_flat), dim=2)  # (B, C)
        
        # Compute Tversky index
        tversky = (true_positives + self.smooth) / (
            true_positives + self.alpha * false_positives + self.beta * false_negatives + self.smooth
        )  # (B, C)
        
        # Compute Tversky loss
        tversky_loss = 1.0 - tversky  # (B, C)
        
        # Apply reduction
        if self.reduction == "mean":
            return tversky_loss.mean()
        elif self.reduction == "sum":
            return tversky_loss.sum()
        else:
            return tversky_loss.mean(dim=0)


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss for handling hard examples and class imbalance."""
    
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.33,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize Focal Tversky loss.
        
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            gamma: Focusing parameter (higher gamma focuses more on hard examples)
            include_background: Whether to include background class
            smooth: Smoothing factor
            reduction: Loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
        
        # Base Tversky loss
        self.tversky_loss = TverskyLoss(
            alpha=alpha,
            beta=beta,
            include_background=include_background,
            smooth=smooth,
            reduction="none",  # Don't reduce here
        )
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Focal Tversky loss.
        
        Args:
            input: Predicted probabilities/logits (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            
        Returns:
            Focal Tversky loss
        """
        # Compute base Tversky loss
        tversky_loss = self.tversky_loss(input, target)  # (B, C) or (C,)
        
        # Apply focal weighting
        focal_tversky_loss = torch.pow(tversky_loss, self.gamma)
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_tversky_loss.mean()
        elif self.reduction == "sum":
            return focal_tversky_loss.sum()
        else:
            return focal_tversky_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for handling different penalties for FP and FN."""
    
    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 0.75,
        gamma: float = 2.0,
        include_background: bool = False,
        reduction: str = "mean",
    ):
        """
        Initialize Asymmetric loss.
        
        Args:
            alpha: Weight for positive class (foreground)
            beta: Weight for negative class (background)
            gamma: Focusing parameter
            include_background: Whether to include background class
            reduction: Loss reduction method
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.include_background = include_background
        self.reduction = reduction
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Asymmetric loss.
        
        Args:
            input: Predicted probabilities/logits (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            
        Returns:
            Asymmetric loss
        """
        # Apply softmax if input contains logits
        if not torch.allclose(input.sum(dim=1), torch.ones_like(input.sum(dim=1))):
            input = F.softmax(input, dim=1)
        
        # Convert target to one-hot if needed
        if target.dim() == 4:  # (B, D, H, W)
            target = F.one_hot(target.long(), num_classes=input.shape[1])
            target = target.permute(0, 4, 1, 2, 3).float()  # (B, C, D, H, W)
        
        # Select classes to compute loss for
        if not self.include_background:
            input = input[:, 1:]
            target = target[:, 1:]
        
        # Compute cross-entropy terms
        eps = 1e-8
        input_clamp = torch.clamp(input, eps, 1.0 - eps)
        
        # Positive term (when target = 1)
        pos_term = target * torch.log(input_clamp)
        pos_weight = self.alpha * torch.pow(1 - input_clamp, self.gamma)
        
        # Negative term (when target = 0)
        neg_term = (1 - target) * torch.log(1 - input_clamp)
        neg_weight = self.beta * torch.pow(input_clamp, self.gamma)
        
        # Combine terms
        loss = -(pos_weight * pos_term + neg_weight * neg_term)
        
        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss.mean(dim=(0, 2, 3, 4))  # Average over batch and spatial dims