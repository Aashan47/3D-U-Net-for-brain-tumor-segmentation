import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DiceLoss(nn.Module):
    """Dice loss for multi-class segmentation."""
    
    def __init__(
        self,
        include_background: bool = False,
        smooth: float = 1e-6,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Dice loss.
        
        Args:
            include_background: Whether to include background class
            smooth: Smoothing factor to avoid division by zero
            weight: Class weights for loss calculation
            reduction: Loss reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.weight = weight
        self.reduction = reduction
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            input: Predicted probabilities/logits (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            
        Returns:
            Dice loss
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
        
        # Compute Dice coefficient for each class
        intersection = torch.sum(input_flat * target_flat, dim=2)  # (B, C)
        union = torch.sum(input_flat, dim=2) + torch.sum(target_flat, dim=2)  # (B, C)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)
        
        # Compute Dice loss (1 - Dice)
        dice_loss = 1.0 - dice  # (B, C)
        
        # Apply class weights if provided
        if self.weight is not None:
            if self.weight.device != dice_loss.device:
                self.weight = self.weight.to(dice_loss.device)
            dice_loss = dice_loss * self.weight.unsqueeze(0)
        
        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss.mean(dim=0)  # Average over batch, keep class dimension


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice loss that handles class imbalance."""
    
    def __init__(
        self,
        include_background: bool = False,
        smooth: float = 1e-6,
        reduction: str = "mean",
    ):
        """
        Initialize Generalized Dice loss.
        
        Args:
            include_background: Whether to include background class
            smooth: Smoothing factor
            reduction: Loss reduction method
        """
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized Dice loss.
        
        Args:
            input: Predicted probabilities/logits (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            
        Returns:
            Generalized Dice loss
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
        
        # Compute class weights (inverse of squared class volumes)
        class_volumes = torch.sum(target_flat, dim=2)  # (B, C)
        class_weights = 1.0 / (class_volumes ** 2 + self.smooth)  # (B, C)
        
        # Compute weighted intersection and union
        intersection = torch.sum(class_weights * torch.sum(input_flat * target_flat, dim=2), dim=1)  # (B,)
        union = torch.sum(class_weights * (torch.sum(input_flat, dim=2) + torch.sum(target_flat, dim=2)), dim=1)  # (B,)
        
        # Compute Generalized Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B,)
        
        # Compute Generalized Dice loss
        dice_loss = 1.0 - dice  # (B,)
        
        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class SoftDiceLoss(nn.Module):
    """Soft Dice loss with optional squared terms."""
    
    def __init__(
        self,
        include_background: bool = False,
        smooth: float = 1e-6,
        squared_pred: bool = False,
        reduction: str = "mean",
    ):
        """
        Initialize Soft Dice loss.
        
        Args:
            include_background: Whether to include background class
            smooth: Smoothing factor
            squared_pred: Whether to square prediction terms
            reduction: Loss reduction method
        """
        super().__init__()
        self.include_background = include_background
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.reduction = reduction
    
    def forward(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Soft Dice loss.
        
        Args:
            input: Predicted probabilities/logits (B, C, D, H, W)
            target: Ground truth labels (B, C, D, H, W) or (B, D, H, W)
            
        Returns:
            Soft Dice loss
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
        
        # Compute intersection and union terms
        intersection = torch.sum(input_flat * target_flat, dim=2)  # (B, C)
        
        if self.squared_pred:
            pred_sum = torch.sum(input_flat ** 2, dim=2)  # (B, C)
        else:
            pred_sum = torch.sum(input_flat, dim=2)  # (B, C)
        
        target_sum = torch.sum(target_flat, dim=2)  # (B, C)
        
        # Compute Soft Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)  # (B, C)
        
        # Compute Soft Dice loss
        dice_loss = 1.0 - dice  # (B, C)
        
        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss.mean(dim=0)