import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from scipy.stats import pearsonr


class VolumeCorrelationMetric(nn.Module):
    """Volume correlation metrics for segmentation evaluation."""
    
    def __init__(
        self,
        spacing: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Initialize volume correlation metric.
        
        Args:
            spacing: Voxel spacing for volume calculation in mm³
        """
        super().__init__()
        self.voxel_volume = np.prod(spacing)  # Volume per voxel in mm³
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute volume correlation metrics.
        
        Args:
            pred: Predicted mask (B, D, H, W) or (B, C, D, H, W)
            target: Ground truth mask (B, D, H, W)
            
        Returns:
            Dictionary with volume metrics
        """
        # Convert to numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle multi-class predictions
        if pred.ndim == 5:  # (B, C, D, H, W)
            pred = np.argmax(pred, axis=1)  # (B, D, H, W)
        
        batch_size = pred.shape[0]
        pred_volumes = []
        target_volumes = []
        volume_errors = []
        volume_error_percentages = []
        
        for i in range(batch_size):
            pred_vol = np.sum(pred[i] > 0) * self.voxel_volume
            target_vol = np.sum(target[i] > 0) * self.voxel_volume
            
            pred_volumes.append(pred_vol)
            target_volumes.append(target_vol)
            
            # Volume error
            vol_error = pred_vol - target_vol
            volume_errors.append(vol_error)
            
            # Volume error percentage
            if target_vol > 0:
                vol_error_pct = (vol_error / target_vol) * 100
            else:
                vol_error_pct = 100.0 if pred_vol > 0 else 0.0
            volume_error_percentages.append(vol_error_pct)
        
        # Compute correlation
        if len(pred_volumes) > 1 and np.std(pred_volumes) > 0 and np.std(target_volumes) > 0:
            correlation, p_value = pearsonr(pred_volumes, target_volumes)
        else:
            correlation = 1.0 if len(pred_volumes) == 1 and pred_volumes[0] == target_volumes[0] else 0.0
            p_value = 1.0
        
        return {
            "volume_correlation": torch.tensor(correlation),
            "volume_correlation_p_value": torch.tensor(p_value),
            "mean_volume_error": torch.tensor(np.mean(volume_errors)),
            "mean_volume_error_percentage": torch.tensor(np.mean(volume_error_percentages)),
            "std_volume_error": torch.tensor(np.std(volume_errors)),
            "mean_predicted_volume": torch.tensor(np.mean(pred_volumes)),
            "mean_target_volume": torch.tensor(np.mean(target_volumes)),
        }


class VolumeOverlapMetrics(nn.Module):
    """Volume overlap metrics including Jaccard and Volumetric Overlap Error."""
    
    def __init__(self):
        """Initialize volume overlap metrics."""
        super().__init__()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute volume overlap metrics.
        
        Args:
            pred: Predicted mask (B, D, H, W) or (B, C, D, H, W)
            target: Ground truth mask (B, D, H, W)
            
        Returns:
            Dictionary with overlap metrics
        """
        # Convert to binary if multi-class
        if pred.dim() == 5:  # Multi-class
            pred = torch.argmax(pred, dim=1) > 0  # Convert to binary
        else:
            pred = pred > 0.5
        
        target = target > 0
        
        # Flatten for computation
        pred_flat = pred.view(pred.shape[0], -1).float()
        target_flat = target.view(target.shape[0], -1).float()
        
        # Intersection and union
        intersection = torch.sum(pred_flat * target_flat, dim=1)
        union = torch.sum(pred_flat, dim=1) + torch.sum(target_flat, dim=1) - intersection
        
        # Volume sizes
        pred_vol = torch.sum(pred_flat, dim=1)
        target_vol = torch.sum(target_flat, dim=1)
        
        # Jaccard Index (IoU)
        jaccard = torch.where(union > 0, intersection / union, torch.ones_like(union))
        
        # Volumetric Overlap Error
        vol_overlap_error = torch.where(
            target_vol > 0,
            1 - (intersection / target_vol),
            torch.where(pred_vol > 0, torch.ones_like(target_vol), torch.zeros_like(target_vol))
        )
        
        # False Positive Rate
        false_pos_rate = torch.where(
            (pred_vol - intersection) + target_vol > 0,
            (pred_vol - intersection) / ((pred_vol - intersection) + target_vol),
            torch.zeros_like(pred_vol)
        )
        
        # False Negative Rate
        false_neg_rate = torch.where(
            target_vol > 0,
            (target_vol - intersection) / target_vol,
            torch.zeros_like(target_vol)
        )
        
        return {
            "jaccard": jaccard.mean(),
            "volumetric_overlap_error": vol_overlap_error.mean(),
            "false_positive_rate": false_pos_rate.mean(),
            "false_negative_rate": false_neg_rate.mean(),
        }


class SensitivitySpecificityMetric(nn.Module):
    """Sensitivity and Specificity metrics."""
    
    def __init__(self):
        """Initialize sensitivity/specificity metrics."""
        super().__init__()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute sensitivity and specificity.
        
        Args:
            pred: Predicted mask (B, D, H, W) or (B, C, D, H, W)
            target: Ground truth mask (B, D, H, W)
            
        Returns:
            Dictionary with sensitivity/specificity metrics
        """
        # Convert to binary
        if pred.dim() == 5:  # Multi-class
            pred = torch.argmax(pred, dim=1) > 0
        else:
            pred = pred > 0.5
        
        target = target > 0
        
        # Flatten
        pred_flat = pred.view(pred.shape[0], -1)
        target_flat = target.view(target.shape[0], -1)
        
        # True/False Positives/Negatives
        tp = torch.sum(pred_flat & target_flat, dim=1).float()
        fn = torch.sum(~pred_flat & target_flat, dim=1).float()
        fp = torch.sum(pred_flat & ~target_flat, dim=1).float()
        tn = torch.sum(~pred_flat & ~target_flat, dim=1).float()
        
        # Sensitivity (Recall, True Positive Rate)
        sensitivity = torch.where(tp + fn > 0, tp / (tp + fn), torch.ones_like(tp))
        
        # Specificity (True Negative Rate)
        specificity = torch.where(tn + fp > 0, tn / (tn + fp), torch.ones_like(tn))
        
        # Precision (Positive Predictive Value)
        precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.ones_like(tp))
        
        # F1 Score
        f1_score = torch.where(
            precision + sensitivity > 0,
            2 * (precision * sensitivity) / (precision + sensitivity),
            torch.zeros_like(precision)
        )
        
        # Balanced Accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        
        return {
            "sensitivity": sensitivity.mean(),
            "specificity": specificity.mean(),
            "precision": precision.mean(),
            "f1_score": f1_score.mean(),
            "balanced_accuracy": balanced_accuracy.mean(),
        }


class BraTSVolumeMetrics(nn.Module):
    """BraTS-specific volume metrics for tumor regions."""
    
    def __init__(
        self,
        spacing: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Initialize BraTS volume metrics.
        
        Args:
            spacing: Voxel spacing
        """
        super().__init__()
        self.voxel_volume = np.prod(spacing)
        
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
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute BraTS volume metrics.
        
        Args:
            pred: Predicted segmentation (B, C, D, H, W) or (B, D, H, W)
            target: Ground truth segmentation (B, D, H, W)
            
        Returns:
            Nested dictionary with region-specific volume metrics
        """
        # Convert predictions to class labels
        if pred.dim() == 5:  # Multi-class output
            pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
        
        # Convert to numpy for easier processing
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        batch_size = pred_np.shape[0]
        results = {}
        
        for region_name, labels in self.regions.items():
            pred_volumes = []
            target_volumes = []
            volume_errors = []
            
            for i in range(batch_size):
                # Create binary masks for region
                pred_region = np.isin(pred_np[i], labels)
                target_region = np.isin(target_np[i], labels)
                
                # Compute volumes
                pred_vol = np.sum(pred_region) * self.voxel_volume
                target_vol = np.sum(target_region) * self.voxel_volume
                
                pred_volumes.append(pred_vol)
                target_volumes.append(target_vol)
                volume_errors.append(pred_vol - target_vol)
            
            # Compute correlation
            if len(pred_volumes) > 1 and np.std(pred_volumes) > 0 and np.std(target_volumes) > 0:
                correlation, p_value = pearsonr(pred_volumes, target_volumes)
            else:
                correlation = 1.0 if pred_volumes[0] == target_volumes[0] else 0.0
                p_value = 1.0
            
            # Store results for this region
            results[region_name] = {
                "volume_correlation": torch.tensor(correlation),
                "mean_volume_error": torch.tensor(np.mean(volume_errors)),
                "std_volume_error": torch.tensor(np.std(volume_errors)),
                "mean_predicted_volume": torch.tensor(np.mean(pred_volumes)),
                "mean_target_volume": torch.tensor(np.mean(target_volumes)),
            }
        
        return results


def compute_volume_statistics(
    volumes_pred: np.ndarray,
    volumes_target: np.ndarray,
) -> Dict[str, float]:
    """
    Compute volume statistics for two sets of volumes.
    
    Args:
        volumes_pred: Predicted volumes
        volumes_target: Target volumes
        
    Returns:
        Dictionary with volume statistics
    """
    # Pearson correlation
    if len(volumes_pred) > 1 and np.std(volumes_pred) > 0 and np.std(volumes_target) > 0:
        correlation, p_value = pearsonr(volumes_pred, volumes_target)
    else:
        correlation = 1.0 if len(volumes_pred) == 1 and volumes_pred[0] == volumes_target[0] else 0.0
        p_value = 1.0
    
    # Volume errors
    volume_errors = volumes_pred - volumes_target
    volume_error_percentages = []
    
    for pred_vol, target_vol in zip(volumes_pred, volumes_target):
        if target_vol > 0:
            error_pct = ((pred_vol - target_vol) / target_vol) * 100
        else:
            error_pct = 100.0 if pred_vol > 0 else 0.0
        volume_error_percentages.append(error_pct)
    
    return {
        "correlation": correlation,
        "correlation_p_value": p_value,
        "mean_volume_error": np.mean(volume_errors),
        "std_volume_error": np.std(volume_errors),
        "mean_volume_error_percentage": np.mean(volume_error_percentages),
        "std_volume_error_percentage": np.std(volume_error_percentages),
        "mean_predicted_volume": np.mean(volumes_pred),
        "mean_target_volume": np.mean(volumes_target),
    }