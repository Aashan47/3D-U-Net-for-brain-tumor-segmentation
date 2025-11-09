import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Optional, Union
import warnings


class HausdorffDistanceMetric(nn.Module):
    """Hausdorff Distance metric for segmentation evaluation."""
    
    def __init__(
        self,
        percentile: float = 95.0,
        spacing: Optional[tuple] = None,
        reduction: str = "mean",
    ):
        """
        Initialize Hausdorff Distance metric.
        
        Args:
            percentile: Percentile for robust Hausdorff distance (95th percentile commonly used)
            spacing: Voxel spacing for distance calculation
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.percentile = percentile
        self.spacing = spacing if spacing is not None else (1.0, 1.0, 1.0)
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Hausdorff distance.
        
        Args:
            pred: Predicted binary mask (B, D, H, W) or (B, C, D, H, W)
            target: Ground truth binary mask (B, D, H, W)
            
        Returns:
            Hausdorff distance(s)
        """
        # Convert to numpy for scipy computation
        if isinstance(pred, torch.Tensor):
            pred = pred.detach().cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()
        
        # Handle multi-class predictions
        if pred.ndim == 5:  # (B, C, D, H, W)
            pred = np.argmax(pred, axis=1)  # (B, D, H, W)
        
        # Compute HD95 for each sample in batch
        hd_distances = []
        
        for i in range(pred.shape[0]):
            pred_i = pred[i].astype(bool)
            target_i = target[i].astype(bool)
            
            hd = self._compute_hausdorff_distance(pred_i, target_i)
            hd_distances.append(hd)
        
        hd_distances = torch.tensor(hd_distances, dtype=torch.float32)
        
        # Apply reduction
        if self.reduction == "mean":
            return hd_distances.mean()
        elif self.reduction == "sum":
            return hd_distances.sum()
        else:
            return hd_distances
    
    def _compute_hausdorff_distance(
        self, 
        pred: np.ndarray, 
        target: np.ndarray
    ) -> float:
        """
        Compute Hausdorff distance for a single pair of binary masks.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            Hausdorff distance
        """
        if not pred.any() and not target.any():
            return 0.0
        
        if not pred.any() or not target.any():
            # One mask is empty, return large distance
            return float('inf')
        
        try:
            # Compute surface points
            pred_surface = self._get_surface_points(pred)
            target_surface = self._get_surface_points(target)
            
            if len(pred_surface) == 0 or len(target_surface) == 0:
                return float('inf')
            
            # Compute directed Hausdorff distances
            hd1 = directed_hausdorff(pred_surface, target_surface)[0]
            hd2 = directed_hausdorff(target_surface, pred_surface)[0]
            
            # Symmetric Hausdorff distance
            distances = np.concatenate([
                np.sqrt(np.sum((pred_surface[:, None, :] - target_surface[None, :, :]) ** 2, axis=2)).min(axis=1),
                np.sqrt(np.sum((target_surface[:, None, :] - pred_surface[None, :, :]) ** 2, axis=2)).min(axis=1)
            ])
            
            # Return percentile-based distance
            if self.percentile < 100:
                return np.percentile(distances, self.percentile)
            else:
                return max(hd1, hd2)
                
        except Exception as e:
            warnings.warn(f"Error computing Hausdorff distance: {e}")
            return float('inf')
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract surface points from binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Array of surface point coordinates
        """
        # Compute distance transform
        if not mask.any():
            return np.array([]).reshape(0, 3)
        
        # Compute boundary by finding voxels adjacent to background
        from scipy.ndimage import binary_erosion
        
        eroded = binary_erosion(mask)
        boundary = mask & ~eroded
        
        # Get boundary coordinates
        coords = np.where(boundary)
        if len(coords[0]) == 0:
            return np.array([]).reshape(0, 3)
        
        # Stack coordinates and apply spacing
        surface_points = np.stack(coords, axis=1).astype(float)
        surface_points *= np.array(self.spacing)
        
        return surface_points


def compute_hausdorff_distance_95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute 95th percentile Hausdorff distance for numpy arrays.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask  
        spacing: Voxel spacing
        
    Returns:
        95th percentile Hausdorff distance
    """
    metric = HausdorffDistanceMetric(percentile=95.0, spacing=spacing)
    
    # Add batch dimension
    pred_tensor = torch.from_numpy(pred[None, ...])
    target_tensor = torch.from_numpy(target[None, ...])
    
    return metric(pred_tensor, target_tensor).item()


def compute_brats_hausdorff(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute Hausdorff distances for BraTS tumor regions.
    
    Args:
        pred: Predicted segmentation
        target: Ground truth segmentation
        spacing: Voxel spacing
        
    Returns:
        Dictionary with region HD95 scores
    """
    # BraTS regions
    regions = {
        "WT": [1, 2, 4],  # Whole tumor
        "TC": [1, 4],     # Tumor core
        "ET": [4],        # Enhancing tumor
    }
    
    hd_scores = {}
    
    for region_name, labels in regions.items():
        pred_region = np.isin(pred, labels)
        target_region = np.isin(target, labels)
        
        if not pred_region.any() and not target_region.any():
            hd = 0.0
        elif not pred_region.any() or not target_region.any():
            hd = float('inf')
        else:
            hd = compute_hausdorff_distance_95(pred_region, target_region, spacing)
        
        hd_scores[region_name] = hd
    
    return hd_scores


class BraTSHausdorffMetric(nn.Module):
    """BraTS-specific Hausdorff distance metric."""
    
    def __init__(
        self,
        percentile: float = 95.0,
        spacing: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Initialize BraTS Hausdorff metric.
        
        Args:
            percentile: Percentile for robust distance
            spacing: Voxel spacing
        """
        super().__init__()
        self.percentile = percentile
        self.spacing = spacing
        
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
        Compute BraTS Hausdorff distances.
        
        Args:
            pred: Predicted segmentation (B, C, D, H, W) or (B, D, H, W)
            target: Ground truth segmentation (B, D, H, W)
            
        Returns:
            Dictionary with region HD95 scores
        """
        # Convert predictions to class labels
        if pred.dim() == 5:  # Multi-class output
            pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
        
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        batch_size = pred_np.shape[0]
        hd_results = {region: [] for region in self.regions}
        
        # Process each sample in batch
        for i in range(batch_size):
            region_hd = compute_brats_hausdorff(
                pred_np[i], target_np[i], self.spacing
            )
            
            for region, hd in region_hd.items():
                hd_results[region].append(hd)
        
        # Convert to tensors and average over batch
        final_results = {}
        for region, hd_list in hd_results.items():
            # Handle infinite values
            finite_hd = [hd for hd in hd_list if np.isfinite(hd)]
            if finite_hd:
                final_results[region] = torch.tensor(np.mean(finite_hd))
            else:
                final_results[region] = torch.tensor(float('inf'))
        
        return final_results


class RobustHausdorffDistance(nn.Module):
    """Robust Hausdorff distance with outlier handling."""
    
    def __init__(
        self,
        percentile: float = 95.0,
        max_distance: float = 373.13,  # Common max for brain images
        spacing: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Initialize robust Hausdorff distance.
        
        Args:
            percentile: Percentile for robust calculation
            max_distance: Maximum distance to cap outliers
            spacing: Voxel spacing
        """
        super().__init__()
        self.percentile = percentile
        self.max_distance = max_distance
        self.spacing = spacing
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute robust Hausdorff distance.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            
        Returns:
            Robust Hausdorff distance
        """
        # Use regular Hausdorff distance computation
        hd_metric = HausdorffDistanceMetric(
            percentile=self.percentile,
            spacing=self.spacing,
            reduction="none"
        )
        
        distances = hd_metric(pred, target)
        
        # Cap distances at maximum value
        distances = torch.clamp(distances, max=self.max_distance)
        
        return distances.mean()