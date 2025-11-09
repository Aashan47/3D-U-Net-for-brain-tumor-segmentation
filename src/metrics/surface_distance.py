import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Optional, Tuple
import warnings


class SurfaceDistanceMetrics(nn.Module):
    """Comprehensive surface distance metrics for segmentation evaluation."""
    
    def __init__(
        self,
        spacing: tuple = (1.0, 1.0, 1.0),
        connectivity: int = 1,
    ):
        """
        Initialize surface distance metrics.
        
        Args:
            spacing: Voxel spacing in mm
            connectivity: Connectivity for surface extraction
        """
        super().__init__()
        self.spacing = np.array(spacing)
        self.connectivity = connectivity
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive surface distance metrics.
        
        Args:
            pred: Predicted mask (B, D, H, W) or (B, C, D, H, W)
            target: Ground truth mask (B, D, H, W)
            
        Returns:
            Dictionary with various surface distance metrics
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
        results = {
            "mean_surface_distance": [],
            "rms_surface_distance": [],
            "max_surface_distance": [],
            "hausdorff_distance": [],
            "hausdorff_95": [],
            "avg_symmetric_surface_distance": [],
        }
        
        for i in range(batch_size):
            metrics = self._compute_surface_distances(
                pred[i].astype(bool), 
                target[i].astype(bool)
            )
            
            for key, value in metrics.items():
                results[key].append(value)
        
        # Convert to tensors and compute batch statistics
        final_results = {}
        for key, values in results.items():
            # Filter out infinite values
            finite_values = [v for v in values if np.isfinite(v)]
            if finite_values:
                final_results[key] = torch.tensor(np.mean(finite_values))
            else:
                final_results[key] = torch.tensor(float('inf'))
        
        return final_results
    
    def _compute_surface_distances(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> Dict[str, float]:
        """
        Compute surface distance metrics for a single pair of masks.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            Dictionary with surface distance metrics
        """
        if not pred.any() and not target.any():
            return {
                "mean_surface_distance": 0.0,
                "rms_surface_distance": 0.0,
                "max_surface_distance": 0.0,
                "hausdorff_distance": 0.0,
                "hausdorff_95": 0.0,
                "avg_symmetric_surface_distance": 0.0,
            }
        
        if not pred.any() or not target.any():
            return {
                "mean_surface_distance": float('inf'),
                "rms_surface_distance": float('inf'),
                "max_surface_distance": float('inf'),
                "hausdorff_distance": float('inf'),
                "hausdorff_95": float('inf'),
                "avg_symmetric_surface_distance": float('inf'),
            }
        
        try:
            # Compute distance transforms
            pred_dt = distance_transform_edt(~pred, sampling=self.spacing)
            target_dt = distance_transform_edt(~target, sampling=self.spacing)
            
            # Get surface points
            pred_surface = self._get_surface_mask(pred)
            target_surface = self._get_surface_mask(target)
            
            if not pred_surface.any() or not target_surface.any():
                return {
                    "mean_surface_distance": float('inf'),
                    "rms_surface_distance": float('inf'),
                    "max_surface_distance": float('inf'),
                    "hausdorff_distance": float('inf'),
                    "hausdorff_95": float('inf'),
                    "avg_symmetric_surface_distance": float('inf'),
                }
            
            # Surface distances from pred to target
            pred_to_target = pred_dt[pred_surface]
            
            # Surface distances from target to pred
            target_to_pred = target_dt[target_surface]
            
            # Combine all distances for symmetric metrics
            all_distances = np.concatenate([pred_to_target, target_to_pred])
            
            # Compute metrics
            metrics = {
                "mean_surface_distance": float(np.mean(pred_to_target)),
                "rms_surface_distance": float(np.sqrt(np.mean(pred_to_target ** 2))),
                "max_surface_distance": float(np.max(pred_to_target)),
                "hausdorff_distance": float(max(np.max(pred_to_target), np.max(target_to_pred))),
                "hausdorff_95": float(np.percentile(all_distances, 95)),
                "avg_symmetric_surface_distance": float(np.mean(all_distances)),
            }
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"Error computing surface distances: {e}")
            return {
                "mean_surface_distance": float('inf'),
                "rms_surface_distance": float('inf'),
                "max_surface_distance": float('inf'),
                "hausdorff_distance": float('inf'),
                "hausdorff_95": float('inf'),
                "avg_symmetric_surface_distance": float('inf'),
            }
    
    def _get_surface_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract surface mask from binary mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Binary surface mask
        """
        from scipy.ndimage import binary_erosion
        
        if not mask.any():
            return np.zeros_like(mask, dtype=bool)
        
        # Surface is the difference between original and eroded mask
        eroded = binary_erosion(mask, structure=np.ones((3, 3, 3)))
        surface = mask & ~eroded
        
        return surface


class AverageSymmetricSurfaceDistance(nn.Module):
    """Average Symmetric Surface Distance (ASSD) metric."""
    
    def __init__(
        self,
        spacing: tuple = (1.0, 1.0, 1.0),
        connectivity: int = 1,
    ):
        """
        Initialize ASSD metric.
        
        Args:
            spacing: Voxel spacing
            connectivity: Surface connectivity
        """
        super().__init__()
        self.spacing = spacing
        self.connectivity = connectivity
        self.surface_metrics = SurfaceDistanceMetrics(spacing, connectivity)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Average Symmetric Surface Distance.
        
        Args:
            pred: Predicted mask
            target: Ground truth mask
            
        Returns:
            ASSD value
        """
        metrics = self.surface_metrics(pred, target)
        return metrics["avg_symmetric_surface_distance"]


class SurfaceDiceMetric(nn.Module):
    """Surface Dice metric for boundary-focused evaluation."""
    
    def __init__(
        self,
        tolerance: float = 1.0,
        spacing: tuple = (1.0, 1.0, 1.0),
    ):
        """
        Initialize Surface Dice metric.
        
        Args:
            tolerance: Distance tolerance in mm
            spacing: Voxel spacing
        """
        super().__init__()
        self.tolerance = tolerance
        self.spacing = np.array(spacing)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Surface Dice coefficient.
        
        Args:
            pred: Predicted mask (B, D, H, W) or (B, C, D, H, W)
            target: Ground truth mask (B, D, H, W)
            
        Returns:
            Surface Dice coefficient
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
        surface_dice_scores = []
        
        for i in range(batch_size):
            score = self._compute_surface_dice(
                pred[i].astype(bool), 
                target[i].astype(bool)
            )
            surface_dice_scores.append(score)
        
        return torch.tensor(np.mean(surface_dice_scores))
    
    def _compute_surface_dice(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """
        Compute Surface Dice for a single pair of masks.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
            
        Returns:
            Surface Dice coefficient
        """
        if not pred.any() and not target.any():
            return 1.0
        
        if not pred.any() or not target.any():
            return 0.0
        
        try:
            # Get surface points
            pred_surface = self._get_surface_points(pred)
            target_surface = self._get_surface_points(target)
            
            if len(pred_surface) == 0 or len(target_surface) == 0:
                return 0.0
            
            # Compute distances from pred surface to target surface
            pred_distances = self._compute_point_distances(pred_surface, target_surface)
            
            # Compute distances from target surface to pred surface  
            target_distances = self._compute_point_distances(target_surface, pred_surface)
            
            # Count points within tolerance
            pred_within_tolerance = np.sum(pred_distances <= self.tolerance)
            target_within_tolerance = np.sum(target_distances <= self.tolerance)
            
            # Surface Dice coefficient
            total_surface_points = len(pred_surface) + len(target_surface)
            surface_dice = (pred_within_tolerance + target_within_tolerance) / total_surface_points
            
            return surface_dice
            
        except Exception as e:
            warnings.warn(f"Error computing Surface Dice: {e}")
            return 0.0
    
    def _get_surface_points(self, mask: np.ndarray) -> np.ndarray:
        """Get surface point coordinates."""
        from scipy.ndimage import binary_erosion
        
        if not mask.any():
            return np.array([]).reshape(0, 3)
        
        # Extract surface
        eroded = binary_erosion(mask)
        surface = mask & ~eroded
        
        # Get coordinates
        coords = np.where(surface)
        if len(coords[0]) == 0:
            return np.array([]).reshape(0, 3)
        
        # Apply spacing
        surface_points = np.stack(coords, axis=1).astype(float)
        surface_points *= self.spacing
        
        return surface_points
    
    def _compute_point_distances(
        self,
        points1: np.ndarray,
        points2: np.ndarray,
    ) -> np.ndarray:
        """Compute minimum distances from points1 to points2."""
        if len(points1) == 0 or len(points2) == 0:
            return np.array([])
        
        # Compute pairwise distances
        diff = points1[:, None, :] - points2[None, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Return minimum distance for each point in points1
        return np.min(distances, axis=1)


def compute_normalized_surface_dice(
    pred: np.ndarray,
    target: np.ndarray,
    tolerance: float = 1.0,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> float:
    """
    Compute Normalized Surface Dice (NSD) metric.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        tolerance: Distance tolerance
        spacing: Voxel spacing
        
    Returns:
        NSD score
    """
    surface_dice_metric = SurfaceDiceMetric(tolerance=tolerance, spacing=spacing)
    
    # Add batch dimension
    pred_tensor = torch.from_numpy(pred[None, ...])
    target_tensor = torch.from_numpy(target[None, ...])
    
    return surface_dice_metric(pred_tensor, target_tensor).item()