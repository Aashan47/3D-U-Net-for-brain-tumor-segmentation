import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union

from .dice import BraTSDiceMetric
from .hausdorff import BraTSHausdorffMetric
from .volume_metrics import BraTSVolumeMetrics
from .surface_distance import SurfaceDistanceMetrics


class BraTSMetrics(nn.Module):
    """Comprehensive BraTS evaluation metrics."""
    
    def __init__(
        self,
        spacing: tuple = (1.0, 1.0, 1.0),
        include_volume_metrics: bool = True,
        include_surface_metrics: bool = True,
    ):
        """
        Initialize BraTS metrics.
        
        Args:
            spacing: Voxel spacing in mm
            include_volume_metrics: Whether to compute volume metrics
            include_surface_metrics: Whether to compute surface metrics
        """
        super().__init__()
        self.spacing = spacing
        self.include_volume_metrics = include_volume_metrics
        self.include_surface_metrics = include_surface_metrics
        
        # Initialize metrics
        self.dice_metric = BraTSDiceMetric()
        self.hausdorff_metric = BraTSHausdorffMetric(spacing=spacing)
        
        if include_volume_metrics:
            self.volume_metric = BraTSVolumeMetrics(spacing=spacing)
        
        if include_surface_metrics:
            self.surface_metric = SurfaceDistanceMetrics(spacing=spacing)
        
        # BraTS regions
        self.regions = ["WT", "TC", "ET"]
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute comprehensive BraTS metrics.
        
        Args:
            pred: Predicted segmentation (B, C, D, H, W) or (B, D, H, W)
            target: Ground truth segmentation (B, D, H, W)
            
        Returns:
            Nested dictionary with all metrics organized by region
        """
        results = {}
        
        # Dice scores
        dice_scores = self.dice_metric(pred, target)
        
        # Hausdorff distances
        hd_scores = self.hausdorff_metric(pred, target)
        
        # Initialize results structure
        for region in self.regions:
            results[region] = {
                "dice": dice_scores[region],
                "hausdorff_95": hd_scores[region],
            }
        
        # Volume metrics
        if self.include_volume_metrics:
            volume_scores = self.volume_metric(pred, target)
            for region in self.regions:
                if region in volume_scores:
                    for metric_name, value in volume_scores[region].items():
                        results[region][metric_name] = value
        
        # Surface metrics (computed per region)
        if self.include_surface_metrics:
            surface_scores = self._compute_region_surface_metrics(pred, target)
            for region in self.regions:
                if region in surface_scores:
                    for metric_name, value in surface_scores[region].items():
                        results[region][metric_name] = value
        
        return results
    
    def _compute_region_surface_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute surface metrics for each BraTS region."""
        # Convert predictions to class labels
        if pred.dim() == 5:  # Multi-class output
            pred = torch.argmax(pred, dim=1)  # (B, D, H, W)
        
        # Convert to numpy
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        # BraTS region definitions
        region_labels = {
            "WT": [1, 2, 4],  # Whole tumor
            "TC": [1, 4],     # Tumor core
            "ET": [4],        # Enhancing tumor
        }
        
        results = {}
        
        for region_name, labels in region_labels.items():
            # Create binary masks for this region
            batch_size = pred_np.shape[0]
            region_results = []
            
            for i in range(batch_size):
                pred_region = np.isin(pred_np[i], labels).astype(np.uint8)
                target_region = np.isin(target_np[i], labels).astype(np.uint8)
                
                # Compute surface metrics for this sample
                pred_tensor = torch.from_numpy(pred_region[None, ...])
                target_tensor = torch.from_numpy(target_region[None, ...])
                
                region_metrics = self.surface_metric(pred_tensor, target_tensor)
                region_results.append(region_metrics)
            
            # Average across batch
            if region_results:
                aggregated = {}
                for key in region_results[0].keys():
                    values = [r[key] for r in region_results if torch.isfinite(r[key])]
                    if values:
                        aggregated[key] = torch.tensor(np.mean([v.item() for v in values]))
                    else:
                        aggregated[key] = torch.tensor(float('inf'))
                
                results[region_name] = aggregated
        
        return results
    
    def compute_summary_metrics(
        self,
        all_results: List[Dict[str, Dict[str, torch.Tensor]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary statistics across multiple evaluations.
        
        Args:
            all_results: List of results from multiple evaluations
            
        Returns:
            Summary statistics (mean, std) for each metric
        """
        summary = {}
        
        for region in self.regions:
            summary[region] = {}
            
            # Collect all metric values across evaluations
            metric_collections = {}
            
            for result in all_results:
                if region in result:
                    for metric_name, value in result[region].items():
                        if metric_name not in metric_collections:
                            metric_collections[metric_name] = []
                        
                        if torch.isfinite(value):
                            metric_collections[metric_name].append(value.item())
            
            # Compute summary statistics
            for metric_name, values in metric_collections.items():
                if values:
                    summary[region][f"{metric_name}_mean"] = np.mean(values)
                    summary[region][f"{metric_name}_std"] = np.std(values)
                    summary[region][f"{metric_name}_median"] = np.median(values)
                    summary[region][f"{metric_name}_min"] = np.min(values)
                    summary[region][f"{metric_name}_max"] = np.max(values)
                else:
                    summary[region][f"{metric_name}_mean"] = float('inf')
                    summary[region][f"{metric_name}_std"] = float('inf')
                    summary[region][f"{metric_name}_median"] = float('inf')
                    summary[region][f"{metric_name}_min"] = float('inf')
                    summary[region][f"{metric_name}_max"] = float('inf')
        
        return summary
    
    def results_to_dataframe(
        self,
        results: Dict[str, Dict[str, torch.Tensor]],
        subject_id: str,
        fold: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame for easy analysis.
        
        Args:
            results: Results dictionary
            subject_id: Subject identifier
            fold: Cross-validation fold (optional)
            
        Returns:
            DataFrame with results
        """
        rows = []
        
        for region, metrics in results.items():
            row = {
                "subject_id": subject_id,
                "region": region,
            }
            
            if fold is not None:
                row["fold"] = fold
            
            # Add all metrics
            for metric_name, value in metrics.items():
                if torch.isfinite(value):
                    row[metric_name] = value.item()
                else:
                    row[metric_name] = float('inf')
            
            rows.append(row)
        
        return pd.DataFrame(rows)


class BraTSEvaluator:
    """Complete BraTS evaluation pipeline."""
    
    def __init__(
        self,
        spacing: tuple = (1.0, 1.0, 1.0),
        save_individual_results: bool = True,
    ):
        """
        Initialize BraTS evaluator.
        
        Args:
            spacing: Voxel spacing
            save_individual_results: Whether to save per-subject results
        """
        self.spacing = spacing
        self.save_individual_results = save_individual_results
        self.metrics = BraTSMetrics(spacing=spacing)
        
        # Storage for results
        self.all_results = []
        self.subject_results = []
    
    def evaluate_subject(
        self,
        pred: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
        subject_id: str,
        fold: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a single subject.
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            subject_id: Subject identifier
            fold: Cross-validation fold
            
        Returns:
            Results dictionary
        """
        # Convert to tensors if needed
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if isinstance(target, np.ndarray):
            target = torch.from_numpy(target)
        
        # Add batch dimension if needed
        if pred.dim() == 3:  # (D, H, W)
            pred = pred.unsqueeze(0)
        if target.dim() == 3:  # (D, H, W)
            target = target.unsqueeze(0)
        
        # Compute metrics
        results = self.metrics(pred, target)
        
        # Convert to regular dict with float values
        results_dict = {}
        for region, metrics in results.items():
            results_dict[region] = {}
            for metric_name, value in metrics.items():
                results_dict[region][metric_name] = value.item() if torch.isfinite(value) else float('inf')
        
        # Store results
        if self.save_individual_results:
            self.all_results.append(results)
            
            # Create DataFrame row
            df_result = self.metrics.results_to_dataframe(
                results, subject_id, fold
            )
            self.subject_results.append(df_result)
        
        return results_dict
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics across all evaluated subjects."""
        if not self.all_results:
            return {}
        
        return self.metrics.compute_summary_metrics(self.all_results)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all results as a single DataFrame."""
        if not self.subject_results:
            return pd.DataFrame()
        
        return pd.concat(self.subject_results, ignore_index=True)
    
    def save_results(
        self, 
        output_path: str,
        include_summary: bool = True,
    ):
        """
        Save results to CSV files.
        
        Args:
            output_path: Output file path (without extension)
            include_summary: Whether to save summary statistics
        """
        # Save detailed results
        results_df = self.get_results_dataframe()
        if not results_df.empty:
            results_df.to_csv(f"{output_path}_detailed.csv", index=False)
        
        # Save summary statistics
        if include_summary:
            summary = self.get_summary_statistics()
            if summary:
                summary_rows = []
                for region, stats in summary.items():
                    row = {"region": region}
                    row.update(stats)
                    summary_rows.append(row)
                
                summary_df = pd.DataFrame(summary_rows)
                summary_df.to_csv(f"{output_path}_summary.csv", index=False)
    
    def reset(self):
        """Reset evaluator state."""
        self.all_results = []
        self.subject_results = []


def evaluate_brats_predictions(
    predictions_dir: str,
    ground_truth_dir: str,
    output_file: str,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> pd.DataFrame:
    """
    Evaluate BraTS predictions from directories.
    
    Args:
        predictions_dir: Directory with prediction files
        ground_truth_dir: Directory with ground truth files
        output_file: Output CSV file path
        spacing: Voxel spacing
        
    Returns:
        Results DataFrame
    """
    import glob
    import nibabel as nib
    from pathlib import Path
    
    evaluator = BraTSEvaluator(spacing=spacing)
    
    # Find all prediction files
    pred_files = glob.glob(os.path.join(predictions_dir, "*.nii.gz"))
    
    for pred_file in pred_files:
        subject_id = Path(pred_file).stem.replace(".nii", "")
        gt_file = os.path.join(ground_truth_dir, f"{subject_id}.nii.gz")
        
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth not found for {subject_id}")
            continue
        
        # Load images
        pred_img = nib.load(pred_file).get_fdata()
        gt_img = nib.load(gt_file).get_fdata()
        
        # Evaluate
        evaluator.evaluate_subject(pred_img, gt_img, subject_id)
    
    # Save results
    evaluator.save_results(output_file.replace('.csv', ''))
    
    return evaluator.get_results_dataframe()