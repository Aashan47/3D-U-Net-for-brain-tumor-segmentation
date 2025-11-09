import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm
import logging

from ..metrics.brats_metrics import BraTSMetrics
from monai.inferers import sliding_window_inference


class BraTSValidator:
    """Validator for BraTS 3D segmentation models."""
    
    def __init__(
        self,
        metrics: BraTSMetrics,
        device: torch.device,
        roi_size: tuple = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = "gaussian",
        sigma_scale: float = 0.125,
    ):
        """
        Initialize BraTS validator.
        
        Args:
            metrics: BraTS metrics instance
            device: Computation device
            roi_size: ROI size for sliding window inference
            sw_batch_size: Batch size for sliding window
            overlap: Overlap ratio for sliding window
            mode: Blending mode for sliding window ('gaussian', 'constant')
            sigma_scale: Gaussian sigma scale for blending
        """
        self.metrics = metrics
        self.device = device
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        self.sigma_scale = sigma_scale
        
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        model: nn.Module,
        data_loader,
        loss_function: Optional[nn.Module] = None,
        save_predictions: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run validation on the data loader.
        
        Args:
            model: Model to validate
            data_loader: Validation data loader
            loss_function: Loss function for validation loss calculation
            save_predictions: Whether to save prediction images
            output_dir: Directory to save predictions
            
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        
        total_loss = 0.0
        all_results = []
        num_samples = 0
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Validation", leave=False)
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                images = batch["image"].to(self.device, non_blocking=True)
                labels = batch["label"].to(self.device, non_blocking=True)
                
                # Get subject IDs if available
                subject_ids = batch.get("subject_id", [f"subject_{batch_idx}"])
                
                # Forward pass with sliding window inference
                predictions = self._sliding_window_inference(model, images)
                
                # Compute loss if provided
                if loss_function is not None:
                    loss = loss_function(predictions, labels)
                    total_loss += loss.item()
                
                # Compute metrics for each sample in batch
                batch_size = images.shape[0]
                for i in range(batch_size):
                    pred_i = predictions[i:i+1]  # Keep batch dimension
                    label_i = labels[i:i+1]     # Keep batch dimension
                    subject_id = subject_ids[i] if isinstance(subject_ids, list) else subject_ids
                    
                    # Compute metrics
                    sample_results = self.metrics(pred_i, label_i)
                    
                    # Store results
                    sample_results["subject_id"] = subject_id
                    all_results.append(sample_results)
                    
                    # Save predictions if requested
                    if save_predictions and output_dir is not None:
                        self._save_prediction(
                            pred_i[0],  # Remove batch dimension
                            label_i[0], # Remove batch dimension
                            subject_id,
                            output_dir,
                        )
                
                num_samples += batch_size
                
                # Update progress bar
                if loss_function is not None:
                    avg_loss = total_loss / (batch_idx + 1)
                    progress_bar.set_postfix({"Val Loss": f"{avg_loss:.4f}"})
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Add loss to results
        if loss_function is not None:
            aggregated_results["loss"] = total_loss / len(data_loader)
        
        # Log results
        self._log_results(aggregated_results)
        
        return aggregated_results
    
    def _sliding_window_inference(
        self,
        model: nn.Module,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform sliding window inference.
        
        Args:
            model: Model for inference
            images: Input images
            
        Returns:
            Predictions
        """
        predictions = sliding_window_inference(
            inputs=images,
            roi_size=self.roi_size,
            sw_batch_size=self.sw_batch_size,
            predictor=model,
            overlap=self.overlap,
            mode=self.mode,
            sigma_scale=self.sigma_scale,
            progress=False,
        )
        
        return predictions
    
    def _aggregate_results(
        self,
        all_results: List[Dict[str, Union[torch.Tensor, str]]],
    ) -> Dict[str, float]:
        """
        Aggregate results across all validation samples.
        
        Args:
            all_results: List of per-sample results
            
        Returns:
            Aggregated results
        """
        if not all_results:
            return {}
        
        # Initialize aggregation
        aggregated = {}
        
        # Get all metric names from first result (excluding subject_id)
        metric_names = [
            key for key in all_results[0].keys() 
            if key != "subject_id" and isinstance(all_results[0][key], dict)
        ]
        
        # Aggregate metrics by region
        for region in ["WT", "TC", "ET"]:
            for result in all_results:
                if region in result:
                    region_metrics = result[region]
                    for metric_name, value in region_metrics.items():
                        key = f"{region}_{metric_name}"
                        if key not in aggregated:
                            aggregated[key] = []
                        
                        if torch.isfinite(value):
                            aggregated[key].append(value.item())
        
        # Compute statistics
        final_results = {}
        for key, values in aggregated.items():
            if values:  # Only if we have finite values
                final_results[f"{key}_mean"] = np.mean(values)
                final_results[f"{key}_std"] = np.std(values)
                final_results[f"{key}_median"] = np.median(values)
                
                # Add individual region means for easier access
                if key.endswith("_dice"):
                    region = key.split("_")[0]
                    final_results[f"{region}_dice"] = np.mean(values)
                elif key.endswith("_hausdorff_95"):
                    region = key.split("_")[0]
                    final_results[f"{region}_hd95"] = np.mean(values)
        
        # Compute overall averages
        dice_metrics = ["WT_dice", "TC_dice", "ET_dice"]
        hd_metrics = ["WT_hd95", "TC_hd95", "ET_hd95"]
        
        if all(metric in final_results for metric in dice_metrics):
            final_results["dice_mean"] = np.mean([final_results[m] for m in dice_metrics])
        
        if all(metric in final_results for metric in hd_metrics):
            # Filter out infinite values for HD95 average
            hd_values = [final_results[m] for m in hd_metrics if np.isfinite(final_results[m])]
            if hd_values:
                final_results["hd95_mean"] = np.mean(hd_values)
        
        return final_results
    
    def _save_prediction(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        subject_id: str,
        output_dir: str,
    ):
        """
        Save prediction and ground truth images.
        
        Args:
            prediction: Predicted segmentation
            ground_truth: Ground truth segmentation
            subject_id: Subject identifier
            output_dir: Output directory
        """
        import nibabel as nib
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to class labels
        if prediction.dim() == 4:  # Multi-class output (C, D, H, W)
            pred_labels = torch.argmax(prediction, dim=0).cpu().numpy()
        else:
            pred_labels = prediction.cpu().numpy()
        
        gt_labels = ground_truth.cpu().numpy()
        
        # Save as NIfTI files
        pred_img = nib.Nifti1Image(pred_labels, affine=np.eye(4))
        gt_img = nib.Nifti1Image(gt_labels, affine=np.eye(4))
        
        pred_path = output_path / f"{subject_id}_prediction.nii.gz"
        gt_path = output_path / f"{subject_id}_ground_truth.nii.gz"
        
        nib.save(pred_img, str(pred_path))
        nib.save(gt_img, str(gt_path))
    
    def _log_results(self, results: Dict[str, float]):
        """
        Log validation results.
        
        Args:
            results: Validation results
        """
        self.logger.info("Validation Results:")
        
        # Log main metrics
        main_metrics = ["dice_mean", "WT_dice", "TC_dice", "ET_dice", "hd95_mean"]
        for metric in main_metrics:
            if metric in results:
                self.logger.info(f"  {metric}: {results[metric]:.4f}")
        
        # Log loss if available
        if "loss" in results:
            self.logger.info(f"  Validation Loss: {results['loss']:.4f}")


class BraTSInference:
    """Inference engine for BraTS models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        roi_size: tuple = (128, 128, 128),
        sw_batch_size: int = 4,
        overlap: float = 0.5,
        mode: str = "gaussian",
    ):
        """
        Initialize BraTS inference engine.
        
        Args:
            model: Trained model
            device: Computation device
            roi_size: ROI size for sliding window
            sw_batch_size: Batch size for sliding window
            overlap: Overlap ratio
            mode: Blending mode
        """
        self.model = model.to(device)
        self.device = device
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.mode = mode
        
        self.model.eval()
    
    def predict(
        self,
        image: torch.Tensor,
        apply_postprocessing: bool = True,
    ) -> torch.Tensor:
        """
        Predict segmentation for input image.
        
        Args:
            image: Input image tensor (C, D, H, W) or (1, C, D, H, W)
            apply_postprocessing: Whether to apply postprocessing
            
        Returns:
            Predicted segmentation
        """
        # Ensure batch dimension
        if image.dim() == 4:
            image = image.unsqueeze(0)  # Add batch dimension
        
        # Move to device
        image = image.to(self.device)
        
        with torch.no_grad():
            # Sliding window inference
            prediction = sliding_window_inference(
                inputs=image,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self.model,
                overlap=self.overlap,
                mode=self.mode,
                progress=False,
            )
            
            # Convert to class labels
            if prediction.shape[1] > 1:  # Multi-class
                prediction = torch.argmax(prediction, dim=1, keepdim=True)
            else:  # Binary
                prediction = (prediction > 0.5).float()
        
        # Apply postprocessing if requested
        if apply_postprocessing:
            prediction = self._apply_postprocessing(prediction)
        
        return prediction.squeeze(0)  # Remove batch dimension
    
    def _apply_postprocessing(
        self,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply postprocessing to predictions.
        
        Args:
            prediction: Raw predictions
            
        Returns:
            Postprocessed predictions
        """
        # Convert to numpy for processing
        pred_np = prediction.squeeze().cpu().numpy().astype(np.uint8)
        
        # Import postprocessing functions
        try:
            from ..postprocessing import postprocess_prediction
            pred_processed = postprocess_prediction(pred_np)
            
            # Convert back to tensor
            return torch.from_numpy(pred_processed).unsqueeze(0).unsqueeze(0).to(self.device)
        
        except ImportError:
            # Return original if postprocessing not available
            return prediction
    
    def predict_batch(
        self,
        images: torch.Tensor,
        apply_postprocessing: bool = True,
    ) -> torch.Tensor:
        """
        Predict segmentations for batch of images.
        
        Args:
            images: Input images tensor (B, C, D, H, W)
            apply_postprocessing: Whether to apply postprocessing
            
        Returns:
            Predicted segmentations (B, 1, D, H, W)
        """
        batch_size = images.shape[0]
        predictions = []
        
        for i in range(batch_size):
            pred = self.predict(images[i], apply_postprocessing)
            predictions.append(pred)
        
        return torch.stack(predictions)