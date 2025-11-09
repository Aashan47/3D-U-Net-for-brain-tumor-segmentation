import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from captum.attr import IntegratedGradients, NoiseTunnel
from tqdm import tqdm


class IntegratedGradients3D:
    """3D Integrated Gradients implementation for medical image analysis."""
    
    def __init__(
        self,
        model: nn.Module,
        baseline_type: str = "zero",
        n_steps: int = 50,
        internal_batch_size: Optional[int] = None,
    ):
        """
        Initialize 3D Integrated Gradients.
        
        Args:
            model: The model to analyze
            baseline_type: Type of baseline ('zero', 'random', 'blur')
            n_steps: Number of integration steps
            internal_batch_size: Batch size for internal computation
        """
        self.model = model
        self.baseline_type = baseline_type
        self.n_steps = n_steps
        self.internal_batch_size = internal_batch_size
        
        # Initialize Captum IntegratedGradients
        self.ig = IntegratedGradients(model)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _create_baseline(
        self,
        input_tensor: torch.Tensor,
        baseline_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Create baseline for integrated gradients.
        
        Args:
            input_tensor: Input tensor
            baseline_type: Type of baseline to create
            
        Returns:
            Baseline tensor
        """
        if baseline_type is None:
            baseline_type = self.baseline_type
        
        if baseline_type == "zero":
            return torch.zeros_like(input_tensor)
        
        elif baseline_type == "random":
            return torch.randn_like(input_tensor) * 0.1
        
        elif baseline_type == "blur":
            # Apply Gaussian blur as baseline
            from scipy.ndimage import gaussian_filter
            
            baseline = input_tensor.clone()
            
            # Process each channel separately
            for b in range(baseline.shape[0]):
                for c in range(baseline.shape[1]):
                    blurred = gaussian_filter(
                        baseline[b, c].cpu().numpy(),
                        sigma=2.0,
                        mode='constant'
                    )
                    baseline[b, c] = torch.from_numpy(blurred).to(baseline.device)
            
            return baseline
        
        elif baseline_type == "mean":
            # Use mean intensity as baseline
            mean_vals = torch.mean(input_tensor, dim=(2, 3, 4), keepdim=True)
            return torch.full_like(input_tensor, 0) + mean_vals
        
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}")
    
    def generate_attributions(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_mask: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
        return_convergence_delta: bool = False,
    ) -> Union[torch.Tensor, tuple]:
        """
        Generate Integrated Gradients attributions.
        
        Args:
            input_tensor: Input tensor (1, C, D, H, W)
            target_class: Target class for attribution
            target_mask: Target segmentation mask
            baseline: Custom baseline (if None, will create one)
            return_convergence_delta: Whether to return convergence delta
            
        Returns:
            Attribution tensor or (attribution, convergence_delta)
        """
        # Ensure batch dimension
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        # Create baseline if not provided
        if baseline is None:
            baseline = self._create_baseline(input_tensor)
        
        # Define target for attribution
        if target_mask is not None:
            # Use mask as target
            def target_fn(output):
                if target_mask.dim() == 3:
                    mask = target_mask.unsqueeze(0).unsqueeze(0)
                elif target_mask.dim() == 4:
                    mask = target_mask.unsqueeze(1)
                else:
                    mask = target_mask
                
                # Compute overlap with predicted segmentation
                if output.dim() == 5:  # Multi-class
                    pred_probs = F.softmax(output, dim=1)
                    return torch.sum(pred_probs * mask.float())
                else:
                    return torch.sum(output * mask.float())
            
            target = target_fn
            
        elif target_class is not None:
            target = target_class
            
        else:
            # Use maximum activation as target
            def target_fn(output):
                if output.dim() == 5:  # Multi-class
                    return torch.max(output, dim=1)[0].sum()
                else:
                    return output.sum()
            
            target = target_fn
        
        # Generate attributions
        attributions = self.ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target,
            n_steps=self.n_steps,
            internal_batch_size=self.internal_batch_size,
            return_convergence_delta=return_convergence_delta,
        )
        
        if return_convergence_delta:
            attributions, convergence_delta = attributions
            return attributions.squeeze(0), convergence_delta
        else:
            return attributions.squeeze(0)  # Remove batch dimension
    
    def generate_multi_class_attributions(
        self,
        input_tensor: torch.Tensor,
        target_classes: List[int],
        baseline: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Generate attributions for multiple classes.
        
        Args:
            input_tensor: Input tensor
            target_classes: List of target classes
            baseline: Custom baseline
            
        Returns:
            Dictionary of {class: attributions}
        """
        attributions = {}
        
        for class_idx in target_classes:
            attr = self.generate_attributions(
                input_tensor,
                target_class=class_idx,
                baseline=baseline,
            )
            attributions[class_idx] = attr
        
        return attributions
    
    def generate_with_noise_tunnel(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_mask: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None,
        nt_samples: int = 10,
        nt_type: str = "smoothgrad_sq",
        stdevs: float = 0.2,
    ) -> torch.Tensor:
        """
        Generate attributions with noise tunnel for robustness.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class
            target_mask: Target mask
            baseline: Custom baseline
            nt_samples: Number of noise tunnel samples
            nt_type: Noise tunnel type
            stdevs: Standard deviation for noise
            
        Returns:
            Robust attribution tensor
        """
        # Create noise tunnel wrapper
        noise_tunnel = NoiseTunnel(self.ig)
        
        # Create baseline if not provided
        if baseline is None:
            baseline = self._create_baseline(input_tensor)
        
        # Define target
        if target_mask is not None:
            def target_fn(output):
                if target_mask.dim() == 3:
                    mask = target_mask.unsqueeze(0).unsqueeze(0)
                elif target_mask.dim() == 4:
                    mask = target_mask.unsqueeze(1)
                else:
                    mask = target_mask
                
                if output.dim() == 5:
                    pred_probs = F.softmax(output, dim=1)
                    return torch.sum(pred_probs * mask.float())
                else:
                    return torch.sum(output * mask.float())
            
            target = target_fn
        else:
            target = target_class
        
        # Generate attributions with noise tunnel
        attributions = noise_tunnel.attribute(
            input_tensor,
            baselines=baseline,
            target=target,
            n_steps=self.n_steps,
            nt_samples=nt_samples,
            nt_type=nt_type,
            stdevs=stdevs,
            internal_batch_size=self.internal_batch_size,
        )
        
        return attributions.squeeze(0)


class SlidingWindowIG:
    """Sliding window Integrated Gradients for large volumes."""
    
    def __init__(
        self,
        model: nn.Module,
        window_size: tuple = (128, 128, 128),
        overlap: float = 0.5,
        n_steps: int = 25,  # Fewer steps for efficiency
        baseline_type: str = "zero",
    ):
        """
        Initialize sliding window IG.
        
        Args:
            model: Model to analyze
            window_size: Window size for sliding
            overlap: Overlap between windows
            n_steps: Integration steps
            baseline_type: Baseline type
        """
        self.model = model
        self.window_size = window_size
        self.overlap = overlap
        self.n_steps = n_steps
        self.baseline_type = baseline_type
        
        self.ig = IntegratedGradients3D(
            model, 
            baseline_type=baseline_type, 
            n_steps=n_steps
        )
    
    def generate_attributions(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate attributions using sliding window.
        
        Args:
            input_tensor: Large input tensor
            target_class: Target class
            show_progress: Whether to show progress bar
            
        Returns:
            Full-resolution attributions
        """
        from monai.inferers import sliding_window_inference
        
        def ig_predictor(x):
            """Predictor function for sliding window."""
            attr = self.ig.generate_attributions(
                x,
                target_class=target_class,
            )
            # Return as prediction-like tensor
            return attr
        
        # Apply sliding window
        attributions = sliding_window_inference(
            inputs=input_tensor,
            roi_size=self.window_size,
            sw_batch_size=1,
            predictor=ig_predictor,
            overlap=self.overlap,
            mode="gaussian",
            progress=show_progress,
        )
        
        return attributions.squeeze(0)  # Remove batch dimension


def generate_ig_for_brats(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_classes: List[int] = [1, 2, 4],
    baseline_type: str = "zero",
    n_steps: int = 50,
    use_noise_tunnel: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Generate Integrated Gradients for BraTS tumor classes.
    
    Args:
        model: Trained BraTS model
        input_tensor: Input image tensor
        target_classes: BraTS classes to analyze
        baseline_type: Type of baseline
        n_steps: Integration steps
        use_noise_tunnel: Whether to use noise tunnel
        
    Returns:
        Dictionary with attributions for each class
    """
    # Class name mapping
    class_names = {1: "NCR", 2: "ED", 4: "ET"}
    
    # Initialize IG
    ig = IntegratedGradients3D(
        model, 
        baseline_type=baseline_type, 
        n_steps=n_steps
    )
    
    results = {}
    
    for class_idx in target_classes:
        if use_noise_tunnel:
            attr = ig.generate_with_noise_tunnel(
                input_tensor,
                target_class=class_idx,
            )
        else:
            attr = ig.generate_attributions(
                input_tensor,
                target_class=class_idx,
            )
        
        class_name = class_names.get(class_idx, f"class_{class_idx}")
        results[class_name] = attr
    
    return results


def compute_attribution_metrics(
    attributions: torch.Tensor,
    input_tensor: torch.Tensor,
    percentile_threshold: float = 95.0,
) -> Dict[str, float]:
    """
    Compute metrics for attribution quality.
    
    Args:
        attributions: Attribution tensor
        input_tensor: Original input tensor
        percentile_threshold: Percentile for top attributions
        
    Returns:
        Dictionary with attribution metrics
    """
    # Convert to numpy for easier computation
    attr_np = attributions.cpu().numpy()
    input_np = input_tensor.cpu().numpy()
    
    # Compute basic statistics
    metrics = {
        "mean_attribution": float(np.mean(np.abs(attr_np))),
        "max_attribution": float(np.max(np.abs(attr_np))),
        "min_attribution": float(np.min(attr_np)),
        "std_attribution": float(np.std(attr_np)),
    }
    
    # Compute percentile-based metrics
    abs_attr = np.abs(attr_np)
    threshold = np.percentile(abs_attr, percentile_threshold)
    
    top_attr_mask = abs_attr >= threshold
    metrics[f"top_{percentile_threshold}p_count"] = int(np.sum(top_attr_mask))
    metrics[f"top_{percentile_threshold}p_mean"] = float(np.mean(abs_attr[top_attr_mask]))
    
    # Compute spatial concentration (how concentrated are top attributions)
    if np.sum(top_attr_mask) > 0:
        # Center of mass for top attributions
        coords = np.where(top_attr_mask)
        if len(coords[0]) > 0:
            center_of_mass = [np.mean(c) for c in coords]
            
            # Compute spread (std of distances from center)
            distances = []
            for i in range(len(coords[0])):
                point = [coords[j][i] for j in range(len(coords))]
                dist = np.sqrt(sum((point[j] - center_of_mass[j])**2 for j in range(len(point))))
                distances.append(dist)
            
            metrics["spatial_concentration"] = float(np.std(distances))
        else:
            metrics["spatial_concentration"] = 0.0
    else:
        metrics["spatial_concentration"] = 0.0
    
    # Compute correlation with input intensity
    flat_attr = attr_np.flatten()
    flat_input = input_np.flatten()
    
    if len(flat_attr) > 0 and len(flat_input) > 0:
        correlation = np.corrcoef(np.abs(flat_attr), np.abs(flat_input))[0, 1]
        metrics["intensity_correlation"] = float(correlation if not np.isnan(correlation) else 0.0)
    else:
        metrics["intensity_correlation"] = 0.0
    
    return metrics


class AttributionEvaluator:
    """Evaluator for attribution quality and consistency."""
    
    def __init__(self):
        """Initialize attribution evaluator."""
        pass
    
    def evaluate_faithfulness(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        attributions: torch.Tensor,
        target_class: Optional[int] = None,
        perturbation_steps: int = 10,
    ) -> Dict[str, float]:
        """
        Evaluate faithfulness of attributions using perturbation.
        
        Args:
            model: Model to evaluate
            input_tensor: Original input
            attributions: Attribution values
            target_class: Target class
            perturbation_steps: Number of perturbation steps
            
        Returns:
            Faithfulness metrics
        """
        model.eval()
        
        # Get original prediction
        with torch.no_grad():
            original_output = model(input_tensor.unsqueeze(0) if input_tensor.dim() == 4 else input_tensor)
            if target_class is not None:
                original_score = original_output[0, target_class].item()
            else:
                original_score = original_output.max().item()
        
        # Sort attributions to get most important pixels
        abs_attr = torch.abs(attributions)
        flat_attr = abs_attr.flatten()
        sorted_indices = torch.argsort(flat_attr, descending=True)
        
        # Perform perturbation
        perturbed_scores = []
        step_size = len(sorted_indices) // perturbation_steps
        
        perturbed_input = input_tensor.clone()
        
        for step in range(perturbation_steps):
            # Zero out top attributions
            end_idx = (step + 1) * step_size
            indices_to_zero = sorted_indices[:end_idx]
            
            # Convert flat indices to 3D coordinates
            shape = attributions.shape
            coords = np.unravel_index(indices_to_zero.cpu().numpy(), shape)
            
            perturbed_input[coords] = 0
            
            # Get prediction on perturbed input
            with torch.no_grad():
                perturbed_output = model(perturbed_input.unsqueeze(0) if perturbed_input.dim() == 4 else perturbed_input)
                if target_class is not None:
                    perturbed_score = perturbed_output[0, target_class].item()
                else:
                    perturbed_score = perturbed_output.max().item()
            
            perturbed_scores.append(perturbed_score)
        
        # Compute faithfulness metrics
        score_drops = [original_score - score for score in perturbed_scores]
        
        metrics = {
            "faithfulness_correlation": float(np.corrcoef(range(len(score_drops)), score_drops)[0, 1]),
            "average_score_drop": float(np.mean(score_drops)),
            "max_score_drop": float(np.max(score_drops)),
            "area_under_curve": float(np.trapz(score_drops)),
        }
        
        return metrics