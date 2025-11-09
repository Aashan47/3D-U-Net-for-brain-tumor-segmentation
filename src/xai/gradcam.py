import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Callable
import numpy as np
from captum.attr import GradCAM, LayerGradCAM
import warnings


class GradCAM3D:
    """3D Grad-CAM implementation for medical image segmentation."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
        use_guided_grads: bool = False,
    ):
        """
        Initialize 3D Grad-CAM.
        
        Args:
            model: The 3D model to analyze
            target_layer: Name of the target layer for Grad-CAM
            use_guided_grads: Whether to use guided gradients
        """
        self.model = model
        self.target_layer = target_layer
        self.use_guided_grads = use_guided_grads
        
        # Find target layer
        self.target_layer_module = self._find_target_layer()
        
        # Set up hooks
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        self._register_hooks()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _find_target_layer(self) -> nn.Module:
        """Find the target layer module by name."""
        modules = dict(self.model.named_modules())
        
        if self.target_layer not in modules:
            available_layers = list(modules.keys())
            raise ValueError(
                f"Target layer '{self.target_layer}' not found. "
                f"Available layers: {available_layers}"
            )
        
        return modules[self.target_layer]
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Register hooks
        self.hooks.append(
            self.target_layer_module.register_forward_hook(forward_hook)
        )
        self.hooks.append(
            self.target_layer_module.register_backward_hook(backward_hook)
        )
    
    def _clear_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_mask: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Generate Grad-CAM for the input tensor.
        
        Args:
            input_tensor: Input tensor (1, C, D, H, W)
            target_class: Target class for classification
            target_mask: Target segmentation mask for guided attention
            normalize: Whether to normalize the CAM
            
        Returns:
            Grad-CAM heatmap (D, H, W)
        """
        # Ensure input has batch dimension
        if input_tensor.dim() == 4:
            input_tensor = input_tensor.unsqueeze(0)
        
        input_tensor = input_tensor.requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Prepare target for backward pass
        if target_mask is not None:
            # Use segmentation mask as target
            if target_mask.dim() == 3:  # (D, H, W)
                target_mask = target_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
            elif target_mask.dim() == 4:  # (1, D, H, W) 
                target_mask = target_mask.unsqueeze(1)  # (1, 1, D, H, W)
            
            # Create target from mask
            target_tensor = target_mask.float()
            
        elif target_class is not None:
            # Use specific class
            if output.dim() == 5:  # Multi-class output (1, C, D, H, W)
                target_tensor = output[:, target_class:target_class+1]
            else:
                target_tensor = output
                
        else:
            # Use maximum activation
            if output.dim() == 5:  # Multi-class
                target_tensor = torch.max(output, dim=1, keepdim=True)[0]
            else:
                target_tensor = output
        
        # Backward pass
        self.model.zero_grad()
        target_tensor.sum().backward(retain_graph=True)
        
        # Generate CAM
        if self.gradients is None or self.activations is None:
            warnings.warn("Gradients or activations not captured. Check target layer.")
            return torch.zeros_like(input_tensor[0, 0])
        
        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)  # (1, C, 1, 1, 1)
        
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, D, H, W)
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Remove batch and channel dimensions
        cam = cam.squeeze()  # (D, H, W)
        
        # Normalize if requested
        if normalize and cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def generate_multi_class_cam(
        self,
        input_tensor: torch.Tensor,
        target_classes: List[int],
        normalize: bool = True,
    ) -> Dict[int, torch.Tensor]:
        """
        Generate Grad-CAM for multiple classes.
        
        Args:
            input_tensor: Input tensor
            target_classes: List of target classes
            normalize: Whether to normalize CAMs
            
        Returns:
            Dictionary of {class: cam}
        """
        cams = {}
        
        for class_idx in target_classes:
            cam = self.generate_cam(
                input_tensor,
                target_class=class_idx,
                normalize=normalize,
            )
            cams[class_idx] = cam
        
        return cams
    
    def __del__(self):
        """Clean up hooks when object is destroyed."""
        self._clear_hooks()


class LayerGradCAM3D:
    """3D Layer Grad-CAM using Captum."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ):
        """
        Initialize Layer Grad-CAM.
        
        Args:
            model: The model to analyze
            target_layer: Target layer module
        """
        self.model = model
        self.target_layer = target_layer
        self.grad_cam = LayerGradCAM(model, target_layer)
    
    def generate_attributions(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
        additional_forward_args: Optional[tuple] = None,
    ) -> torch.Tensor:
        """
        Generate Layer Grad-CAM attributions.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class index
            additional_forward_args: Additional arguments for forward pass
            
        Returns:
            Attribution tensor
        """
        attributions = self.grad_cam.attribute(
            input_tensor,
            target=target_class,
            additional_forward_args=additional_forward_args,
        )
        
        return attributions


class GuidedGradCAM3D:
    """Guided Grad-CAM for 3D medical images."""
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: str,
    ):
        """
        Initialize Guided Grad-CAM.
        
        Args:
            model: The model to analyze
            target_layer: Target layer name
        """
        self.model = model
        self.target_layer = target_layer
        self.grad_cam = GradCAM3D(model, target_layer, use_guided_grads=True)
        
        # Modify ReLU activations for guided backprop
        self._modify_relu_backward()
    
    def _modify_relu_backward(self):
        """Modify ReLU backward pass for guided gradients."""
        def guided_relu_backward_hook(module, grad_input, grad_output):
            """
            If there is a negative gradient, discard it.
            """
            if isinstance(module, nn.ReLU):
                return (F.relu(grad_input[0]),)
            return grad_input
        
        # Register hooks for all ReLU layers
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(guided_relu_backward_hook)
    
    def generate_guided_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate Guided Grad-CAM.
        
        Args:
            input_tensor: Input tensor
            target_class: Target class
            target_mask: Target mask
            
        Returns:
            Guided Grad-CAM result
        """
        # Generate regular Grad-CAM
        cam = self.grad_cam.generate_cam(
            input_tensor,
            target_class=target_class,
            target_mask=target_mask,
            normalize=True,
        )
        
        # Generate guided gradients
        input_tensor = input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        
        if target_class is not None:
            target_tensor = output[:, target_class]
        else:
            target_tensor = output.max(dim=1)[0]
        
        guided_grads = torch.autograd.grad(
            target_tensor.sum(),
            input_tensor,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Combine CAM with guided gradients
        guided_grads = guided_grads.squeeze()  # Remove batch dimension
        
        # Upsample CAM to input resolution
        cam_upsampled = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=guided_grads.shape[-3:],
            mode='trilinear',
            align_corners=False,
        ).squeeze()
        
        # Element-wise multiplication
        guided_cam = guided_grads * cam_upsampled
        
        return guided_cam


def generate_gradcam_for_brats(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str = "decoder.final_conv",
    target_classes: List[int] = [1, 2, 4],  # NCR, ED, ET
    normalize: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Generate Grad-CAM for BraTS tumor classes.
    
    Args:
        model: Trained BraTS model
        input_tensor: Input image tensor
        target_layer: Target layer for Grad-CAM
        target_classes: BraTS classes to analyze
        normalize: Whether to normalize CAMs
        
    Returns:
        Dictionary with CAMs for each class
    """
    # Class name mapping
    class_names = {1: "NCR", 2: "ED", 4: "ET"}
    
    # Initialize Grad-CAM
    grad_cam = GradCAM3D(model, target_layer)
    
    # Generate CAMs
    results = {}
    
    for class_idx in target_classes:
        cam = grad_cam.generate_cam(
            input_tensor,
            target_class=class_idx,
            normalize=normalize,
        )
        
        class_name = class_names.get(class_idx, f"class_{class_idx}")
        results[class_name] = cam
    
    return results


def sliding_window_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: str,
    window_size: tuple = (128, 128, 128),
    overlap: float = 0.5,
    target_class: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate Grad-CAM using sliding window for large volumes.
    
    Args:
        model: Model to analyze
        input_tensor: Large input tensor
        target_layer: Target layer name
        window_size: Sliding window size
        overlap: Window overlap ratio
        target_class: Target class
        
    Returns:
        Full-resolution Grad-CAM
    """
    from monai.inferers import sliding_window_inference
    
    # Initialize Grad-CAM
    grad_cam = GradCAM3D(model, target_layer)
    
    def gradcam_predictor(x):
        """Wrapper for sliding window inference."""
        cam = grad_cam.generate_cam(
            x,
            target_class=target_class,
            normalize=False,  # Don't normalize individual windows
        )
        # Add batch and channel dimensions for sliding window
        return cam.unsqueeze(0).unsqueeze(0)
    
    # Apply sliding window
    full_cam = sliding_window_inference(
        inputs=input_tensor,
        roi_size=window_size,
        sw_batch_size=1,
        predictor=gradcam_predictor,
        overlap=overlap,
        mode="gaussian",
    )
    
    # Remove batch and channel dimensions
    full_cam = full_cam.squeeze()
    
    # Normalize final result
    if full_cam.max() > 0:
        full_cam = full_cam / full_cam.max()
    
    return full_cam