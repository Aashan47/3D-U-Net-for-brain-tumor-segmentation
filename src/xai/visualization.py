import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from pathlib import Path
import nibabel as nib
import torch
from typing import Dict, List, Optional, Tuple, Union
import warnings


class XAIVisualizer:
    """Visualizer for XAI results in medical imaging."""
    
    def __init__(
        self,
        colormap: str = "hot",
        alpha_overlay: float = 0.6,
        figsize: Tuple[int, int] = (15, 10),
    ):
        """
        Initialize XAI visualizer.
        
        Args:
            colormap: Colormap for heatmaps
            alpha_overlay: Alpha for overlay transparency
            figsize: Figure size for plots
        """
        self.colormap = colormap
        self.alpha_overlay = alpha_overlay
        self.figsize = figsize
        
        # BraTS color scheme
        self.brats_colors = {
            0: [0, 0, 0],      # Background - black
            1: [255, 0, 0],    # NCR - red
            2: [0, 255, 0],    # ED - green  
            4: [0, 0, 255],    # ET - blue
        }
    
    def create_overlay_slices(
        self,
        image: np.ndarray,
        attribution: np.ndarray,
        slice_indices: Optional[List[int]] = None,
        planes: List[str] = ["axial", "coronal", "sagittal"],
        normalize_attribution: bool = True,
        threshold_percentile: float = 90.0,
        save_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Create overlay slices for attribution visualization.
        
        Args:
            image: Original image (D, H, W) or (C, D, H, W)
            attribution: Attribution map (D, H, W) or (C, D, H, W)
            slice_indices: Specific slice indices (if None, uses middle slices)
            planes: Anatomical planes to visualize
            normalize_attribution: Whether to normalize attributions
            threshold_percentile: Percentile threshold for attribution display
            save_path: Path to save visualizations
            
        Returns:
            Dictionary with overlay images for each plane
        """
        # Handle multi-channel inputs
        if image.ndim == 4:
            # Use first channel or average
            image = image[0] if image.shape[0] <= 4 else np.mean(image, axis=0)
        
        if attribution.ndim == 4:
            # Use mean attribution across channels
            attribution = np.mean(np.abs(attribution), axis=0)
        
        # Normalize image to [0, 1]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Normalize attribution
        if normalize_attribution:
            attribution = np.abs(attribution)
            attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
        
        # Apply threshold
        threshold = np.percentile(attribution, threshold_percentile)
        attribution_thresholded = np.where(attribution >= threshold, attribution, 0)
        
        # Get slice indices if not provided
        if slice_indices is None:
            slice_indices = {
                "axial": image.shape[0] // 2,
                "coronal": image.shape[1] // 2,
                "sagittal": image.shape[2] // 2,
            }
        elif isinstance(slice_indices, list):
            # Convert list to dict using middle slices as defaults
            default_indices = [image.shape[i] // 2 for i in range(3)]
            slice_dict = {}
            for i, plane in enumerate(["axial", "coronal", "sagittal"]):
                if i < len(slice_indices):
                    slice_dict[plane] = slice_indices[i]
                else:
                    slice_dict[plane] = default_indices[i]
            slice_indices = slice_dict
        
        overlays = {}
        
        # Create overlays for each plane
        if "axial" in planes:
            idx = slice_indices["axial"]
            img_slice = image[idx]
            attr_slice = attribution_thresholded[idx]
            overlays["axial"] = self._create_single_overlay(img_slice, attr_slice)
        
        if "coronal" in planes:
            idx = slice_indices["coronal"]
            img_slice = image[:, idx, :]
            attr_slice = attribution_thresholded[:, idx, :]
            overlays["coronal"] = self._create_single_overlay(img_slice, attr_slice)
        
        if "sagittal" in planes:
            idx = slice_indices["sagittal"]
            img_slice = image[:, :, idx]
            attr_slice = attribution_thresholded[:, :, idx]
            overlays["sagittal"] = self._create_single_overlay(img_slice, attr_slice)
        
        # Create visualization plot
        if len(planes) > 0:
            self._plot_overlays(overlays, save_path)
        
        return overlays
    
    def _create_single_overlay(
        self,
        image_slice: np.ndarray,
        attribution_slice: np.ndarray,
    ) -> np.ndarray:
        """Create overlay for a single slice."""
        # Create RGB image
        img_rgb = np.stack([image_slice] * 3, axis=-1)
        
        # Create attribution overlay
        cmap = cm.get_cmap(self.colormap)
        attr_colored = cmap(attribution_slice)[:, :, :3]  # Remove alpha channel
        
        # Combine image and attribution
        mask = attribution_slice > 0
        overlay = img_rgb.copy()
        overlay[mask] = (
            (1 - self.alpha_overlay) * img_rgb[mask] + 
            self.alpha_overlay * attr_colored[mask]
        )
        
        return overlay
    
    def _plot_overlays(
        self,
        overlays: Dict[str, np.ndarray],
        save_path: Optional[str] = None,
    ):
        """Plot overlay images."""
        num_planes = len(overlays)
        fig, axes = plt.subplots(1, num_planes, figsize=self.figsize)
        
        if num_planes == 1:
            axes = [axes]
        
        for idx, (plane, overlay) in enumerate(overlays.items()):
            axes[idx].imshow(overlay, aspect='auto')
            axes[idx].set_title(f"{plane.capitalize()} View")
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def visualize_brats_attributions(
        self,
        image: np.ndarray,
        attributions: Dict[str, np.ndarray],
        segmentation: Optional[np.ndarray] = None,
        slice_indices: Optional[Dict[str, int]] = None,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Visualize attributions for BraTS tumor classes.
        
        Args:
            image: Original image
            attributions: Dictionary of {class_name: attribution}
            segmentation: Ground truth segmentation (optional)
            slice_indices: Slice indices for each plane
            save_dir: Directory to save visualizations
            
        Returns:
            Dictionary with overlays for each class and plane
        """
        all_overlays = {}
        
        # Default slice indices
        if slice_indices is None:
            slice_indices = {
                "axial": image.shape[0] // 2,
                "coronal": image.shape[1] // 2,
                "sagittal": image.shape[2] // 2,
            }
        
        # Process each class
        for class_name, attribution in attributions.items():
            print(f"Processing {class_name} attributions...")
            
            # Create save path for this class
            class_save_path = None
            if save_dir:
                save_dir_path = Path(save_dir)
                save_dir_path.mkdir(parents=True, exist_ok=True)
                class_save_path = save_dir_path / f"{class_name}_attribution_overlay.png"
            
            # Create overlays
            overlays = self.create_overlay_slices(
                image=image,
                attribution=attribution,
                slice_indices=[slice_indices["axial"], slice_indices["coronal"], slice_indices["sagittal"]],
                save_path=str(class_save_path) if class_save_path else None,
            )
            
            all_overlays[class_name] = overlays
        
        # Create combined visualization if segmentation is provided
        if segmentation is not None:
            self._visualize_with_segmentation(
                image, attributions, segmentation, slice_indices, save_dir
            )
        
        return all_overlays
    
    def _visualize_with_segmentation(
        self,
        image: np.ndarray,
        attributions: Dict[str, np.ndarray],
        segmentation: np.ndarray,
        slice_indices: Dict[str, int],
        save_dir: Optional[str] = None,
    ):
        """Visualize attributions alongside segmentation."""
        planes = ["axial", "coronal", "sagittal"]
        class_names = list(attributions.keys())
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(
            len(class_names) + 1, len(planes), 
            figsize=(5 * len(planes), 4 * (len(class_names) + 1))
        )
        
        if len(planes) == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot segmentation in first row
        for j, plane in enumerate(planes):
            if plane == "axial":
                seg_slice = segmentation[slice_indices[plane]]
                img_slice = image[slice_indices[plane]]
            elif plane == "coronal":
                seg_slice = segmentation[:, slice_indices[plane], :]
                img_slice = image[:, slice_indices[plane], :]
            else:  # sagittal
                seg_slice = segmentation[:, :, slice_indices[plane]]
                img_slice = image[:, :, slice_indices[plane]]
            
            # Overlay segmentation on image
            seg_overlay = self._create_segmentation_overlay(img_slice, seg_slice)
            
            axes[0, j].imshow(seg_overlay, aspect='auto')
            axes[0, j].set_title(f"Segmentation - {plane.capitalize()}")
            axes[0, j].axis('off')
        
        # Plot attributions for each class
        for i, (class_name, attribution) in enumerate(attributions.items()):
            for j, plane in enumerate(planes):
                if plane == "axial":
                    attr_slice = attribution[slice_indices[plane]]
                    img_slice = image[slice_indices[plane]]
                elif plane == "coronal":
                    attr_slice = attribution[:, slice_indices[plane], :]
                    img_slice = image[:, slice_indices[plane], :]
                else:  # sagittal
                    attr_slice = attribution[:, :, slice_indices[plane]]
                    img_slice = image[:, :, slice_indices[plane]]
                
                # Create overlay
                overlay = self._create_single_overlay(img_slice, np.abs(attr_slice))
                
                axes[i + 1, j].imshow(overlay, aspect='auto')
                axes[i + 1, j].set_title(f"{class_name} - {plane.capitalize()}")
                axes[i + 1, j].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / "combined_attribution_analysis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined visualization to {save_path}")
        
        plt.show()
    
    def _create_segmentation_overlay(
        self,
        image_slice: np.ndarray,
        seg_slice: np.ndarray,
    ) -> np.ndarray:
        """Create segmentation overlay on image."""
        # Normalize image
        img_norm = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min() + 1e-8)
        
        # Create RGB image
        img_rgb = np.stack([img_norm] * 3, axis=-1)
        
        # Add segmentation colors
        for label, color in self.brats_colors.items():
            if label == 0:  # Skip background
                continue
            
            mask = seg_slice == label
            if np.any(mask):
                color_norm = np.array(color) / 255.0
                img_rgb[mask] = (
                    (1 - self.alpha_overlay) * img_rgb[mask] + 
                    self.alpha_overlay * color_norm
                )
        
        return img_rgb


def save_attribution_nifti(
    attribution: Union[np.ndarray, torch.Tensor],
    reference_image_path: str,
    output_path: str,
    affine: Optional[np.ndarray] = None,
) -> None:
    """
    Save attribution map as NIfTI file.
    
    Args:
        attribution: Attribution map
        reference_image_path: Path to reference image for header info
        output_path: Output file path
        affine: Custom affine matrix (if None, uses reference)
    """
    # Convert to numpy if tensor
    if isinstance(attribution, torch.Tensor):
        attribution = attribution.cpu().numpy()
    
    # Load reference image for header
    try:
        ref_img = nib.load(reference_image_path)
        if affine is None:
            affine = ref_img.affine
        header = ref_img.header.copy()
    except:
        warnings.warn(f"Could not load reference image {reference_image_path}")
        if affine is None:
            affine = np.eye(4)
        header = None
    
    # Create NIfTI image
    attr_img = nib.Nifti1Image(
        attribution.astype(np.float32),
        affine=affine,
        header=header,
    )
    
    # Save
    nib.save(attr_img, output_path)
    print(f"Saved attribution to {output_path}")


def create_html_report(
    subject_id: str,
    image: np.ndarray,
    attributions: Dict[str, np.ndarray],
    segmentation: Optional[np.ndarray] = None,
    output_dir: str = "./xai_reports",
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
) -> str:
    """
    Create HTML report for XAI analysis.
    
    Args:
        subject_id: Subject identifier
        image: Original image
        attributions: Attribution maps
        segmentation: Segmentation mask
        output_dir: Output directory
        metrics: Attribution metrics
        
    Returns:
        Path to created HTML file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create visualizations
    visualizer = XAIVisualizer()
    
    # Save slice images
    slice_dir = output_path / f"{subject_id}_slices"
    slice_dir.mkdir(exist_ok=True)
    
    overlays = visualizer.visualize_brats_attributions(
        image=image,
        attributions=attributions,
        segmentation=segmentation,
        save_dir=str(slice_dir),
    )
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>XAI Report - {subject_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; }}
            .section {{ margin: 20px 0; }}
            .image-grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .image-item {{ text-align: center; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>XAI Analysis Report</h1>
            <h2>Subject: {subject_id}</h2>
        </div>
        
        <div class="section">
            <h3>Attribution Visualizations</h3>
            <div class="image-grid">
    """
    
    # Add images for each class
    for class_name in attributions.keys():
        img_path = f"{subject_id}_slices/{class_name}_attribution_overlay.png"
        html_content += f"""
            <div class="image-item">
                <img src="{img_path}" alt="{class_name} Attribution" style="max-width: 400px;">
                <p><strong>{class_name}</strong></p>
            </div>
        """
    
    html_content += """
            </div>
        </div>
    """
    
    # Add metrics if provided
    if metrics:
        html_content += """
        <div class="section">
            <h3>Attribution Metrics</h3>
            <table class="metrics-table">
                <tr>
                    <th>Class</th>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
        """
        
        for class_name, class_metrics in metrics.items():
            for metric_name, value in class_metrics.items():
                html_content += f"""
                <tr>
                    <td>{class_name}</td>
                    <td>{metric_name}</td>
                    <td>{value:.4f}</td>
                </tr>
                """
        
        html_content += """
            </table>
        </div>
        """
    
    # Close HTML
    html_content += """
        <div class="section">
            <h3>Clinical Interpretation</h3>
            <p>This analysis shows the regions that the AI model focuses on when making predictions for each tumor class:</p>
            <ul>
                <li><strong>NCR:</strong> Non-enhancing tumor core regions</li>
                <li><strong>ED:</strong> Edematous regions surrounding the tumor</li>
                <li><strong>ET:</strong> Enhancing tumor regions with active growth</li>
            </ul>
            <p><em>High attention areas (bright regions) indicate where the model is focusing its decision-making process.</em></p>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    html_path = output_path / f"{subject_id}_xai_report.html"
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Created HTML report: {html_path}")
    return str(html_path)


def batch_create_overlays(
    images: List[np.ndarray],
    attributions: List[Dict[str, np.ndarray]],
    subject_ids: List[str],
    output_dir: str,
    n_jobs: int = 1,
) -> List[str]:
    """
    Create overlay visualizations for multiple subjects.
    
    Args:
        images: List of images
        attributions: List of attribution dictionaries
        subject_ids: List of subject IDs
        output_dir: Output directory
        n_jobs: Number of parallel jobs
        
    Returns:
        List of created file paths
    """
    if n_jobs == 1:
        # Sequential processing
        created_files = []
        for i, (image, attr, subject_id) in enumerate(zip(images, attributions, subject_ids)):
            print(f"Processing {subject_id} ({i+1}/{len(images)})...")
            
            visualizer = XAIVisualizer()
            subject_dir = Path(output_dir) / subject_id
            
            overlays = visualizer.visualize_brats_attributions(
                image=image,
                attributions=attr,
                save_dir=str(subject_dir),
            )
            
            created_files.append(str(subject_dir))
        
        return created_files
    
    else:
        # Parallel processing
        from joblib import Parallel, delayed
        
        def process_single(image, attr, subject_id):
            visualizer = XAIVisualizer()
            subject_dir = Path(output_dir) / subject_id
            
            visualizer.visualize_brats_attributions(
                image=image,
                attributions=attr,
                save_dir=str(subject_dir),
            )
            
            return str(subject_dir)
        
        created_files = Parallel(n_jobs=n_jobs)(
            delayed(process_single)(images[i], attributions[i], subject_ids[i])
            for i in range(len(images))
        )
        
        return created_files