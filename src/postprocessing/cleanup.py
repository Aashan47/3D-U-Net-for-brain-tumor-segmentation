import numpy as np
import cc3d
from scipy import ndimage
from scipy.ndimage import binary_fill_holes, binary_closing, binary_opening
from skimage.morphology import remove_small_objects, remove_small_holes
from typing import Dict, Tuple, Optional, Union
import warnings


def postprocess_prediction(
    prediction: np.ndarray,
    min_component_sizes: Optional[Dict[int, int]] = None,
    apply_morphological_ops: bool = True,
    enforce_constraints: bool = True,
    fill_holes_3d: bool = True,
) -> np.ndarray:
    """
    Complete postprocessing pipeline for BraTS predictions.
    
    Args:
        prediction: Predicted segmentation mask
        min_component_sizes: Minimum component sizes for each class
        apply_morphological_ops: Whether to apply morphological operations
        enforce_constraints: Whether to enforce anatomical constraints
        fill_holes_3d: Whether to fill holes in 3D
        
    Returns:
        Postprocessed segmentation mask
    """
    if min_component_sizes is None:
        min_component_sizes = {
            1: 500,   # NCR/NET
            2: 500,   # ED
            4: 200,   # ET (smaller threshold for enhancing tumor)
        }
    
    # Work on a copy
    result = prediction.copy()
    
    # Step 1: Remove small connected components
    result = remove_small_components(result, min_component_sizes)
    
    # Step 2: Apply morphological operations
    if apply_morphological_ops:
        result = apply_morphological_operations(result)
    
    # Step 3: Fill holes
    if fill_holes_3d:
        result = fill_holes(result)
    
    # Step 4: Enforce anatomical constraints
    if enforce_constraints:
        result = enforce_anatomical_constraints(result)
    
    # Step 5: Final cleanup - remove very small artifacts
    result = final_cleanup(result, min_size=50)
    
    return result


def remove_small_components(
    segmentation: np.ndarray,
    min_component_sizes: Dict[int, int],
) -> np.ndarray:
    """
    Remove small connected components for each class.
    
    Args:
        segmentation: Input segmentation mask
        min_component_sizes: Dictionary of {class_label: min_size}
        
    Returns:
        Cleaned segmentation mask
    """
    result = segmentation.copy()
    
    for class_label, min_size in min_component_sizes.items():
        # Get binary mask for this class
        class_mask = (segmentation == class_label)
        
        if not class_mask.any():
            continue
        
        # Find connected components
        labels, num_components = cc3d.connected_components(
            class_mask.astype(np.uint32),
            connectivity=26,  # 26-connectivity for 3D
            return_N=True
        )
        
        if num_components == 0:
            continue
        
        # Compute component sizes
        component_sizes = np.bincount(labels.flatten())
        
        # Keep only components larger than threshold
        keep_components = np.where(component_sizes >= min_size)[0]
        
        # Create mask of components to keep
        keep_mask = np.isin(labels, keep_components)
        
        # Update result
        result[class_mask & ~keep_mask] = 0
    
    return result


def fill_holes(
    segmentation: np.ndarray,
    max_hole_size: int = 1000,
) -> np.ndarray:
    """
    Fill holes in segmented regions.
    
    Args:
        segmentation: Input segmentation mask
        max_hole_size: Maximum size of holes to fill
        
    Returns:
        Segmentation with filled holes
    """
    result = segmentation.copy()
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    for label in unique_labels:
        # Get binary mask for this label
        binary_mask = (segmentation == label)
        
        if not binary_mask.any():
            continue
        
        # Fill holes using ndimage
        filled_mask = binary_fill_holes(binary_mask)
        
        # Only fill small holes to avoid filling large gaps
        hole_mask = filled_mask & ~binary_mask
        
        if hole_mask.any():
            # Find hole components
            hole_labels = cc3d.connected_components(hole_mask.astype(np.uint32))
            hole_sizes = np.bincount(hole_labels.flatten())
            
            # Only fill holes smaller than threshold
            small_holes = np.where(hole_sizes <= max_hole_size)[0]
            small_hole_mask = np.isin(hole_labels, small_holes)
            
            # Fill small holes
            result[hole_mask & small_hole_mask] = label
    
    return result


def apply_morphological_operations(
    segmentation: np.ndarray,
    iterations: int = 1,
) -> np.ndarray:
    """
    Apply morphological operations to clean up segmentation.
    
    Args:
        segmentation: Input segmentation mask
        iterations: Number of iterations for morphological operations
        
    Returns:
        Morphologically processed segmentation
    """
    result = segmentation.copy()
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    for label in unique_labels:
        # Get binary mask for this label
        binary_mask = (segmentation == label)
        
        if not binary_mask.any():
            continue
        
        # Apply closing to fill small gaps
        closed_mask = binary_closing(
            binary_mask,
            structure=np.ones((3, 3, 3)),
            iterations=iterations
        )
        
        # Apply opening to remove small protrusions
        opened_mask = binary_opening(
            closed_mask,
            structure=np.ones((3, 3, 3)),
            iterations=iterations
        )
        
        # Update result
        result[binary_mask] = 0  # Remove original
        result[opened_mask] = label  # Add processed
    
    return result


def enforce_anatomical_constraints(
    segmentation: np.ndarray,
) -> np.ndarray:
    """
    Enforce BraTS anatomical constraints: ET ⊆ TC ⊆ WT.
    
    Args:
        segmentation: Input segmentation mask
        
    Returns:
        Constrained segmentation mask
    """
    result = segmentation.copy()
    
    # BraTS label definitions:
    # 1: NCR/NET (non-enhancing tumor core)
    # 2: ED (edema)  
    # 4: ET (enhancing tumor)
    
    # Define region masks
    et_mask = (result == 4)  # Enhancing tumor
    ncr_mask = (result == 1)  # Non-enhancing tumor core
    ed_mask = (result == 2)   # Edema
    
    # TC (tumor core) = NCR + ET
    tc_mask = ncr_mask | et_mask
    
    # WT (whole tumor) = NCR + ET + ED
    wt_mask = tc_mask | ed_mask
    
    # Constraint 1: ET must be within TC
    # If ET exists outside TC, convert to NCR
    et_outside_tc = et_mask & ~tc_mask
    result[et_outside_tc] = 1
    
    # Constraint 2: TC must be within WT
    # If NCR exists outside WT, convert to ED
    tc_outside_wt = tc_mask & ~wt_mask
    result[tc_outside_wt] = 2
    
    # Additional constraint: Ensure ET is surrounded by tumor tissue
    # ET should not be directly adjacent to background
    if et_mask.any():
        # Dilate ET mask
        from scipy.ndimage import binary_dilation
        dilated_et = binary_dilation(et_mask, structure=np.ones((3, 3, 3)))
        
        # Find ET voxels adjacent to background
        background_mask = (result == 0)
        et_near_background = et_mask & binary_dilation(background_mask, structure=np.ones((3, 3, 3)))
        
        # Convert ET near background to NCR
        result[et_near_background] = 1
    
    return result


def final_cleanup(
    segmentation: np.ndarray,
    min_size: int = 50,
) -> np.ndarray:
    """
    Final cleanup to remove very small artifacts.
    
    Args:
        segmentation: Input segmentation mask
        min_size: Minimum component size to keep
        
    Returns:
        Final cleaned segmentation
    """
    result = segmentation.copy()
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    for label in unique_labels:
        class_mask = (segmentation == label)
        
        if not class_mask.any():
            continue
        
        # Remove very small components
        labels = cc3d.connected_components(class_mask.astype(np.uint32))
        component_sizes = np.bincount(labels.flatten())
        
        # Remove components smaller than threshold
        small_components = np.where(component_sizes < min_size)[0]
        small_component_mask = np.isin(labels, small_components)
        
        result[class_mask & small_component_mask] = 0
    
    return result


def restore_original_spacing(
    segmentation: np.ndarray,
    original_affine: np.ndarray,
    target_affine: np.ndarray,
    original_shape: Tuple[int, int, int],
    order: int = 0,
) -> np.ndarray:
    """
    Restore segmentation to original spacing and orientation.
    
    Args:
        segmentation: Segmentation in processed spacing
        original_affine: Original image affine matrix
        target_affine: Target affine matrix
        original_shape: Original image shape
        order: Interpolation order (0 for nearest neighbor)
        
    Returns:
        Segmentation in original spacing
    """
    from scipy.ndimage import affine_transform
    
    # Compute transformation matrix
    # This is a simplified version - in practice, you'd use more sophisticated resampling
    transform_matrix = np.linalg.inv(original_affine) @ target_affine
    
    # Apply inverse transformation
    restored = affine_transform(
        segmentation,
        transform_matrix[:3, :3],
        offset=transform_matrix[:3, 3],
        output_shape=original_shape,
        order=order,
        mode='constant',
        cval=0,
    )
    
    return restored.astype(segmentation.dtype)


def compute_volume_statistics(
    segmentation: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Dict[str, float]:
    """
    Compute volume statistics for different tumor regions.
    
    Args:
        segmentation: Segmentation mask
        spacing: Voxel spacing in mm
        
    Returns:
        Dictionary with volume statistics
    """
    voxel_volume = np.prod(spacing)  # Volume per voxel in mm³
    
    # BraTS region definitions
    regions = {
        "WT": [1, 2, 4],  # Whole tumor
        "TC": [1, 4],     # Tumor core
        "ET": [4],        # Enhancing tumor
        "NCR": [1],       # Non-enhancing tumor core
        "ED": [2],        # Edema
    }
    
    volumes = {}
    for region_name, labels in regions.items():
        mask = np.isin(segmentation, labels)
        volume_voxels = np.sum(mask)
        volume_mm3 = volume_voxels * voxel_volume
        volumes[f"{region_name}_volume_voxels"] = volume_voxels
        volumes[f"{region_name}_volume_mm3"] = volume_mm3
    
    return volumes


def postprocess_batch(
    predictions: np.ndarray,
    min_component_sizes: Optional[Dict[int, int]] = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Postprocess a batch of predictions.
    
    Args:
        predictions: Batch of predictions (B, D, H, W)
        min_component_sizes: Minimum component sizes
        n_jobs: Number of parallel jobs
        
    Returns:
        Postprocessed predictions
    """
    if n_jobs == 1:
        # Serial processing
        results = []
        for i in range(predictions.shape[0]):
            result = postprocess_prediction(
                predictions[i],
                min_component_sizes=min_component_sizes,
            )
            results.append(result)
        return np.stack(results)
    
    else:
        # Parallel processing
        from joblib import Parallel, delayed
        
        def process_single(pred):
            return postprocess_prediction(
                pred,
                min_component_sizes=min_component_sizes,
            )
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(process_single)(predictions[i])
            for i in range(predictions.shape[0])
        )
        
        return np.stack(results)


class BraTSPostprocessor:
    """Class-based postprocessor for BraTS predictions."""
    
    def __init__(
        self,
        min_component_sizes: Optional[Dict[int, int]] = None,
        apply_morphological_ops: bool = True,
        enforce_constraints: bool = True,
        fill_holes_3d: bool = True,
        verbose: bool = False,
    ):
        """
        Initialize BraTS postprocessor.
        
        Args:
            min_component_sizes: Minimum component sizes for each class
            apply_morphological_ops: Whether to apply morphological operations
            enforce_constraints: Whether to enforce anatomical constraints
            fill_holes_3d: Whether to fill holes in 3D
            verbose: Whether to print processing information
        """
        if min_component_sizes is None:
            min_component_sizes = {1: 500, 2: 500, 4: 200}
        
        self.min_component_sizes = min_component_sizes
        self.apply_morphological_ops = apply_morphological_ops
        self.enforce_constraints = enforce_constraints
        self.fill_holes_3d = fill_holes_3d
        self.verbose = verbose
    
    def __call__(self, prediction: np.ndarray) -> np.ndarray:
        """
        Process a single prediction.
        
        Args:
            prediction: Input prediction
            
        Returns:
            Postprocessed prediction
        """
        return postprocess_prediction(
            prediction,
            min_component_sizes=self.min_component_sizes,
            apply_morphological_ops=self.apply_morphological_ops,
            enforce_constraints=self.enforce_constraints,
            fill_holes_3d=self.fill_holes_3d,
        )
    
    def process_batch(
        self,
        predictions: np.ndarray,
        n_jobs: int = 1,
    ) -> np.ndarray:
        """
        Process a batch of predictions.
        
        Args:
            predictions: Batch of predictions
            n_jobs: Number of parallel jobs
            
        Returns:
            Postprocessed predictions
        """
        return postprocess_batch(
            predictions,
            min_component_sizes=self.min_component_sizes,
            n_jobs=n_jobs,
        )
    
    def get_statistics(self, prediction: np.ndarray) -> Dict[str, Union[float, int]]:
        """
        Get statistics about the prediction.
        
        Args:
            prediction: Input prediction
            
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        # Basic statistics
        unique_labels, counts = np.unique(prediction, return_counts=True)
        for label, count in zip(unique_labels, counts):
            stats[f"label_{label}_count"] = count
        
        # Volume statistics
        volume_stats = compute_volume_statistics(prediction)
        stats.update(volume_stats)
        
        # Component statistics
        for label in [1, 2, 4]:
            if label in unique_labels:
                class_mask = (prediction == label)
                labels = cc3d.connected_components(class_mask.astype(np.uint32))
                num_components = len(np.unique(labels)) - 1  # Exclude background
                stats[f"label_{label}_num_components"] = num_components
                
                if num_components > 0:
                    component_sizes = np.bincount(labels.flatten())[1:]  # Exclude background
                    stats[f"label_{label}_max_component_size"] = np.max(component_sizes)
                    stats[f"label_{label}_mean_component_size"] = np.mean(component_sizes)
        
        return stats