#!/usr/bin/env python3
"""
Generate explainability maps for BraTS predictions.
"""

import argparse
import os
import sys
import yaml
import torch
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.transforms import get_inference_transforms
from models import create_unet3d
from xai import GradCAM3D, IntegratedGradients3D, XAIVisualizer
from xai import generate_gradcam_for_brats, generate_ig_for_brats
from xai import save_attribution_nifti, create_html_report


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate XAI for BraTS Predictions")
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save XAI results")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    
    # Model configuration
    parser.add_argument("--config", type=str,
                        help="Path to model configuration file")
    
    # XAI methods
    parser.add_argument("--methods", type=str, nargs="+", 
                        default=["gradcam", "integrated_gradients"],
                        choices=["gradcam", "integrated_gradients", "both"],
                        help="XAI methods to apply")
    
    # Grad-CAM parameters
    parser.add_argument("--target_layer", type=str, default="decoder.final_conv",
                        help="Target layer for Grad-CAM")
    parser.add_argument("--target_classes", type=int, nargs="+", default=[1, 2, 4],
                        help="Target classes for analysis (BraTS: 1=NCR, 2=ED, 4=ET)")
    
    # Integrated Gradients parameters
    parser.add_argument("--baseline_type", type=str, default="zero",
                        choices=["zero", "random", "blur", "mean"],
                        help="Baseline type for Integrated Gradients")
    parser.add_argument("--n_steps", type=int, default=50,
                        help="Number of integration steps")
    parser.add_argument("--use_noise_tunnel", action="store_true",
                        help="Use noise tunnel for robust attributions")
    
    # Output options
    parser.add_argument("--save_nifti", action="store_true",
                        help="Save attribution maps as NIfTI files")
    parser.add_argument("--save_visualizations", action="store_true", default=True,
                        help="Save PNG visualizations")
    parser.add_argument("--create_html_reports", action="store_true",
                        help="Create HTML reports")
    
    # Subjects selection
    parser.add_argument("--subjects", type=str, nargs="+",
                        help="Specific subjects to process (if not specified, processes all)")
    parser.add_argument("--max_subjects", type=int,
                        help="Maximum number of subjects to process")
    
    # Ground truth
    parser.add_argument("--ground_truth_dir", type=str,
                        help="Directory with ground truth segmentations")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_checkpoint(checkpoint_path: str, config: dict = None) -> torch.nn.Module:
    """Create model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get model config from checkpoint or provided config
    if config is None:
        if hasattr(checkpoint, 'config') and hasattr(checkpoint['config'], 'model'):
            model_config = checkpoint['config']['model']
        else:
            # Use default config
            model_config = {
                "input_channels": 4,
                "output_channels": 4,
                "features": [32, 64, 128, 256, 512],
                "deep_supervision": False,
            }
    else:
        model_config = config.get("model", {})
    
    # Create model
    model = create_unet3d(
        input_channels=model_config.get("input_channels", 4),
        output_channels=model_config.get("output_channels", 4),
        feature_maps=model_config.get("features", [32, 64, 128, 256, 512]),
        deep_supervision=model_config.get("deep_supervision", False),
        use_attention=model_config.get("use_attention", False),
        dropout_rate=model_config.get("dropout_rate", 0.1),
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model


def find_subject_files(input_dir: str, subjects: list = None) -> list:
    """Find subject directories in input directory."""
    input_path = Path(input_dir)
    all_subjects = []
    
    # Look for subject directories
    for subject_dir in input_path.iterdir():
        if subject_dir.is_dir() and not subject_dir.name.startswith("."):
            subject_id = subject_dir.name
            
            # Filter by specified subjects if provided
            if subjects and subject_id not in subjects:
                continue
            
            # Check if required modalities exist
            modalities = ["t1", "t1ce", "t2", "flair"]
            files_exist = all(
                (subject_dir / f"{subject_id}_{mod}.nii.gz").exists()
                for mod in modalities
            )
            
            if files_exist:
                all_subjects.append({
                    "subject_id": subject_id,
                    "directory": subject_dir,
                    "files": {
                        mod: subject_dir / f"{subject_id}_{mod}.nii.gz"
                        for mod in modalities
                    }
                })
    
    return all_subjects


def load_subject_data(subject_info: dict, transforms) -> torch.Tensor:
    """Load and preprocess subject data."""
    # Create data dict for transforms
    data_dict = {
        "image": [str(subject_info["files"][mod]) for mod in ["t1", "t1ce", "t2", "flair"]],
        "subject_id": subject_info["subject_id"],
    }
    
    # Apply transforms
    transformed = transforms(data_dict)
    
    return transformed["image"]


def load_ground_truth(subject_id: str, ground_truth_dir: str) -> np.ndarray:
    """Load ground truth segmentation if available."""
    if ground_truth_dir is None:
        return None
    
    # Try different possible paths
    possible_paths = [
        Path(ground_truth_dir) / f"{subject_id}_seg.nii.gz",
        Path(ground_truth_dir) / subject_id / f"{subject_id}_seg.nii.gz",
        Path(ground_truth_dir) / f"{subject_id}.nii.gz",
    ]
    
    for path in possible_paths:
        if path.exists():
            return nib.load(str(path)).get_fdata()
    
    return None


def generate_subject_attributions(
    subject_info: dict,
    model: torch.nn.Module,
    device: torch.device,
    transforms,
    methods: list,
    target_layer: str,
    target_classes: list,
    baseline_type: str,
    n_steps: int,
    use_noise_tunnel: bool,
) -> tuple:
    """Generate attribution maps for a subject."""
    # Load data
    image = load_subject_data(subject_info, transforms)
    image = image.to(device)
    
    attributions = {}
    
    # Grad-CAM
    if "gradcam" in methods or "both" in methods:
        print(f"  Generating Grad-CAM...")
        gradcam_results = generate_gradcam_for_brats(
            model=model,
            input_tensor=image,
            target_layer=target_layer,
            target_classes=target_classes,
            normalize=True,
        )
        
        for class_name, cam in gradcam_results.items():
            attributions[f"gradcam_{class_name}"] = cam.cpu().numpy()
    
    # Integrated Gradients
    if "integrated_gradients" in methods or "both" in methods:
        print(f"  Generating Integrated Gradients...")
        ig_results = generate_ig_for_brats(
            model=model,
            input_tensor=image,
            target_classes=target_classes,
            baseline_type=baseline_type,
            n_steps=n_steps,
            use_noise_tunnel=use_noise_tunnel,
        )
        
        for class_name, attr in ig_results.items():
            # Take mean across input channels for visualization
            if attr.dim() == 4:  # (C, D, H, W)
                attr_viz = torch.mean(torch.abs(attr), dim=0).cpu().numpy()
            else:
                attr_viz = torch.abs(attr).cpu().numpy()
            
            attributions[f"ig_{class_name}"] = attr_viz
    
    # Convert image to numpy for visualization
    if image.dim() == 4:  # (C, D, H, W)
        image_np = image.cpu().numpy()
    else:  # (1, C, D, H, W)
        image_np = image.squeeze(0).cpu().numpy()
    
    # Use first modality (T1) for visualization background
    image_for_viz = image_np[0]  # T1 modality
    
    return attributions, image_for_viz


def save_subject_results(
    subject_id: str,
    attributions: dict,
    image: np.ndarray,
    segmentation: np.ndarray,
    output_dir: Path,
    reference_image_path: str,
    save_nifti: bool,
    save_visualizations: bool,
    create_html_report_flag: bool,
) -> list:
    """Save all results for a subject."""
    subject_output_dir = output_dir / subject_id
    subject_output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    
    # Save NIfTI attribution maps
    if save_nifti:
        nifti_dir = subject_output_dir / "nifti"
        nifti_dir.mkdir(exist_ok=True)
        
        for attr_name, attr_data in attributions.items():
            nifti_path = nifti_dir / f"{attr_name}.nii.gz"
            save_attribution_nifti(
                attribution=attr_data,
                reference_image_path=reference_image_path,
                output_path=str(nifti_path),
            )
            created_files.append(str(nifti_path))
    
    # Save visualizations
    if save_visualizations:
        viz_dir = subject_output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Group attributions by method
        gradcam_attrs = {k.replace("gradcam_", ""): v for k, v in attributions.items() if k.startswith("gradcam_")}
        ig_attrs = {k.replace("ig_", ""): v for k, v in attributions.items() if k.startswith("ig_")}
        
        visualizer = XAIVisualizer()
        
        # Create visualizations for each method
        for method_name, attrs in [("gradcam", gradcam_attrs), ("ig", ig_attrs)]:
            if attrs:
                method_viz_dir = viz_dir / method_name
                overlays = visualizer.visualize_brats_attributions(
                    image=image,
                    attributions=attrs,
                    segmentation=segmentation,
                    save_dir=str(method_viz_dir),
                )
                created_files.extend([str(method_viz_dir / f"{k}_attribution_overlay.png") for k in attrs.keys()])
    
    # Create HTML report
    if create_html_report_flag:
        html_path = create_html_report(
            subject_id=subject_id,
            image=image,
            attributions=attributions,
            segmentation=segmentation,
            output_dir=str(subject_output_dir),
        )
        created_files.append(html_path)
    
    return created_files


def main():
    """Main function for XAI generation."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load configuration
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = create_model_from_checkpoint(args.model_path, config)
    model = model.to(device)
    model.eval()
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Get transforms
    transforms = get_inference_transforms()
    
    # Find subjects
    print(f"Searching for subjects in {args.input_dir}")
    subjects = find_subject_files(args.input_dir, args.subjects)
    
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    
    print(f"Found {len(subjects)} subjects to process")
    
    if len(subjects) == 0:
        print("No valid subjects found!")
        return
    
    # Process each subject
    all_created_files = []
    
    print("Generating explainability maps...")
    for i, subject_info in enumerate(tqdm(subjects, desc="Processing subjects")):
        subject_id = subject_info["subject_id"]
        
        try:
            print(f"\nProcessing {subject_id} ({i+1}/{len(subjects)})")
            
            # Generate attributions
            attributions, image = generate_subject_attributions(
                subject_info=subject_info,
                model=model,
                device=device,
                transforms=transforms,
                methods=args.methods,
                target_layer=args.target_layer,
                target_classes=args.target_classes,
                baseline_type=args.baseline_type,
                n_steps=args.n_steps,
                use_noise_tunnel=args.use_noise_tunnel,
            )
            
            # Load ground truth if available
            segmentation = load_ground_truth(subject_id, args.ground_truth_dir)
            
            # Save results
            created_files = save_subject_results(
                subject_id=subject_id,
                attributions=attributions,
                image=image,
                segmentation=segmentation,
                output_dir=output_dir,
                reference_image_path=str(subject_info["files"]["t1"]),
                save_nifti=args.save_nifti,
                save_visualizations=args.save_visualizations,
                create_html_report_flag=args.create_html_reports,
            )
            
            all_created_files.extend(created_files)
            print(f"✓ Processed {subject_id}")
            
        except Exception as e:
            print(f"✗ Error processing {subject_id}: {str(e)}")
            continue
    
    # Save summary
    summary_file = output_dir / "xai_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"XAI Generation Summary\n")
        f.write(f"=====================\n")
        f.write(f"Total subjects processed: {len([s for s in subjects])}\n")
        f.write(f"Methods used: {args.methods}\n")
        f.write(f"Target layer: {args.target_layer}\n")
        f.write(f"Target classes: {args.target_classes}\n")
        f.write(f"Created files: {len(all_created_files)}\n")
        f.write(f"\nCreated files:\n")
        for file_path in sorted(all_created_files):
            f.write(f"  {file_path}\n")
    
    print(f"\nXAI generation completed!")
    print(f"Processed {len([s for s in subjects])}/{len(subjects)} subjects")
    print(f"Results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_file}")
    
    if args.create_html_reports:
        print(f"\nHTML reports created for each subject in their respective directories")
    
    print("\nDone!")


if __name__ == "__main__":
    main()