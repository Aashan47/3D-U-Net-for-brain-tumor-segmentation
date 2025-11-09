#!/usr/bin/env python3
"""
Inference script for BraTS 3D segmentation model.
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

from data import BraTSDataset
from data.transforms import get_inference_transforms
from models import create_unet3d
from engine import BraTSInference
from postprocessing import postprocess_prediction
from metrics import BraTSEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BraTS 3D Segmentation Inference")
    
    # Input/Output
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save predictions")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    
    # Model configuration
    parser.add_argument("--config", type=str,
                        help="Path to model configuration file")
    
    # Inference parameters
    parser.add_argument("--roi_size", type=int, nargs=3, default=[128, 128, 128],
                        help="ROI size for sliding window inference")
    parser.add_argument("--sw_batch_size", type=int, default=4,
                        help="Batch size for sliding window")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Overlap ratio for sliding window")
    parser.add_argument("--mode", type=str, default="gaussian", choices=["gaussian", "constant"],
                        help="Blending mode for sliding window")
    
    # Postprocessing
    parser.add_argument("--apply_postprocessing", action="store_true",
                        help="Apply postprocessing to predictions")
    parser.add_argument("--min_component_sizes", type=int, nargs=3, default=[500, 500, 200],
                        help="Minimum component sizes for NCR, ED, ET")
    
    # Evaluation
    parser.add_argument("--ground_truth_dir", type=str,
                        help="Directory with ground truth for evaluation")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate predictions against ground truth")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    
    # Output format
    parser.add_argument("--save_probabilities", action="store_true",
                        help="Save probability maps in addition to segmentations")
    parser.add_argument("--output_format", type=str, default="nifti", choices=["nifti", "numpy"],
                        help="Output format for predictions")
    
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


def find_subject_files(input_dir: str) -> list:
    """Find all subject directories in input directory."""
    input_path = Path(input_dir)
    subjects = []
    
    # Look for subject directories
    for subject_dir in input_path.iterdir():
        if subject_dir.is_dir() and not subject_dir.name.startswith("."):
            # Check if required modalities exist
            modalities = ["t1", "t1ce", "t2", "flair"]
            subject_id = subject_dir.name
            
            files_exist = all(
                (subject_dir / f"{subject_id}_{mod}.nii.gz").exists()
                for mod in modalities
            )
            
            if files_exist:
                subjects.append({
                    "subject_id": subject_id,
                    "directory": subject_dir,
                    "files": {
                        mod: subject_dir / f"{subject_id}_{mod}.nii.gz"
                        for mod in modalities
                    }
                })
    
    return subjects


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


def save_prediction(
    prediction: np.ndarray,
    reference_path: str,
    output_path: str,
    output_format: str = "nifti",
):
    """Save prediction to file."""
    if output_format == "nifti":
        # Load reference image for header info
        ref_img = nib.load(reference_path)
        
        # Create prediction image
        pred_img = nib.Nifti1Image(
            prediction.astype(np.uint8),
            affine=ref_img.affine,
            header=ref_img.header,
        )
        
        # Save
        nib.save(pred_img, output_path)
        
    elif output_format == "numpy":
        np.save(output_path, prediction)
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def run_inference_single_subject(
    subject_info: dict,
    inference_engine: BraTSInference,
    transforms,
    output_dir: str,
    apply_postprocessing: bool = True,
    min_component_sizes: dict = None,
    save_probabilities: bool = False,
    output_format: str = "nifti",
) -> str:
    """Run inference on a single subject."""
    subject_id = subject_info["subject_id"]
    
    # Load and preprocess data
    image = load_subject_data(subject_info, transforms)
    
    # Run inference
    with torch.no_grad():
        prediction = inference_engine.predict(
            image,
            apply_postprocessing=False,  # We'll do custom postprocessing
        )
    
    # Convert to numpy
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.cpu().numpy()
    
    # Remove batch dimension if present
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        prediction = prediction[0]
    
    # Convert to class labels if multi-class output
    if prediction.ndim == 4:  # (C, D, H, W)
        prediction = np.argmax(prediction, axis=0)
    elif prediction.ndim == 3:  # Already class labels
        pass
    else:
        raise ValueError(f"Unexpected prediction shape: {prediction.shape}")
    
    # Apply postprocessing
    if apply_postprocessing:
        if min_component_sizes is None:
            min_component_sizes = {1: 500, 2: 500, 4: 200}
        
        prediction = postprocess_prediction(
            prediction,
            min_component_sizes=min_component_sizes,
        )
    
    # Save prediction
    output_path = Path(output_dir) / subject_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    if output_format == "nifti":
        pred_file = output_path / f"{subject_id}_prediction.nii.gz"
        reference_file = str(subject_info["files"]["t1"])  # Use T1 as reference
    else:
        pred_file = output_path / f"{subject_id}_prediction.npy"
        reference_file = None
    
    save_prediction(
        prediction,
        reference_file,
        str(pred_file),
        output_format,
    )
    
    return str(pred_file)


def main():
    """Main inference function."""
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
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create inference engine
    inference_engine = BraTSInference(
        model=model,
        device=device,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
        mode=args.mode,
    )
    
    # Get transforms
    transforms = get_inference_transforms()
    
    # Find subjects
    print(f"Searching for subjects in {args.input_dir}")
    subjects = find_subject_files(args.input_dir)
    print(f"Found {len(subjects)} subjects")
    
    if len(subjects) == 0:
        print("No valid subjects found!")
        return
    
    # Prepare min component sizes
    min_component_sizes = {
        1: args.min_component_sizes[0],  # NCR
        2: args.min_component_sizes[1],  # ED
        4: args.min_component_sizes[2],  # ET
    }
    
    # Run inference on all subjects
    prediction_files = []
    
    print("Starting inference...")
    for subject_info in tqdm(subjects, desc="Processing subjects"):
        try:
            pred_file = run_inference_single_subject(
                subject_info=subject_info,
                inference_engine=inference_engine,
                transforms=transforms,
                output_dir=str(output_dir),
                apply_postprocessing=args.apply_postprocessing,
                min_component_sizes=min_component_sizes,
                save_probabilities=args.save_probabilities,
                output_format=args.output_format,
            )
            
            prediction_files.append((subject_info["subject_id"], pred_file))
            print(f"✓ Processed {subject_info['subject_id']}")
            
        except Exception as e:
            print(f"✗ Error processing {subject_info['subject_id']}: {str(e)}")
            continue
    
    print(f"Inference completed! Processed {len(prediction_files)}/{len(subjects)} subjects")
    
    # Save prediction list
    pred_list_file = output_dir / "prediction_list.txt"
    with open(pred_list_file, 'w') as f:
        for subject_id, pred_file in prediction_files:
            f.write(f"{subject_id}: {pred_file}\n")
    
    print(f"Prediction list saved to {pred_list_file}")
    
    # Run evaluation if ground truth is provided
    if args.evaluate and args.ground_truth_dir:
        print("Running evaluation...")
        
        # Create evaluator
        evaluator = BraTSEvaluator()
        
        # Evaluate each prediction
        for subject_id, pred_file in tqdm(prediction_files, desc="Evaluating"):
            # Find ground truth file
            gt_file = Path(args.ground_truth_dir) / f"{subject_id}_seg.nii.gz"
            if not gt_file.exists():
                gt_file = Path(args.ground_truth_dir) / subject_id / f"{subject_id}_seg.nii.gz"
            
            if gt_file.exists():
                try:
                    # Load prediction and ground truth
                    if args.output_format == "nifti":
                        pred_img = nib.load(pred_file).get_fdata()
                    else:
                        pred_img = np.load(pred_file)
                    
                    gt_img = nib.load(str(gt_file)).get_fdata()
                    
                    # Evaluate
                    evaluator.evaluate_subject(pred_img, gt_img, subject_id)
                    
                except Exception as e:
                    print(f"Error evaluating {subject_id}: {str(e)}")
            else:
                print(f"Ground truth not found for {subject_id}")
        
        # Save evaluation results
        eval_output_file = output_dir / "evaluation_results"
        evaluator.save_results(str(eval_output_file))
        
        # Print summary statistics
        summary = evaluator.get_summary_statistics()
        print("\nEvaluation Summary:")
        for region, stats in summary.items():
            print(f"{region}:")
            for metric, value in stats.items():
                if "mean" in metric:
                    print(f"  {metric}: {value:.4f}")
    
    print("All done!")


if __name__ == "__main__":
    main()