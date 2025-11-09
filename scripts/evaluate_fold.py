#!/usr/bin/env python3
"""
Evaluation script for a specific fold.
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data import BraTSDataModule
from models import create_unet3d
from losses import DiceFocalLoss, DiceBCELoss, FocalTverskyLoss
from engine import BraTSValidator
from metrics import BraTSMetrics, BraTSEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate BraTS Model on Fold")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preprocessed BraTS data")
    parser.add_argument("--fold", type=int, required=True,
                        help="Cross-validation fold to evaluate (0-4)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    
    # Model configuration
    parser.add_argument("--config", type=str,
                        help="Path to model configuration file")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Save prediction images")
    
    # Evaluation parameters
    parser.add_argument("--roi_size", type=int, nargs=3, default=[128, 128, 128],
                        help="ROI size for sliding window inference")
    parser.add_argument("--sw_batch_size", type=int, default=4,
                        help="Batch size for sliding window")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Overlap ratio for sliding window")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    
    # Data loading
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
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
        if 'config' in checkpoint and hasattr(checkpoint['config'], '__dict__'):
            model_config = checkpoint['config'].__dict__.get('model', {})
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


def create_loss_function(config: dict) -> nn.Module:
    """Create loss function from configuration."""
    if config is None:
        # Default loss
        return DiceFocalLoss(dice_weight=1.0, focal_weight=0.5)
    
    loss_config = config.get("loss", {})
    loss_name = loss_config.get("name", "DiceFocalLoss")
    
    if loss_name == "DiceFocalLoss":
        loss_fn = DiceFocalLoss(
            dice_weight=loss_config.get("dice_weight", 1.0),
            focal_weight=loss_config.get("focal_weight", 0.5),
            alpha=loss_config.get("alpha", 0.25),
            gamma=loss_config.get("gamma", 2.0),
            include_background=loss_config.get("include_background", False),
        )
    elif loss_name == "DiceBCELoss":
        class_weights = loss_config.get("class_weights")
        if class_weights:
            class_weights = torch.tensor(class_weights)
        
        loss_fn = DiceBCELoss(
            dice_weight=loss_config.get("dice_weight", 1.0),
            bce_weight=loss_config.get("bce_weight", 1.0),
            class_weights=class_weights,
            include_background=loss_config.get("include_background", False),
        )
    elif loss_name == "FocalTverskyLoss":
        loss_fn = FocalTverskyLoss(
            alpha=loss_config.get("alpha", 0.7),
            beta=loss_config.get("beta", 0.3),
            gamma=loss_config.get("gamma", 1.33),
            include_background=loss_config.get("include_background", False),
        )
    else:
        # Default fallback
        loss_fn = DiceFocalLoss(dice_weight=1.0, focal_weight=0.5)
    
    return loss_fn


def main():
    """Main evaluation function."""
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
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss function (for validation loss calculation)
    loss_function = create_loss_function(config)
    loss_function = loss_function.to(device)
    
    # Create data module
    data_config = config.get("data", {}) if config else {}
    data_module = BraTSDataModule(
        data_dir=args.data_path,
        batch_size=1,  # Evaluation with batch size 1
        num_workers=args.num_workers,
        cache_rate=data_config.get("cache_rate", 0.1),
        patch_size=data_config.get("patch_size", [128, 128, 128]),
        samples_per_epoch=data_config.get("samples_per_epoch", 1000),
        foreground_ratio=data_config.get("foreground_ratio", 0.7),
        n_folds=5,
    )
    
    # Get validation data loader for the specified fold
    train_loader, val_loader = data_module.get_fold_dataloaders(args.fold)
    
    print(f"Evaluating fold {args.fold}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Initialize metrics and validator
    metrics = BraTSMetrics(spacing=(1.0, 1.0, 1.0))
    validator = BraTSValidator(
        metrics=metrics,
        device=device,
        roi_size=tuple(args.roi_size),
        sw_batch_size=args.sw_batch_size,
        overlap=args.overlap,
    )
    
    # Run evaluation
    print("Starting evaluation...")
    
    predictions_dir = None
    if args.save_predictions:
        predictions_dir = output_dir / f"fold_{args.fold}_predictions"
        predictions_dir.mkdir(exist_ok=True)
    
    validation_results = validator.validate(
        model=model,
        data_loader=val_loader,
        loss_function=loss_function,
        save_predictions=args.save_predictions,
        output_dir=str(predictions_dir) if predictions_dir else None,
    )
    
    print("Evaluation completed!")
    
    # Print results
    print("\nValidation Results:")
    print("=" * 50)
    
    # Main metrics
    main_metrics = ["dice_mean", "WT_dice", "TC_dice", "ET_dice", "hd95_mean"]
    for metric in main_metrics:
        if metric in validation_results:
            print(f"{metric:15}: {validation_results[metric]:.4f}")
    
    # Loss
    if "loss" in validation_results:
        print(f"{'validation_loss':15}: {validation_results['loss']:.4f}")
    
    # Detailed metrics
    print("\nDetailed Metrics:")
    print("-" * 30)
    regions = ["WT", "TC", "ET"]
    for region in regions:
        print(f"\n{region} Region:")
        for metric_name, value in validation_results.items():
            if metric_name.startswith(f"{region}_") and not metric_name.endswith("_std"):
                metric_short = metric_name.replace(f"{region}_", "")
                print(f"  {metric_short:12}: {value:.4f}")
    
    # Save detailed results
    results_file = output_dir / f"fold_{args.fold}_validation_results.yaml"
    with open(results_file, 'w') as f:
        yaml.dump(validation_results, f, default_flow_style=False)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Save summary statistics
    summary_stats = {
        "fold": args.fold,
        "num_subjects": len(val_loader.dataset),
        "model_path": args.model_path,
        "main_metrics": {
            metric: validation_results.get(metric, 0.0)
            for metric in main_metrics
            if metric in validation_results
        }
    }
    
    summary_file = output_dir / f"fold_{args.fold}_summary.yaml"
    with open(summary_file, 'w') as f:
        yaml.dump(summary_stats, f, default_flow_style=False)
    
    print(f"Summary saved to: {summary_file}")
    
    if args.save_predictions:
        print(f"Predictions saved to: {predictions_dir}")
    
    # Print conclusion
    overall_dice = validation_results.get("dice_mean", 0.0)
    print(f"\nOverall Performance:")
    print(f"Mean Dice Score: {overall_dice:.4f}")
    
    if overall_dice >= 0.85:
        print("🎉 Excellent performance!")
    elif overall_dice >= 0.80:
        print("✅ Good performance!")
    elif overall_dice >= 0.75:
        print("⚠️  Acceptable performance")
    else:
        print("❌ Performance needs improvement")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()