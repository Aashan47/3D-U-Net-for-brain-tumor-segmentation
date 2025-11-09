#!/usr/bin/env python3
"""
Training script for BraTS 3D segmentation model.
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
from engine import BraTSTrainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train BraTS 3D Segmentation Model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to preprocessed BraTS data")
    parser.add_argument("--fold", type=int, default=0,
                        help="Cross-validation fold (0-4)")
    
    # Model arguments
    parser.add_argument("--config", type=str, 
                        help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for checkpoints and logs")
    
    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=300,
                        help="Maximum number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use")
    
    # Resume training
    parser.add_argument("--resume", type=str,
                        help="Path to checkpoint to resume training")
    
    # Experiment tracking
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="brats-3d-segmentation",
                        help="W&B project name")
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict) -> nn.Module:
    """Create model from configuration."""
    model_config = config.get("model", {})
    
    model = create_unet3d(
        input_channels=model_config.get("input_channels", 4),
        output_channels=model_config.get("output_channels", 4),
        feature_maps=model_config.get("features", [32, 64, 128, 256, 512]),
        deep_supervision=model_config.get("deep_supervision", False),
        use_attention=model_config.get("use_attention", False),
        dropout_rate=model_config.get("dropout_rate", 0.1),
    )
    
    return model


def create_loss_function(config: dict) -> nn.Module:
    """Create loss function from configuration."""
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
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    return loss_fn


def create_training_config(config: dict, args) -> TrainingConfig:
    """Create training configuration."""
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    loss_config = config.get("loss", {})
    optimization_config = config.get("optimization", {})
    validation_config = config.get("validation", {})
    logging_config = config.get("logging", {})
    
    # Override with command line arguments
    training_config = TrainingConfig(
        # Model parameters
        model_name=model_config.get("name", "UNet3D"),
        input_channels=model_config.get("input_channels", 4),
        output_channels=model_config.get("output_channels", 4),
        feature_maps=model_config.get("features", [32, 64, 128, 256, 512]),
        deep_supervision=model_config.get("deep_supervision", False),
        
        # Training parameters
        max_epochs=args.max_epochs or training_config.get("max_epochs", 300),
        batch_size=args.batch_size or training_config.get("batch_size", 2),
        learning_rate=args.learning_rate or training_config.get("learning_rate", 1e-4),
        weight_decay=training_config.get("weight_decay", 1e-5),
        gradient_clip_norm=training_config.get("gradient_clip_norm", 1.0),
        
        # Loss parameters
        loss_name=loss_config.get("name", "DiceFocalLoss"),
        dice_weight=loss_config.get("dice_weight", 1.0),
        focal_weight=loss_config.get("focal_weight", 0.5),
        class_weights=loss_config.get("class_weights"),
        
        # Optimization
        optimizer=optimization_config.get("optimizer", "AdamW"),
        scheduler=optimization_config.get("scheduler", "CosineAnnealingWarmRestarts"),
        warmup_epochs=optimization_config.get("warmup_epochs", 10),
        T_0=optimization_config.get("T_0", 50),
        T_mult=optimization_config.get("T_mult", 2),
        eta_min=optimization_config.get("eta_min", 1e-6),
        
        # Validation
        validation_interval=validation_config.get("interval", 5),
        save_top_k=validation_config.get("save_top_k", 3),
        patience=validation_config.get("patience", 50),
        monitor_metric=validation_config.get("monitor_metric", "dice_mean"),
        monitor_mode=validation_config.get("monitor_mode", "max"),
        
        # Mixed precision
        use_amp=training_config.get("use_amp", True),
        
        # Logging
        use_wandb=args.use_wandb or logging_config.get("use_wandb", False),
        log_images=logging_config.get("log_images", True),
        image_log_interval=logging_config.get("image_log_interval", 20),
        
        # Paths
        output_dir=args.output_dir,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        log_dir=os.path.join(args.output_dir, "logs"),
        
        # Device
        device=args.device,
    )
    
    return training_config


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Use default configuration
        config = {
            "data": {
                "cache_rate": 0.1,
                "patch_size": [128, 128, 128],
                "samples_per_epoch": 1000,
                "foreground_ratio": 0.7,
            },
            "model": {
                "input_channels": 4,
                "output_channels": 4,
                "features": [32, 64, 128, 256, 512],
                "deep_supervision": False,
            },
            "loss": {
                "name": "DiceFocalLoss",
                "dice_weight": 1.0,
                "focal_weight": 0.5,
            },
            "training": {
                "max_epochs": 300,
                "batch_size": 2,
                "learning_rate": 1e-4,
                "use_amp": True,
            },
        }
    
    # Create training configuration
    training_config = create_training_config(config, args)
    
    # Set up device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create data module
    data_config = config.get("data", {})
    data_module = BraTSDataModule(
        data_dir=args.data_path,
        batch_size=training_config.batch_size,
        num_workers=args.num_workers,
        cache_rate=data_config.get("cache_rate", 0.1),
        patch_size=data_config.get("patch_size", [128, 128, 128]),
        samples_per_epoch=data_config.get("samples_per_epoch", 1000),
        foreground_ratio=data_config.get("foreground_ratio", 0.7),
        n_folds=5,
    )
    
    # Print dataset info
    data_module.print_dataset_info()
    
    # Get data loaders for the specified fold
    train_loader, val_loader = data_module.get_fold_dataloaders(args.fold)
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    model = create_model(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create loss function
    loss_function = create_loss_function(config)
    print(f"Using loss function: {loss_function.__class__.__name__}")
    
    # Initialize trainer
    trainer = BraTSTrainer(
        config=training_config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        fold=args.fold,
    )
    
    # Resume training if checkpoint provided
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.resume_training(args.resume)
    else:
        # Start training
        print("Starting training...")
        trainer.train()
    
    print("Training completed!")


if __name__ == "__main__":
    main()