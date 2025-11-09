import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import wandb

from ..metrics.brats_metrics import BraTSMetrics
from .validator import BraTSValidator


@dataclass
class TrainingConfig:
    """Configuration for BraTS training."""
    
    # Model parameters
    model_name: str = "UNet3D"
    input_channels: int = 4
    output_channels: int = 4
    feature_maps: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    deep_supervision: bool = False
    
    # Training parameters
    max_epochs: int = 300
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    
    # Loss parameters
    loss_name: str = "DiceFocalLoss"
    dice_weight: float = 1.0
    focal_weight: float = 0.5
    class_weights: Optional[List[float]] = field(default_factory=lambda: [0.1, 0.3, 0.3, 0.3])
    
    # Optimization
    optimizer: str = "AdamW"
    scheduler: str = "CosineAnnealingWarmRestarts"
    warmup_epochs: int = 10
    T_0: int = 50
    T_mult: int = 2
    eta_min: float = 1e-6
    
    # Validation
    validation_interval: int = 5
    save_top_k: int = 3
    patience: int = 50
    monitor_metric: str = "dice_mean"
    monitor_mode: str = "max"
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    use_wandb: bool = False
    log_images: bool = True
    image_log_interval: int = 20
    
    # Paths
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./outputs/checkpoints"
    log_dir: str = "./outputs/logs"
    
    # Device
    device: str = "auto"
    
    # Reproducibility
    seed: int = 42


class BraTSTrainer:
    """Trainer for BraTS 3D segmentation models."""
    
    def __init__(
        self,
        config: TrainingConfig,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_function: nn.Module,
        fold: Optional[int] = None,
    ):
        """
        Initialize BraTS trainer.
        
        Args:
            config: Training configuration
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_function: Loss function
            fold: Cross-validation fold number
        """
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_function = loss_function
        self.fold = fold
        
        # Set up device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set up logging
        self._setup_logging()
        
        # Set up directories
        self._setup_directories()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Set up mixed precision
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Initialize metrics
        self.metrics = BraTSMetrics(spacing=(1.0, 1.0, 1.0))
        self.validator = BraTSValidator(self.metrics, self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('-inf') if config.monitor_mode == "max" else float('inf')
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_dice_wt": [],
            "val_dice_tc": [],
            "val_dice_et": [],
            "val_dice_mean": [],
            "val_hausdorff_wt": [],
            "val_hausdorff_tc": [],
            "val_hausdorff_et": [],
        }
        
        # Set up logging
        self._setup_tensorboard()
        if config.use_wandb:
            self._setup_wandb()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_format)
        self.logger = logging.getLogger(__name__)
    
    def _setup_directories(self):
        """Create necessary directories."""
        for directory in [self.config.output_dir, self.config.checkpoint_dir, self.config.log_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
        if self.fold is not None:
            self.fold_dir = Path(self.config.output_dir) / f"fold_{self.fold}"
            self.fold_checkpoint_dir = self.fold_dir / "checkpoints"
            self.fold_log_dir = self.fold_dir / "logs"
            
            for directory in [self.fold_dir, self.fold_checkpoint_dir, self.fold_log_dir]:
                directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self):
        """Set up optimizer and scheduler."""
        # Optimizer
        if self.config.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
        
        # Scheduler
        if self.config.scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.T_0,
                T_mult=self.config.T_mult,
                eta_min=self.config.eta_min,
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.monitor_mode,
                factor=0.5,
                patience=20,
                min_lr=self.config.eta_min,
            )
        elif self.config.scheduler == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=50,
                gamma=0.1,
            )
        else:
            self.scheduler = None
    
    def _setup_tensorboard(self):
        """Set up TensorBoard logging."""
        log_dir = self.fold_log_dir if self.fold is not None else self.config.log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def _setup_wandb(self):
        """Set up Weights & Biases logging."""
        project_name = "brats-3d-segmentation"
        run_name = f"fold_{self.fold}" if self.fold is not None else "training"
        
        wandb.init(
            project=project_name,
            name=run_name,
            config=self.config.__dict__,
        )
        wandb.watch(self.model)
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.max_epochs}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.use_amp and self.scaler is not None:
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss = self.loss_function(predictions, labels)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss = self.loss_function(predictions, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                # Optimizer step
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
            })
            
            # Log batch metrics
            global_step = self.current_epoch * num_batches + batch_idx
            self.writer.add_scalar("train/batch_loss", loss.item(), global_step)
            self.writer.add_scalar("train/learning_rate", self.optimizer.param_groups[0]['lr'], global_step)
            
            if self.config.use_wandb:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                    "global_step": global_step,
                })
        
        # Calculate average loss
        avg_loss = total_loss / num_batches
        
        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                # Will be called after validation
                pass
            else:
                self.scheduler.step()
        
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.logger.info("Running validation...")
        
        val_results = self.validator.validate(
            model=self.model,
            data_loader=self.val_loader,
            loss_function=self.loss_function,
        )
        
        return val_results
    
    def save_checkpoint(
        self,
        metrics: Dict[str, float],
        is_best: bool = False,
        is_last: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint_dir = self.fold_checkpoint_dir if self.fold is not None else self.config.checkpoint_dir
        
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_metric": self.best_metric,
            "training_history": self.training_history,
            "config": self.config,
            "metrics": metrics,
        }
        
        # Save regular checkpoint
        if self.fold is not None:
            checkpoint_path = checkpoint_dir / f"fold_{self.fold}_epoch_{self.current_epoch:03d}.pth"
        else:
            checkpoint_path = checkpoint_dir / f"epoch_{self.current_epoch:03d}.pth"
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = checkpoint_dir / ("fold_best.pth" if self.fold is not None else "best.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint: {best_path}")
        
        # Save last checkpoint
        if is_last:
            last_path = checkpoint_dir / ("fold_last.pth" if self.fold is not None else "last.pth")
            torch.save(checkpoint, last_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.scheduler and checkpoint["scheduler_state_dict"]:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if self.scaler and checkpoint["scaler_state_dict"]:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.current_epoch = checkpoint["epoch"]
        self.best_metric = checkpoint["best_metric"]
        self.training_history = checkpoint.get("training_history", self.training_history)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.current_epoch, self.config.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_loss = self.train_epoch()
            self.training_history["train_loss"].append(train_loss)
            
            # Validation
            if (epoch + 1) % self.config.validation_interval == 0:
                val_results = self.validate()
                
                # Extract key metrics
                val_loss = val_results.get("loss", float('inf'))
                val_dice_wt = val_results.get("WT_dice", 0.0)
                val_dice_tc = val_results.get("TC_dice", 0.0)
                val_dice_et = val_results.get("ET_dice", 0.0)
                val_dice_mean = (val_dice_wt + val_dice_tc + val_dice_et) / 3
                
                # Update history
                self.training_history["val_loss"].append(val_loss)
                self.training_history["val_dice_wt"].append(val_dice_wt)
                self.training_history["val_dice_tc"].append(val_dice_tc)
                self.training_history["val_dice_et"].append(val_dice_et)
                self.training_history["val_dice_mean"].append(val_dice_mean)
                
                # Check for best model
                current_metric = val_results.get(self.config.monitor_metric, val_dice_mean)
                is_best = False
                
                if self.config.monitor_mode == "max":
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                else:
                    if current_metric < self.best_metric:
                        self.best_metric = current_metric
                        is_best = True
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                
                # Update scheduler if using ReduceLROnPlateau
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(current_metric)
                
                # Log metrics
                epoch_time = time.time() - epoch_start_time
                
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Dice (WT/TC/ET): {val_dice_wt:.3f}/{val_dice_tc:.3f}/{val_dice_et:.3f}, "
                    f"Time: {epoch_time:.2f}s"
                )
                
                # TensorBoard logging
                self.writer.add_scalar("train/loss", train_loss, epoch)
                self.writer.add_scalar("val/loss", val_loss, epoch)
                self.writer.add_scalar("val/dice_wt", val_dice_wt, epoch)
                self.writer.add_scalar("val/dice_tc", val_dice_tc, epoch)
                self.writer.add_scalar("val/dice_et", val_dice_et, epoch)
                self.writer.add_scalar("val/dice_mean", val_dice_mean, epoch)
                
                # Wandb logging
                if self.config.use_wandb:
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/dice_wt": val_dice_wt,
                        "val/dice_tc": val_dice_tc,
                        "val/dice_et": val_dice_et,
                        "val/dice_mean": val_dice_mean,
                        "val/best_metric": self.best_metric,
                        "patience_counter": self.patience_counter,
                    })
                
                # Save checkpoint
                self.save_checkpoint(val_results, is_best=is_best)
                
                # Early stopping
                if self.patience_counter >= self.config.patience:
                    self.logger.info(f"Early stopping triggered after {self.config.patience} epochs without improvement")
                    break
            
            else:
                # Log training loss only
                self.writer.add_scalar("train/loss", train_loss, epoch)
                if self.config.use_wandb:
                    wandb.log({"epoch": epoch, "train/loss": train_loss})
        
        # Save final checkpoint
        self.save_checkpoint({}, is_last=True)
        
        self.logger.info("Training completed!")
        self.logger.info(f"Best {self.config.monitor_metric}: {self.best_metric:.4f}")
        
        # Close logging
        self.writer.close()
        if self.config.use_wandb:
            wandb.finish()
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        self.load_checkpoint(checkpoint_path)
        self.train()