import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .blocks import ResidualUnit3D, DownBlock3D, UpBlock3D, AttentionGate3D


@dataclass
class UNet3DConfig:
    """Configuration class for 3D U-Net model."""
    
    input_channels: int = 4
    output_channels: int = 4
    feature_maps: List[int] = None
    num_residual_units: int = 2
    dropout_rate: float = 0.1
    deep_supervision: bool = False
    use_attention: bool = False
    use_transpose_conv: bool = True
    
    def __post_init__(self):
        if self.feature_maps is None:
            self.feature_maps = [32, 64, 128, 256, 512]


class UNet3D(nn.Module):
    """3D U-Net with residual blocks for medical image segmentation."""
    
    def __init__(self, config: UNet3DConfig):
        """
        Initialize 3D U-Net model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        self.deep_supervision = config.deep_supervision
        self.use_attention = config.use_attention
        
        # Input convolution
        self.input_conv = ResidualUnit3D(
            config.input_channels,
            config.feature_maps[0],
            stride=1,
            dropout_rate=config.dropout_rate,
        )
        
        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        in_channels = config.feature_maps[0]
        
        for i, out_channels in enumerate(config.feature_maps[1:]):
            block = DownBlock3D(
                in_channels=in_channels,
                out_channels=out_channels,
                num_residual_units=config.num_residual_units,
                dropout_rate=config.dropout_rate,
            )
            self.encoder_blocks.append(block)
            in_channels = out_channels
        
        # Bottleneck
        bottleneck_channels = config.feature_maps[-1]
        self.bottleneck = nn.Sequential(
            ResidualUnit3D(
                bottleneck_channels,
                bottleneck_channels,
                stride=1,
                dropout_rate=0.2,  # Higher dropout in bottleneck
            ),
            ResidualUnit3D(
                bottleneck_channels,
                bottleneck_channels,
                stride=1,
                dropout_rate=0.2,
            ),
        )
        
        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.attention_gates = nn.ModuleList() if config.use_attention else None
        
        decoder_channels = list(reversed(config.feature_maps))
        
        for i in range(len(decoder_channels) - 1):
            in_channels = decoder_channels[i]
            skip_channels = decoder_channels[i + 1]
            out_channels = decoder_channels[i + 1]
            
            # Attention gate
            if config.use_attention:
                attention_gate = AttentionGate3D(
                    gate_channels=in_channels,
                    skip_channels=skip_channels,
                )
                self.attention_gates.append(attention_gate)
            
            # Decoder block
            block = UpBlock3D(
                in_channels=in_channels,
                skip_channels=skip_channels,
                out_channels=out_channels,
                num_residual_units=config.num_residual_units,
                dropout_rate=config.dropout_rate,
                use_transpose_conv=config.use_transpose_conv,
            )
            self.decoder_blocks.append(block)
        
        # Final output convolution
        self.final_conv = nn.Conv3d(
            config.feature_maps[0],
            config.output_channels,
            kernel_size=1,
        )
        
        # Deep supervision outputs
        if config.deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for i in range(len(decoder_channels) - 1):
                out_channels = decoder_channels[i + 1]
                ds_conv = nn.Conv3d(out_channels, config.output_channels, kernel_size=1)
                self.deep_supervision_outputs.append(ds_conv)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 3D U-Net.
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            Output tensor of shape (B, num_classes, D, H, W)
            If deep supervision enabled, returns list of outputs
        """
        # Input convolution
        x = self.input_conv(x)
        
        # Encoder path with skip connections
        skip_connections = [x]
        
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            skip_connections.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        skip_connections = skip_connections[:-1]  # Remove last (bottleneck)
        skip_connections.reverse()  # Reverse for decoder
        
        deep_outputs = []
        
        for i, (decoder_block, skip) in enumerate(zip(self.decoder_blocks, skip_connections)):
            # Apply attention gate if enabled
            if self.use_attention:
                skip = self.attention_gates[i](gate=x, skip=skip)
            
            # Decoder block
            x = decoder_block(x, skip)
            
            # Deep supervision output
            if self.deep_supervision:
                ds_out = self.deep_supervision_outputs[i](x)
                deep_outputs.append(ds_out)
        
        # Final output
        output = self.final_conv(x)
        
        if self.deep_supervision:
            deep_outputs.append(output)
            return deep_outputs
        else:
            return output
    
    def get_model_size(self) -> Tuple[int, float]:
        """
        Get model size information.
        
        Returns:
            Tuple of (total_parameters, model_size_mb)
        """
        total_params = sum(p.numel() for p in self.parameters())
        model_size_mb = total_params * 4 / (1024 ** 2)  # Assuming float32
        return total_params, model_size_mb
    
    def print_model_info(self):
        """Print model architecture information."""
        total_params, model_size_mb = self.get_model_size()
        
        print(f"3D U-Net Model Information:")
        print(f"  Input channels: {self.config.input_channels}")
        print(f"  Output channels: {self.config.output_channels}")
        print(f"  Feature maps: {self.config.feature_maps}")
        print(f"  Deep supervision: {self.config.deep_supervision}")
        print(f"  Attention gates: {self.config.use_attention}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Model size: {model_size_mb:.2f} MB")
        
        # Print layer-wise information
        print(f"\nArchitecture Summary:")
        print(f"  Input Conv: {self.config.input_channels} -> {self.config.feature_maps[0]}")
        
        for i, channels in enumerate(self.config.feature_maps[1:]):
            prev_channels = self.config.feature_maps[i]
            print(f"  Encoder {i+1}: {prev_channels} -> {channels}")
        
        for i in range(len(self.config.feature_maps) - 1):
            in_ch = self.config.feature_maps[-(i+1)]
            out_ch = self.config.feature_maps[-(i+2)]
            print(f"  Decoder {i+1}: {in_ch} -> {out_ch}")
        
        print(f"  Final Conv: {self.config.feature_maps[0]} -> {self.config.output_channels}")


def create_unet3d(
    input_channels: int = 4,
    output_channels: int = 4,
    feature_maps: Optional[List[int]] = None,
    deep_supervision: bool = False,
    use_attention: bool = False,
    dropout_rate: float = 0.1,
) -> UNet3D:
    """
    Factory function to create 3D U-Net model.
    
    Args:
        input_channels: Number of input channels
        output_channels: Number of output channels  
        feature_maps: List of feature map sizes for each level
        deep_supervision: Whether to use deep supervision
        use_attention: Whether to use attention gates
        dropout_rate: Dropout probability
        
    Returns:
        Configured UNet3D model
    """
    config = UNet3DConfig(
        input_channels=input_channels,
        output_channels=output_channels,
        feature_maps=feature_maps,
        deep_supervision=deep_supervision,
        use_attention=use_attention,
        dropout_rate=dropout_rate,
    )
    
    return UNet3D(config)