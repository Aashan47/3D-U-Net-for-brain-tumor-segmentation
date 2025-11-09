import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualUnit3D(nn.Module):
    """3D Residual block with instance normalization and dropout."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize 3D residual unit.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Convolution stride
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True)
        
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True)
        
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else None
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual unit."""
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += identity
        out = self.activation(out)
        
        return out


class DownBlock3D(nn.Module):
    """Downsampling block with residual units."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_residual_units: int = 2,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize downsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            num_residual_units: Number of residual units in block
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Downsampling residual unit (stride=2)
        self.down_conv = ResidualUnit3D(
            in_channels, out_channels, stride=2, dropout_rate=dropout_rate
        )
        
        # Additional residual units (stride=1)
        self.residual_units = nn.ModuleList([
            ResidualUnit3D(out_channels, out_channels, stride=1, dropout_rate=dropout_rate)
            for _ in range(num_residual_units - 1)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through downsampling block."""
        x = self.down_conv(x)
        
        for unit in self.residual_units:
            x = unit(x)
        
        return x


class UpBlock3D(nn.Module):
    """Upsampling block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_residual_units: int = 2,
        dropout_rate: float = 0.1,
        use_transpose_conv: bool = True,
    ):
        """
        Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            skip_channels: Number of skip connection channels
            out_channels: Number of output channels
            num_residual_units: Number of residual units in block
            dropout_rate: Dropout probability
            use_transpose_conv: Whether to use transpose convolution for upsampling
        """
        super().__init__()
        
        self.use_transpose_conv = use_transpose_conv
        
        if use_transpose_conv:
            # Transposed convolution for upsampling
            self.upsample = nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        else:
            # Trilinear interpolation + convolution
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                nn.Conv3d(in_channels, out_channels, kernel_size=1),
            )
        
        # Fusion convolution after concatenating skip connection
        combined_channels = out_channels + skip_channels
        self.fusion_conv = nn.Conv3d(
            combined_channels, out_channels, kernel_size=1, bias=False
        )
        self.fusion_norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        # Residual units
        self.residual_units = nn.ModuleList([
            ResidualUnit3D(out_channels, out_channels, stride=1, dropout_rate=dropout_rate)
            for _ in range(num_residual_units)
        ])
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through upsampling block.
        
        Args:
            x: Input tensor
            skip: Skip connection tensor
            
        Returns:
            Upsampled and fused tensor
        """
        # Upsample input
        x = self.upsample(x)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Fusion
        x = self.fusion_conv(x)
        x = self.fusion_norm(x)
        x = self.activation(x)
        
        # Apply residual units
        for unit in self.residual_units:
            x = unit(x)
        
        return x


class AttentionGate3D(nn.Module):
    """3D Attention gate for skip connections."""
    
    def __init__(
        self,
        gate_channels: int,
        skip_channels: int,
        inter_channels: Optional[int] = None,
    ):
        """
        Initialize 3D attention gate.
        
        Args:
            gate_channels: Number of gate signal channels
            skip_channels: Number of skip connection channels
            inter_channels: Number of intermediate channels
        """
        super().__init__()
        
        if inter_channels is None:
            inter_channels = skip_channels // 2
        
        # Gate signal processing
        self.gate_conv = nn.Sequential(
            nn.Conv3d(gate_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels),
        )
        
        # Skip connection processing
        self.skip_conv = nn.Sequential(
            nn.Conv3d(skip_channels, inter_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(inter_channels),
        )
        
        # Attention computation
        self.attention_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv3d(inter_channels, 1, kernel_size=1, bias=False),
            nn.InstanceNorm3d(1),
            nn.Sigmoid(),
        )
    
    def forward(
        self, gate: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through attention gate.
        
        Args:
            gate: Gate signal tensor
            skip: Skip connection tensor
            
        Returns:
            Attention-weighted skip tensor
        """
        # Process gate and skip signals
        gate_processed = self.gate_conv(gate)
        skip_processed = self.skip_conv(skip)
        
        # Upsample gate to match skip dimensions if needed
        if gate_processed.shape[2:] != skip_processed.shape[2:]:
            gate_processed = F.interpolate(
                gate_processed, 
                size=skip_processed.shape[2:], 
                mode="trilinear",
                align_corners=False
            )
        
        # Compute attention weights
        attention = self.attention_conv(gate_processed + skip_processed)
        
        # Apply attention to skip connection
        return skip * attention