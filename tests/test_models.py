import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.unet3d import UNet3D, UNet3DConfig, create_unet3d
from models.blocks import ResidualUnit3D, DownBlock3D, UpBlock3D, AttentionGate3D


class TestResidualUnit3D:
    """Test 3D residual unit block."""
    
    def test_residual_unit_forward(self):
        """Test forward pass through residual unit."""
        unit = ResidualUnit3D(in_channels=32, out_channels=64, stride=1)
        
        x = torch.randn(2, 32, 16, 16, 16)
        output = unit(x)
        
        assert output.shape == (2, 64, 16, 16, 16)
    
    def test_residual_unit_stride(self):
        """Test residual unit with stride > 1."""
        unit = ResidualUnit3D(in_channels=32, out_channels=64, stride=2)
        
        x = torch.randn(2, 32, 16, 16, 16)
        output = unit(x)
        
        assert output.shape == (2, 64, 8, 8, 8)
    
    def test_residual_unit_same_channels(self):
        """Test residual unit with same input/output channels."""
        unit = ResidualUnit3D(in_channels=64, out_channels=64, stride=1)
        
        x = torch.randn(2, 64, 16, 16, 16)
        output = unit(x)
        
        assert output.shape == x.shape
    
    def test_residual_unit_dropout(self):
        """Test residual unit with dropout."""
        unit = ResidualUnit3D(in_channels=32, out_channels=64, dropout_rate=0.5)
        
        x = torch.randn(2, 32, 16, 16, 16)
        
        # Test training mode (dropout active)
        unit.train()
        output_train = unit(x)
        
        # Test evaluation mode (dropout inactive)
        unit.eval()
        output_eval = unit(x)
        
        assert output_train.shape == output_eval.shape == (2, 64, 16, 16, 16)


class TestDownBlock3D:
    """Test 3D downsampling block."""
    
    def test_down_block_forward(self):
        """Test downsampling block forward pass."""
        block = DownBlock3D(in_channels=32, out_channels=64, num_residual_units=2)
        
        x = torch.randn(2, 32, 16, 16, 16)
        output = block(x)
        
        assert output.shape == (2, 64, 8, 8, 8)  # Halved spatial dimensions
    
    def test_down_block_residual_units(self):
        """Test downsampling block with different numbers of residual units."""
        for num_units in [1, 2, 3]:
            block = DownBlock3D(
                in_channels=32, 
                out_channels=64, 
                num_residual_units=num_units
            )
            
            x = torch.randn(1, 32, 16, 16, 16)
            output = block(x)
            
            assert output.shape == (1, 64, 8, 8, 8)
            assert len(block.residual_units) == num_units - 1  # First unit is down_conv


class TestUpBlock3D:
    """Test 3D upsampling block."""
    
    def test_up_block_forward(self):
        """Test upsampling block forward pass."""
        block = UpBlock3D(
            in_channels=128, 
            skip_channels=64, 
            out_channels=64,
            num_residual_units=2
        )
        
        x = torch.randn(2, 128, 8, 8, 8)
        skip = torch.randn(2, 64, 16, 16, 16)
        output = block(x, skip)
        
        assert output.shape == (2, 64, 16, 16, 16)
    
    def test_up_block_transpose_conv(self):
        """Test upsampling block with transpose convolution."""
        block = UpBlock3D(
            in_channels=128, 
            skip_channels=64, 
            out_channels=64,
            use_transpose_conv=True
        )
        
        x = torch.randn(1, 128, 8, 8, 8)
        skip = torch.randn(1, 64, 16, 16, 16)
        output = block(x, skip)
        
        assert output.shape == (1, 64, 16, 16, 16)
    
    def test_up_block_interpolation(self):
        """Test upsampling block with interpolation."""
        block = UpBlock3D(
            in_channels=128, 
            skip_channels=64, 
            out_channels=64,
            use_transpose_conv=False
        )
        
        x = torch.randn(1, 128, 8, 8, 8)
        skip = torch.randn(1, 64, 16, 16, 16)
        output = block(x, skip)
        
        assert output.shape == (1, 64, 16, 16, 16)


class TestAttentionGate3D:
    """Test 3D attention gate."""
    
    def test_attention_gate_forward(self):
        """Test attention gate forward pass."""
        attention = AttentionGate3D(gate_channels=256, skip_channels=128)
        
        gate = torch.randn(2, 256, 8, 8, 8)
        skip = torch.randn(2, 128, 16, 16, 16)
        
        output = attention(gate, skip)
        
        assert output.shape == skip.shape  # Should preserve skip shape
    
    def test_attention_gate_same_size(self):
        """Test attention gate with same spatial size."""
        attention = AttentionGate3D(gate_channels=128, skip_channels=128)
        
        gate = torch.randn(1, 128, 16, 16, 16)
        skip = torch.randn(1, 128, 16, 16, 16)
        
        output = attention(gate, skip)
        
        assert output.shape == skip.shape


class TestUNet3D:
    """Test complete 3D U-Net model."""
    
    def test_default_unet3d(self):
        """Test default U-Net configuration."""
        config = UNet3DConfig()
        model = UNet3D(config)
        
        x = torch.randn(2, 4, 64, 64, 64)
        output = model(x)
        
        assert output.shape == (2, 4, 64, 64, 64)
    
    def test_custom_unet3d_config(self):
        """Test U-Net with custom configuration."""
        config = UNet3DConfig(
            input_channels=1,
            output_channels=2,
            feature_maps=[16, 32, 64, 128],
            deep_supervision=False,
        )
        model = UNet3D(config)
        
        x = torch.randn(1, 1, 32, 32, 32)
        output = model(x)
        
        assert output.shape == (1, 2, 32, 32, 32)
    
    def test_unet3d_deep_supervision(self):
        """Test U-Net with deep supervision."""
        config = UNet3DConfig(
            input_channels=4,
            output_channels=4,
            feature_maps=[32, 64, 128, 256],
            deep_supervision=True,
        )
        model = UNet3D(config)
        
        x = torch.randn(1, 4, 32, 32, 32)
        outputs = model(x)
        
        # Should return list of outputs
        assert isinstance(outputs, list)
        assert len(outputs) == len(config.feature_maps) - 1  # Number of decoder levels
        
        # Final output should have correct shape
        assert outputs[-1].shape == (1, 4, 32, 32, 32)
    
    def test_unet3d_attention(self):
        """Test U-Net with attention gates."""
        config = UNet3DConfig(
            input_channels=4,
            output_channels=4,
            use_attention=True,
        )
        model = UNet3D(config)
        
        x = torch.randn(1, 4, 64, 64, 64)
        output = model(x)
        
        assert output.shape == (1, 4, 64, 64, 64)
        assert model.attention_gates is not None
    
    def test_create_unet3d_factory(self):
        """Test U-Net factory function."""
        model = create_unet3d(
            input_channels=3,
            output_channels=5,
            feature_maps=[16, 32, 64],
            deep_supervision=True,
        )
        
        x = torch.randn(1, 3, 32, 32, 32)
        outputs = model(x)
        
        assert isinstance(outputs, list)
        assert outputs[-1].shape == (1, 5, 32, 32, 32)
    
    def test_model_parameter_count(self):
        """Test that model has reasonable parameter count."""
        config = UNet3DConfig(
            feature_maps=[32, 64, 128, 256, 512]
        )
        model = UNet3D(config)
        
        total_params, model_size_mb = model.get_model_size()
        
        assert total_params > 1000  # Should have at least 1K parameters
        assert total_params < 100_000_000  # Should be less than 100M parameters
        assert model_size_mb > 0
    
    def test_model_modes(self):
        """Test model in training and evaluation modes."""
        model = create_unet3d()
        x = torch.randn(1, 4, 32, 32, 32)
        
        # Test training mode
        model.train()
        output_train = model(x)
        
        # Test evaluation mode
        model.eval()
        with torch.no_grad():
            output_eval = model(x)
        
        assert output_train.shape == output_eval.shape
    
    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = create_unet3d()
        
        input_sizes = [
            (16, 16, 16),
            (32, 32, 32),
            (64, 64, 64),
            (48, 48, 48),
        ]
        
        for size in input_sizes:
            x = torch.randn(1, 4, *size)
            output = model(x)
            assert output.shape == (1, 4, *size)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = create_unet3d()
        x = torch.randn(1, 4, 32, 32, 32, requires_grad=True)
        target = torch.randint(0, 4, (1, 32, 32, 32)).float()
        
        output = model(x)
        loss = nn.functional.mse_loss(output, target.unsqueeze(1).expand_as(output))
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        
        # Check that model parameters have gradients
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
    
    def test_model_device_placement(self):
        """Test model device placement."""
        model = create_unet3d()
        
        if torch.cuda.is_available():
            # Test CUDA placement
            model = model.cuda()
            x = torch.randn(1, 4, 32, 32, 32).cuda()
            
            output = model(x)
            assert output.is_cuda
        
        # Test CPU placement
        model = model.cpu()
        x = torch.randn(1, 4, 32, 32, 32).cpu()
        
        output = model(x)
        assert not output.is_cuda


def test_model_memory_efficiency():
    """Test model memory usage."""
    # Test with different batch sizes
    batch_sizes = [1, 2, 4]
    model = create_unet3d()
    
    for batch_size in batch_sizes:
        x = torch.randn(batch_size, 4, 32, 32, 32)
        
        # Forward pass should not raise memory errors
        with torch.no_grad():
            output = model(x)
            assert output.shape[0] == batch_size


def test_model_serialization():
    """Test model save/load functionality."""
    import tempfile
    import os
    
    model = create_unet3d()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save model
        model_path = os.path.join(temp_dir, "test_model.pth")
        torch.save(model.state_dict(), model_path)
        
        # Load model
        new_model = create_unet3d()
        new_model.load_state_dict(torch.load(model_path))
        
        # Test that models produce same output
        x = torch.randn(1, 4, 32, 32, 32)
        
        model.eval()
        new_model.eval()
        
        with torch.no_grad():
            output1 = model(x)
            output2 = new_model(x)
        
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])