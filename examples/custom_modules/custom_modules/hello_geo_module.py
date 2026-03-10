import torch
import torch.nn as nn
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY

# Register the class with the backbone registry
@TERRATORCH_BACKBONE_REGISTRY.register
class HelloGeoModule(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super().__init__()
        # A simple convolution to demonstrate processing
        self.conv = nn.Conv2d(
            input_channels, 
            output_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        
        # Required attribute for EncoderDecoderFactory
        self.out_channels = [output_channels]

    def forward(self, x):
        print(f"Input shape: {x.shape}")
        x = self.conv(x)
        x = self.relu(x)
        print(f"Output shape: {x.shape}")
        # Return as a list of tensors (required by EncoderDecoderFactory)
        return [x]