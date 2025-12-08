"""
Convolutional Autoencoder for Anomaly Detection.

Architecture Overview:
    Input (3, 256, 256) 
        ↓
    [Encoder] - Series of Conv + BatchNorm + ReLU + MaxPool
        ↓ 
    Latent Space (256, 16, 16) - Compressed representation
        ↓
    [Decoder] - Series of ConvTranspose + BatchNorm + ReLU
        ↓
    Output (3, 256, 256) - Reconstructed image

The model learns to compress and reconstruct NORMAL images.
Anomalies cause high reconstruction error since the model
never learned to represent them.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder network - compresses input image to latent representation.
    
    Each block: Conv2d -> BatchNorm -> LeakyReLU -> MaxPool
    
    Spatial dimensions: 256 -> 128 -> 64 -> 32 -> 16
    Channels: 3 -> 32 -> 64 -> 128 -> 256
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        
        # Block 1: 256x256x3 -> 128x128x32
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)  # Halves spatial dimensions
        )
        
        # Block 2: 128x128x32 -> 64x64x64
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 3: 64x64x64 -> 32x32x128
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Block 4: 32x32x128 -> 16x16x256
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2, 2)
        )
    
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        return x


class Decoder(nn.Module):
    """
    Decoder network - reconstructs image from latent representation.
    
    Each block: ConvTranspose2d (upsample) -> BatchNorm -> ReLU -> Conv
    
    Spatial dimensions: 16 -> 32 -> 64 -> 128 -> 256
    Channels: 256 -> 128 -> 64 -> 32 -> 3
    """
    
    def __init__(self, out_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        
        # Block 1: 16x16x256 -> 32x32x128
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Block 2: 32x32x128 -> 64x64x64
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Block 3: 64x64x64 -> 128x128x32
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # Block 4: 128x128x32 -> 256x256x3
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range to match input normalization
        )
    
    def forward(self, x):
        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)
        x = self.dec4(x)
        return x


class ConvAutoencoder(nn.Module):
    """
    Complete Convolutional Autoencoder.
    
    Combines encoder and decoder into single model.
    
    Usage:
        model = ConvAutoencoder()
        reconstruction = model(image)
        loss = F.mse_loss(reconstruction, image)
    """
    
    def __init__(self, in_channels: int = 3, latent_dim: int = 256):
        super().__init__()
        
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Forward pass - encode then decode.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            Reconstructed image tensor [B, C, H, W]
        """
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
    
    def get_latent(self, x):
        """Get latent representation without decoding."""
        return self.encoder(x)
    
    def get_reconstruction_error(self, x, per_pixel: bool = False):
        """
        Compute reconstruction error (anomaly score).
        
        Args:
            x: Input image tensor [B, C, H, W]
            per_pixel: If True, return error map. If False, return scalar.
            
        Returns:
            If per_pixel: Error map [B, 1, H, W]
            Else: Scalar error per sample [B]
        """
        recon = self.forward(x)
        
        # MSE per pixel, averaged across channels
        error = (x - recon) ** 2
        error = error.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        if per_pixel:
            return error
        else:
            # Global anomaly score per image
            return error.mean(dim=[1, 2, 3])  # [B]


# Quick test
if __name__ == '__main__':
    model = ConvAutoencoder()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(4, 3, 256, 256)
    recon = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {recon.shape}")
    
    # Test error computation
    error_map = model.get_reconstruction_error(x, per_pixel=True)
    error_scalar = model.get_reconstruction_error(x, per_pixel=False)
    print(f"Error map shape: {error_map.shape}")
    print(f"Error scalar shape: {error_scalar.shape}")