"""
Custom loss functions for anomaly detection.

SSIM (Structural Similarity Index) measures perceptual similarity
between images based on luminance, contrast, and structure.
It's better than MSE for detecting structural defects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM Loss - measures structural similarity between images.
    
    SSIM looks at:
        - Luminance: mean pixel intensity
        - Contrast: variance of pixel intensity  
        - Structure: correlation between pixels
    
    Returns 1 - SSIM, so lower is better (like MSE).
    """
    
    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        
        # Create Gaussian window for weighted averaging
        self.window = self._create_gaussian_window(window_size, channels)
    
    def _create_gaussian_window(self, size: int, channels: int):
        """Create a Gaussian kernel for local statistics."""
        # 1D Gaussian
        sigma = 1.5
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        gauss = torch.exp(-coords**2 / (2 * sigma**2))
        gauss = gauss / gauss.sum()
        
        # 2D Gaussian (outer product)
        window_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        
        # Expand for conv2d: [channels, 1, size, size]
        window = window_2d.unsqueeze(0).unsqueeze(0)
        window = window.expand(channels, 1, size, size).contiguous()
        
        return window
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        Compute SSIM loss.
        
        Args:
            pred: Predicted image [B, C, H, W]
            target: Target image [B, C, H, W]
            
        Returns:
            1 - SSIM (so we can minimize it)
        """
        # Move window to same device as input
        if self.window.device != pred.device:
            self.window = self.window.to(pred.device)
        
        channels = pred.shape[1]
        window = self.window
        
        # Compute local means using Gaussian-weighted average
        mu_pred = F.conv2d(pred, window, padding=self.window_size//2, groups=channels)
        mu_target = F.conv2d(target, window, padding=self.window_size//2, groups=channels)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        # Compute local variances and covariance
        sigma_pred_sq = F.conv2d(pred**2, window, padding=self.window_size//2, groups=channels) - mu_pred_sq
        sigma_target_sq = F.conv2d(target**2, window, padding=self.window_size//2, groups=channels) - mu_target_sq
        sigma_pred_target = F.conv2d(pred*target, window, padding=self.window_size//2, groups=channels) - mu_pred_target
        
        # Constants for numerical stability
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # SSIM formula
        numerator = (2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)
        denominator = (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        
        ssim_map = numerator / denominator
        
        # Return 1 - SSIM (so we minimize loss)
        return 1 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """
    Combined MSE + SSIM loss.
    
    MSE captures pixel-level accuracy.
    SSIM captures structural/perceptual similarity.
    Together they work better than either alone.
    """
    
    def __init__(self, alpha: float = 0.5, window_size: int = 11):
        """
        Args:
            alpha: Weight for SSIM loss. Final loss = (1-alpha)*MSE + alpha*SSIM
                   alpha=0.5 means equal weighting
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss(window_size=window_size)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        mse_loss = self.mse(pred, target)
        ssim_loss = self.ssim(pred, target)
        
        combined = (1 - self.alpha) * mse_loss + self.alpha * ssim_loss
        return combined


# Test the losses
if __name__ == '__main__':
    # Create test images
    img1 = torch.randn(4, 3, 256, 256)
    img2 = img1 + 0.1 * torch.randn_like(img1)  # Slightly noisy version
    img3 = torch.randn(4, 3, 256, 256)  # Completely different
    
    ssim_loss = SSIMLoss()
    combined_loss = CombinedLoss(alpha=0.5)
    
    print("SSIM Loss (similar images):", ssim_loss(img1, img2).item())
    print("SSIM Loss (different images):", ssim_loss(img1, img3).item())
    print()
    print("Combined Loss (similar):", combined_loss(img1, img2).item())
    print("Combined Loss (different):", combined_loss(img1, img3).item())