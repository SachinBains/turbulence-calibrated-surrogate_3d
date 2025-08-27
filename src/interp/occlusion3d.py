import torch
import torch.nn.functional as F
import numpy as np
from typing import Literal

def occlusion_importance(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: str = 'pressure',
    patch: int = 8,
    stride: int = 8,
    agg: Literal['sum', 'mean'] = 'sum'
) -> torch.Tensor:
    """
    Compute occlusion-based importance for 3D input.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (1, C, D, H, W)
        target: Target field name (unused)
        patch: Size of occlusion patch (cubic)
        stride: Stride for sliding window
        agg: Aggregation method for output ('sum' or 'mean')
    
    Returns:
        Importance tensor (downsampled grid based on patch/stride)
    """
    model.eval()
    
    with torch.no_grad():
        # Get original prediction
        original_output = model(x)
        if agg == 'sum':
            original_score = original_output.sum()
        else:
            original_score = original_output.mean()
    
    B, C, D, H, W = x.shape
    
    # Calculate output grid dimensions
    d_steps = max(1, (D - patch) // stride + 1)
    h_steps = max(1, (H - patch) // stride + 1)
    w_steps = max(1, (W - patch) // stride + 1)
    
    importance = torch.zeros((d_steps, h_steps, w_steps), device=x.device)
    
    # Slide occlusion patch
    for i, d_start in enumerate(range(0, D - patch + 1, stride)):
        for j, h_start in enumerate(range(0, H - patch + 1, stride)):
            for k, w_start in enumerate(range(0, W - patch + 1, stride)):
                # Create occluded input (zero out patch)
                x_occluded = x.clone()
                x_occluded[:, :, 
                          d_start:d_start + patch,
                          h_start:h_start + patch,
                          w_start:w_start + patch] = 0
                
                with torch.no_grad():
                    # Get occluded prediction
                    occluded_output = model(x_occluded)
                    if agg == 'sum':
                        occluded_score = occluded_output.sum()
                    else:
                        occluded_score = occluded_output.mean()
                
                # Importance is the difference (how much performance drops)
                importance[i, j, k] = abs(original_score - occluded_score)
    
    return importance.detach()

def blur_occlusion_importance(
    model: torch.nn.Module,
    x: torch.Tensor,
    target: str = 'pressure',
    patch: int = 8,
    stride: int = 8,
    blur_sigma: float = 2.0,
    agg: Literal['sum', 'mean'] = 'sum'
) -> torch.Tensor:
    """
    Compute occlusion-based importance using Gaussian blur instead of zeroing.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (1, C, D, H, W)
        target: Target field name (unused)
        patch: Size of occlusion patch (cubic)
        stride: Stride for sliding window
        blur_sigma: Standard deviation for Gaussian blur
        agg: Aggregation method for output ('sum' or 'mean')
    
    Returns:
        Importance tensor (downsampled grid based on patch/stride)
    """
    model.eval()
    
    with torch.no_grad():
        # Get original prediction
        original_output = model(x)
        if agg == 'sum':
            original_score = original_output.sum()
        else:
            original_score = original_output.mean()
    
    B, C, D, H, W = x.shape
    
    # Calculate output grid dimensions
    d_steps = max(1, (D - patch) // stride + 1)
    h_steps = max(1, (H - patch) // stride + 1)
    w_steps = max(1, (W - patch) // stride + 1)
    
    importance = torch.zeros((d_steps, h_steps, w_steps), device=x.device)
    
    # Create Gaussian kernel for blurring
    kernel_size = min(patch, 7)  # Limit kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Slide occlusion patch
    for i, d_start in enumerate(range(0, D - patch + 1, stride)):
        for j, h_start in enumerate(range(0, H - patch + 1, stride)):
            for k, w_start in enumerate(range(0, W - patch + 1, stride)):
                # Create blurred input
                x_occluded = x.clone()
                
                # Extract patch
                patch_region = x_occluded[:, :, 
                                        d_start:d_start + patch,
                                        h_start:h_start + patch,
                                        w_start:w_start + patch]
                
                # Apply Gaussian blur to patch (simplified 1D blur for efficiency)
                blurred_patch = F.gaussian_blur(
                    patch_region.view(-1, patch, patch, patch),
                    kernel_size=[kernel_size, kernel_size],
                    sigma=[blur_sigma, blur_sigma]
                ).view_as(patch_region)
                
                x_occluded[:, :, 
                          d_start:d_start + patch,
                          h_start:h_start + patch,
                          w_start:w_start + patch] = blurred_patch
                
                with torch.no_grad():
                    # Get occluded prediction
                    occluded_output = model(x_occluded)
                    if agg == 'sum':
                        occluded_score = occluded_output.sum()
                    else:
                        occluded_score = occluded_output.mean()
                
                # Importance is the difference
                importance[i, j, k] = abs(original_score - occluded_score)
    
    return importance.detach()
