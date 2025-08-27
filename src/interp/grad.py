import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Literal

def integrated_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    steps: int = 32,
    baseline: Literal['zeros', 'gauss'] = 'zeros',
    target: str = 'pressure'
) -> torch.Tensor:
    """
    Compute Integrated Gradients for 3D input.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (1, C, D, H, W)
        steps: Number of integration steps
        baseline: 'zeros' or 'gauss' baseline
        target: Target field name (unused, assumes model outputs pressure)
    
    Returns:
        Attribution tensor of same shape as input
    """
    model.eval()
    x = x.requires_grad_(True)
    
    # Create baseline
    if baseline == 'zeros':
        baseline_tensor = torch.zeros_like(x)
    elif baseline == 'gauss':
        baseline_tensor = torch.randn_like(x) * 0.1
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    
    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, steps + 1, device=x.device)
    gradients = []
    
    for alpha in alphas:
        # Interpolate between baseline and input
        interpolated = baseline_tensor + alpha * (x - baseline_tensor)
        interpolated.requires_grad_(True)
        
        # Forward pass
        output = model(interpolated)
        
        # Compute gradient w.r.t. interpolated input
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=interpolated,
            create_graph=False,
            retain_graph=False
        )[0]
        
        gradients.append(grad)
    
    # Average gradients and multiply by input difference
    avg_gradients = torch.stack(gradients).mean(dim=0)
    integrated_grads = (x - baseline_tensor) * avg_gradients
    
    return integrated_grads.detach()

def gradient_shap(
    model: torch.nn.Module,
    x: torch.Tensor,
    nsamples: int = 20,
    target: str = 'pressure'
) -> torch.Tensor:
    """
    Compute GradientSHAP for 3D input using noise baselines.
    
    Args:
        model: PyTorch model
        x: Input tensor of shape (1, C, D, H, W)
        nsamples: Number of noise baseline samples
        target: Target field name (unused)
    
    Returns:
        Attribution tensor of same shape as input
    """
    model.eval()
    x = x.requires_grad_(True)
    
    attributions = []
    
    for _ in range(nsamples):
        # Random noise baseline
        baseline = torch.randn_like(x) * 0.1
        
        # Random interpolation coefficient
        alpha = torch.rand(1, device=x.device)
        
        # Interpolated input
        interpolated = baseline + alpha * (x - baseline)
        interpolated.requires_grad_(True)
        
        # Forward pass
        output = model(interpolated)
        
        # Compute gradient
        grad = torch.autograd.grad(
            outputs=output.sum(),
            inputs=interpolated,
            create_graph=False,
            retain_graph=False
        )[0]
        
        # Attribution for this sample
        attr = (x - baseline) * grad
        attributions.append(attr)
    
    # Average over samples
    mean_attribution = torch.stack(attributions).mean(dim=0)
    
    return mean_attribution.detach()
