"""
Split-conformal prediction for uncertainty quantification.
"""
import numpy as np

def compute_residuals(y_true, y_pred):
    """
    Compute L1 residuals per sample.
    
    Args:
        y_true: Ground truth array (n_samples, ...)
        y_pred: Predictions array (n_samples, ...)
    
    Returns:
        residuals: L1 residuals per sample (n_samples,)
    """
    # Flatten spatial dimensions for per-sample residuals
    y_true_flat = y_true.reshape(len(y_true), -1)
    y_pred_flat = y_pred.reshape(len(y_pred), -1)
    
    # Compute L1 residual per sample
    residuals = np.mean(np.abs(y_true_flat - y_pred_flat), axis=1)
    
    return residuals

def fit_conformal(residuals, alpha):
    """
    Fit conformal quantile from validation residuals.
    
    Args:
        residuals: Validation residuals (n_val_samples,)
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
    
    Returns:
        q_alpha: Conformal quantile
    """
    n = len(residuals)
    # Conformal quantile: (n+1)(1-alpha)/n percentile
    q_level = 1 - alpha
    q_alpha = np.quantile(residuals, q_level)
    
    return q_alpha

def apply_conformal(mean, base_sigma, q_alpha, mode='absolute'):
    """
    Apply conformal prediction to get prediction intervals.
    
    Args:
        mean: Predicted mean (n_samples, ...)
        base_sigma: Base uncertainty estimate (n_samples, ...)
        q_alpha: Conformal quantile
        mode: 'absolute' or 'scaled'
    
    Returns:
        lo: Lower prediction interval
        hi: Upper prediction interval
    """
    if mode == 'absolute':
        # Absolute intervals: mean ± q_alpha
        lo = mean - q_alpha
        hi = mean + q_alpha
    elif mode == 'scaled':
        # Scaled intervals: mean ± q_alpha * base_sigma
        lo = mean - q_alpha * base_sigma
        hi = mean + q_alpha * base_sigma
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return lo, hi

def compute_coverage_metrics(y_true, lo, hi):
    """
    Compute coverage metrics for prediction intervals.
    
    Args:
        y_true: Ground truth (n_samples, ...)
        lo: Lower bounds (n_samples, ...)
        hi: Upper bounds (n_samples, ...)
    
    Returns:
        metrics: Dictionary with coverage and width metrics
    """
    # Check if true values are within intervals
    within_interval = (y_true >= lo) & (y_true <= hi)
    
    # Coverage: fraction of true values within intervals
    coverage = np.mean(within_interval)
    
    # Average interval width
    avg_width = np.mean(hi - lo)
    
    # Per-sample coverage (for spatial data)
    if len(y_true.shape) > 1:
        # Spatial coverage: fraction of voxels covered per sample
        spatial_coverage = np.mean(within_interval, axis=tuple(range(1, len(y_true.shape))))
        per_sample_coverage = np.mean(spatial_coverage)
    else:
        per_sample_coverage = coverage
    
    return {
        'coverage': float(coverage),
        'avg_width': float(avg_width),
        'per_sample_coverage': float(per_sample_coverage)
    }

def compute_adaptive_coverage(y_true, lo, hi, confidence_levels=[0.8, 0.9, 0.95]):
    """
    Compute coverage at multiple confidence levels.
    
    Args:
        y_true: Ground truth
        lo: Lower bounds  
        hi: Upper bounds
        confidence_levels: List of confidence levels to evaluate
    
    Returns:
        coverage_dict: Coverage at each confidence level
    """
    coverage_dict = {}
    
    for conf_level in confidence_levels:
        # Compute coverage metrics
        metrics = compute_coverage_metrics(y_true, lo, hi)
        coverage_dict[f'cov{int(conf_level*100)}'] = metrics['coverage']
        coverage_dict[f'width{int(conf_level*100)}'] = metrics['avg_width']
    
    return coverage_dict
