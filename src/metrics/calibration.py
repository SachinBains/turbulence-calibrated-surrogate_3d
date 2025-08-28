"""
Calibration metrics and reliability analysis for uncertainty quantification.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

def reliability_diagram_bins(y_true: np.ndarray, y_pred: np.ndarray, 
                           uncertainty: np.ndarray, n_bins: int = 10) -> Dict:
    """
    Compute reliability diagram data for calibration assessment.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values  
        uncertainty: Predicted uncertainties (standard deviations)
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary with bin data for reliability diagram
    """
    # Compute prediction errors
    errors = np.abs(y_true - y_pred)
    
    # Create bins based on predicted uncertainty
    uncertainty_flat = uncertainty.flatten()
    errors_flat = errors.flatten()
    
    # Remove any invalid values
    valid_mask = np.isfinite(uncertainty_flat) & np.isfinite(errors_flat)
    uncertainty_flat = uncertainty_flat[valid_mask]
    errors_flat = errors_flat[valid_mask]
    
    if len(uncertainty_flat) == 0:
        return {'bin_boundaries': [], 'bin_lowers': [], 'bin_uppers': [], 
                'bin_centers': [], 'observed_freq': [], 'expected_freq': [],
                'bin_counts': []}
    
    # Create bins
    bin_boundaries = np.linspace(uncertainty_flat.min(), uncertainty_flat.max(), n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    observed_freq = []
    expected_freq = []
    bin_counts = []
    
    for i in range(n_bins):
        # Find points in this bin
        in_bin = (uncertainty_flat >= bin_boundaries[i]) & (uncertainty_flat < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include upper boundary in last bin
            in_bin = (uncertainty_flat >= bin_boundaries[i]) & (uncertainty_flat <= bin_boundaries[i + 1])
        
        bin_count = np.sum(in_bin)
        bin_counts.append(bin_count)
        
        if bin_count > 0:
            # Observed frequency: fraction of predictions within 1-sigma
            bin_uncertainties = uncertainty_flat[in_bin]
            bin_errors = errors_flat[in_bin]
            
            # Count how many errors are within predicted uncertainty
            within_1sigma = bin_errors <= bin_uncertainties
            observed = np.mean(within_1sigma)
            observed_freq.append(observed)
            
            # Expected frequency for normal distribution (≈68.3% for 1-sigma)
            expected_freq.append(0.683)
        else:
            observed_freq.append(0.0)
            expected_freq.append(0.683)
    
    return {
        'bin_boundaries': bin_boundaries,
        'bin_lowers': bin_boundaries[:-1],
        'bin_uppers': bin_boundaries[1:],
        'bin_centers': bin_centers,
        'observed_freq': np.array(observed_freq),
        'expected_freq': np.array(expected_freq),
        'bin_counts': np.array(bin_counts)
    }

def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                             uncertainty: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        uncertainty: Predicted uncertainties
        n_bins: Number of bins
        
    Returns:
        Expected calibration error
    """
    rel_data = reliability_diagram_bins(y_true, y_pred, uncertainty, n_bins)
    
    if len(rel_data['bin_counts']) == 0:
        return np.nan
    
    # Weighted average of calibration gaps
    bin_counts = rel_data['bin_counts']
    total_count = np.sum(bin_counts)
    
    if total_count == 0:
        return np.nan
    
    observed = rel_data['observed_freq']
    expected = rel_data['expected_freq']
    
    # ECE = sum of |observed - expected| weighted by bin size
    ece = np.sum(bin_counts * np.abs(observed - expected)) / total_count
    
    return float(ece)

def maximum_calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                            uncertainty: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        uncertainty: Predicted uncertainties
        n_bins: Number of bins
        
    Returns:
        Maximum calibration error
    """
    rel_data = reliability_diagram_bins(y_true, y_pred, uncertainty, n_bins)
    
    if len(rel_data['bin_counts']) == 0:
        return np.nan
    
    observed = rel_data['observed_freq']
    expected = rel_data['expected_freq']
    
    # MCE = maximum |observed - expected| across all bins
    mce = np.max(np.abs(observed - expected))
    
    return float(mce)

def sharpness_metric(uncertainty: np.ndarray) -> float:
    """
    Compute sharpness (average predicted uncertainty).
    Lower values indicate sharper (more confident) predictions.
    
    Args:
        uncertainty: Predicted uncertainties
        
    Returns:
        Average uncertainty (sharpness)
    """
    return float(np.mean(uncertainty))

def coverage_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    uncertainty: np.ndarray) -> Dict[str, float]:
    """
    Compute coverage metrics for different confidence levels.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        uncertainty: Predicted uncertainties (standard deviations)
        
    Returns:
        Dictionary with coverage metrics
    """
    errors = np.abs(y_true - y_pred)
    
    # Coverage for different sigma levels
    coverage_1sigma = np.mean(errors <= uncertainty)
    coverage_2sigma = np.mean(errors <= 2 * uncertainty)
    coverage_3sigma = np.mean(errors <= 3 * uncertainty)
    
    return {
        'coverage_1sigma': float(coverage_1sigma),
        'coverage_2sigma': float(coverage_2sigma), 
        'coverage_3sigma': float(coverage_3sigma),
        'expected_1sigma': 0.683,
        'expected_2sigma': 0.954,
        'expected_3sigma': 0.997
    }

def interval_score(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                  alpha: float = 0.1) -> float:
    """
    Compute interval score for prediction intervals.
    Lower is better.
    
    Args:
        y_true: Ground truth values
        lower: Lower bounds of prediction intervals
        upper: Upper bounds of prediction intervals
        alpha: Miscoverage level (e.g., 0.1 for 90% intervals)
        
    Returns:
        Average interval score
    """
    width = upper - lower
    
    # Penalty for being below lower bound
    lower_penalty = (2 / alpha) * (lower - y_true) * (y_true < lower)
    
    # Penalty for being above upper bound  
    upper_penalty = (2 / alpha) * (y_true - upper) * (y_true > upper)
    
    # Total score = width + penalties
    scores = width + lower_penalty + upper_penalty
    
    return float(np.mean(scores))

def prediction_interval_coverage_probability(y_true: np.ndarray, 
                                           lower: np.ndarray, 
                                           upper: np.ndarray) -> float:
    """
    Compute empirical coverage probability for prediction intervals.
    
    Args:
        y_true: Ground truth values
        lower: Lower bounds
        upper: Upper bounds
        
    Returns:
        Coverage probability (fraction of points within intervals)
    """
    within_interval = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within_interval))

def compute_all_calibration_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                  uncertainty: np.ndarray, 
                                  lower: Optional[np.ndarray] = None,
                                  upper: Optional[np.ndarray] = None,
                                  n_bins: int = 10) -> Dict[str, float]:
    """
    Compute comprehensive calibration metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        uncertainty: Predicted uncertainties
        lower: Optional lower bounds for intervals
        upper: Optional upper bounds for intervals
        n_bins: Number of bins for reliability diagram
        
    Returns:
        Dictionary with all calibration metrics
    """
    metrics = {}
    
    # Basic coverage metrics
    coverage = coverage_metrics(y_true, y_pred, uncertainty)
    metrics.update(coverage)
    
    # Calibration errors
    metrics['ece'] = expected_calibration_error(y_true, y_pred, uncertainty, n_bins)
    metrics['mce'] = maximum_calibration_error(y_true, y_pred, uncertainty, n_bins)
    
    # Sharpness
    metrics['sharpness'] = sharpness_metric(uncertainty)
    
    # Interval metrics if bounds provided
    if lower is not None and upper is not None:
        metrics['interval_coverage'] = prediction_interval_coverage_probability(y_true, lower, upper)
        metrics['interval_score_90'] = interval_score(y_true, lower, upper, alpha=0.1)
        metrics['interval_score_95'] = interval_score(y_true, lower, upper, alpha=0.05)
        metrics['avg_interval_width'] = float(np.mean(upper - lower))
    
    return metrics

def calibration_slope_intercept(y_true: np.ndarray, y_pred: np.ndarray,
                              uncertainty: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute calibration slope and intercept via linear regression.
    Well-calibrated predictions should have slope ≈ 1, intercept ≈ 0.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        uncertainty: Predicted uncertainties
        
    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    # Use squared errors vs predicted variance for regression
    squared_errors = (y_true - y_pred) ** 2
    predicted_variance = uncertainty ** 2
    
    # Flatten arrays
    squared_errors_flat = squared_errors.flatten()
    predicted_variance_flat = predicted_variance.flatten()
    
    # Remove invalid values
    valid_mask = np.isfinite(squared_errors_flat) & np.isfinite(predicted_variance_flat)
    squared_errors_flat = squared_errors_flat[valid_mask]
    predicted_variance_flat = predicted_variance_flat[valid_mask]
    
    if len(squared_errors_flat) < 2:
        return np.nan, np.nan, np.nan
    
    # Linear regression: squared_error = slope * predicted_variance + intercept
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            predicted_variance_flat, squared_errors_flat
        )
        r_squared = r_value ** 2
        return float(slope), float(intercept), float(r_squared)
    except:
        return np.nan, np.nan, np.nan
