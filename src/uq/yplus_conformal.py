"""
Y+-conditional conformal prediction for turbulence models.
Implements band-wise coverage control as required for the thesis scope.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

class YPlusConformPredictor:
    """Y+-conditional conformal prediction with band-wise coverage control."""
    
    def __init__(self, yplus_bands: List[Tuple[float, float]] = None):
        """
        Initialize Y+-conditional conformal predictor.
        
        Args:
            yplus_bands: List of (y_min, y_max) tuples defining Y+ bands
        """
        if yplus_bands is None:
            # Default bands from thesis scope
            self.yplus_bands = [(0, 30), (30, 100), (100, 370), (370, 1000)]
        else:
            self.yplus_bands = yplus_bands
        
        self.band_quantiles = {}
        self.calibrated = False
    
    def compute_yplus_coordinates(self, grid_shape: Tuple[int, int, int], 
                                 channel_height: float = 2.0) -> np.ndarray:
        """
        Compute Y+ coordinates for channel flow grid.
        
        Args:
            grid_shape: (D, H, W) shape of velocity grid
            channel_height: Channel half-height in wall units
            
        Returns:
            yplus_coords: Y+ coordinates array of shape (D, H, W)
        """
        D, H, W = grid_shape
        
        # Y-coordinate (wall-normal direction, assuming y is the second dimension)
        y_coords = np.linspace(-channel_height, channel_height, H)
        
        # Convert to Y+ (distance from nearest wall)
        yplus_coords = np.minimum(np.abs(y_coords + channel_height), 
                                 np.abs(y_coords - channel_height))
        
        # Broadcast to full grid
        yplus_grid = np.zeros(grid_shape)
        for i in range(D):
            for k in range(W):
                yplus_grid[i, :, k] = yplus_coords
        
        return yplus_grid
    
    def assign_yplus_bands(self, yplus_coords: np.ndarray) -> np.ndarray:
        """
        Assign each grid point to a Y+ band.
        
        Args:
            yplus_coords: Y+ coordinates array
            
        Returns:
            band_indices: Band assignment for each grid point (-1 if outside all bands)
        """
        band_indices = np.full(yplus_coords.shape, -1, dtype=int)
        
        for band_idx, (y_min, y_max) in enumerate(self.yplus_bands):
            mask = (yplus_coords >= y_min) & (yplus_coords < y_max)
            band_indices[mask] = band_idx
        
        return band_indices
    
    def compute_band_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                              yplus_coords: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Compute residuals grouped by Y+ bands.
        
        Args:
            y_true: Ground truth velocity fields (N, C, D, H, W)
            y_pred: Predicted velocity fields (N, C, D, H, W)
            yplus_coords: Y+ coordinates (D, H, W)
            
        Returns:
            band_residuals: Dictionary mapping band_idx -> residuals array
        """
        # Compute absolute residuals per component and voxel
        residuals = np.abs(y_true - y_pred)  # (N, C, D, H, W)
        
        # Get band assignments
        band_indices = self.assign_yplus_bands(yplus_coords)
        
        band_residuals = {}
        
        for band_idx in range(len(self.yplus_bands)):
            # Mask for this band
            band_mask = (band_indices == band_idx)
            
            if np.any(band_mask):
                # Extract residuals for this band across all samples and components
                band_res = residuals[:, :, band_mask]  # (N, C, n_voxels_in_band)
                
                # Flatten to get per-voxel residuals
                band_residuals[band_idx] = band_res.flatten()
            else:
                band_residuals[band_idx] = np.array([])
        
        return band_residuals
    
    def calibrate(self, y_true_cal: np.ndarray, y_pred_cal: np.ndarray,
                  yplus_coords: np.ndarray, alpha: float = 0.1) -> Dict:
        """
        Calibrate conformal quantiles for each Y+ band.
        
        Args:
            y_true_cal: Calibration ground truth (N_cal, C, D, H, W)
            y_pred_cal: Calibration predictions (N_cal, C, D, H, W)
            yplus_coords: Y+ coordinates (D, H, W)
            alpha: Miscoverage level (0.1 for 90% coverage)
            
        Returns:
            calibration_info: Dictionary with calibration statistics
        """
        print(f"Calibrating Y+-conditional conformal prediction (alpha={alpha})...")
        
        # Compute band-wise residuals
        band_residuals = self.compute_band_residuals(y_true_cal, y_pred_cal, yplus_coords)
        
        calibration_info = {
            'alpha': alpha,
            'target_coverage': 1 - alpha,
            'yplus_bands': self.yplus_bands,
            'band_stats': {}
        }
        
        # Compute quantiles for each band
        for band_idx, (y_min, y_max) in enumerate(self.yplus_bands):
            residuals = band_residuals[band_idx]
            
            if len(residuals) > 0:
                # Conformal quantile: ceil((n+1)(1-alpha))/n percentile
                n = len(residuals)
                q_level = np.ceil((n + 1) * (1 - alpha)) / n
                q_level = min(q_level, 1.0)  # Cap at 100th percentile
                
                quantile = np.quantile(residuals, q_level)
                self.band_quantiles[band_idx] = quantile
                
                # Store statistics
                calibration_info['band_stats'][band_idx] = {
                    'yplus_range': [y_min, y_max],
                    'n_residuals': len(residuals),
                    'quantile': float(quantile),
                    'q_level': float(q_level),
                    'residual_mean': float(np.mean(residuals)),
                    'residual_std': float(np.std(residuals))
                }
                
                print(f"  Band {band_idx} Y+[{y_min}, {y_max}): "
                      f"n={len(residuals)}, q={quantile:.6f}")
            else:
                self.band_quantiles[band_idx] = 0.0
                calibration_info['band_stats'][band_idx] = {
                    'yplus_range': [y_min, y_max],
                    'n_residuals': 0,
                    'quantile': 0.0,
                    'q_level': 0.0,
                    'residual_mean': 0.0,
                    'residual_std': 0.0
                }
                print(f"  Band {band_idx} Y+[{y_min}, {y_max}): No data")
        
        self.calibrated = True
        return calibration_info
    
    def predict_intervals(self, y_pred: np.ndarray, yplus_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate prediction intervals using calibrated quantiles.
        
        Args:
            y_pred: Predicted velocity fields (N, C, D, H, W)
            yplus_coords: Y+ coordinates (D, H, W)
            
        Returns:
            lower_bounds: Lower prediction intervals (N, C, D, H, W)
            upper_bounds: Upper prediction intervals (N, C, D, H, W)
        """
        if not self.calibrated:
            raise ValueError("Must calibrate before generating prediction intervals")
        
        # Initialize bounds
        lower_bounds = np.zeros_like(y_pred)
        upper_bounds = np.zeros_like(y_pred)
        
        # Get band assignments
        band_indices = self.assign_yplus_bands(yplus_coords)
        
        # Apply band-specific quantiles
        for band_idx in range(len(self.yplus_bands)):
            band_mask = (band_indices == band_idx)
            
            if np.any(band_mask) and band_idx in self.band_quantiles:
                quantile = self.band_quantiles[band_idx]
                
                # Apply to all samples and components
                lower_bounds[:, :, band_mask] = y_pred[:, :, band_mask] - quantile
                upper_bounds[:, :, band_mask] = y_pred[:, :, band_mask] + quantile
        
        return lower_bounds, upper_bounds
    
    def evaluate_coverage(self, y_true: np.ndarray, lower_bounds: np.ndarray,
                         upper_bounds: np.ndarray, yplus_coords: np.ndarray) -> Dict:
        """
        Evaluate coverage metrics by Y+ band.
        
        Args:
            y_true: Ground truth (N, C, D, H, W)
            lower_bounds: Lower prediction intervals (N, C, D, H, W)
            upper_bounds: Upper prediction intervals (N, C, D, H, W)
            yplus_coords: Y+ coordinates (D, H, W)
            
        Returns:
            coverage_metrics: Dictionary with band-wise coverage metrics
        """
        # Check coverage
        within_interval = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        interval_width = upper_bounds - lower_bounds
        
        # Get band assignments
        band_indices = self.assign_yplus_bands(yplus_coords)
        
        coverage_metrics = {
            'overall': {
                'picp': float(np.mean(within_interval)),
                'mpiw': float(np.mean(interval_width))
            },
            'by_band': {}
        }
        
        # Compute band-wise metrics
        for band_idx, (y_min, y_max) in enumerate(self.yplus_bands):
            band_mask = (band_indices == band_idx)
            
            if np.any(band_mask):
                band_coverage = within_interval[:, :, band_mask]
                band_width = interval_width[:, :, band_mask]
                
                picp = float(np.mean(band_coverage))
                mpiw = float(np.mean(band_width))
                
                coverage_metrics['by_band'][band_idx] = {
                    'yplus_range': [y_min, y_max],
                    'picp': picp,
                    'mpiw': mpiw,
                    'n_voxels': int(np.sum(band_mask)),
                    'target_coverage': 1 - self.alpha if hasattr(self, 'alpha') else 0.9
                }
            else:
                coverage_metrics['by_band'][band_idx] = {
                    'yplus_range': [y_min, y_max],
                    'picp': 0.0,
                    'mpiw': 0.0,
                    'n_voxels': 0,
                    'target_coverage': 1 - self.alpha if hasattr(self, 'alpha') else 0.9
                }
        
        return coverage_metrics
    
    def save_calibration(self, save_path: str):
        """Save calibration information to file."""
        if not self.calibrated:
            raise ValueError("No calibration to save")
        
        calibration_data = {
            'yplus_bands': self.yplus_bands,
            'band_quantiles': {str(k): float(v) for k, v in self.band_quantiles.items()},
            'calibrated': True
        }
        
        with open(save_path, 'w') as f:
            json.dump(calibration_data, f, indent=2)
    
    def load_calibration(self, load_path: str):
        """Load calibration information from file."""
        with open(load_path, 'r') as f:
            calibration_data = json.load(f)
        
        self.yplus_bands = calibration_data['yplus_bands']
        self.band_quantiles = {int(k): float(v) for k, v in calibration_data['band_quantiles'].items()}
        self.calibrated = calibration_data['calibrated']
