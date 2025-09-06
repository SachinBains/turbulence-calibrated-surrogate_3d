"""
Germano dynamic procedure for effective Smagorinsky coefficient recovery.
Implements the law-like coefficient C_s^eff(y+) extraction as required for thesis scope.
"""
import numpy as np
from scipy import ndimage
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from pathlib import Path

class GermanoCoeffRecovery:
    """Germano dynamic procedure for effective Smagorinsky coefficient recovery."""
    
    def __init__(self, delta_base: float = 1.0, test_multiplier: float = 2.0):
        """
        Initialize Germano coefficient recovery.
        
        Args:
            delta_base: Base filter width (grid scale)
            test_multiplier: Test filter scale multiplier (typically 2.0)
        """
        self.delta_base = delta_base
        self.delta_test = delta_base * test_multiplier
        self.test_multiplier = test_multiplier
    
    def apply_filter_3d(self, field: np.ndarray, filter_width: float, 
                       filter_type: str = 'box') -> np.ndarray:
        """
        Apply 3D spatial filter to velocity field.
        
        Args:
            field: Input field (D, H, W)
            filter_width: Filter width in grid units
            filter_type: 'box' or 'gaussian'
            
        Returns:
            filtered_field: Filtered field
        """
        if filter_type == 'box':
            # Box filter using uniform convolution
            kernel_size = max(1, int(np.round(filter_width)))
            kernel = np.ones((kernel_size, kernel_size, kernel_size))
            kernel = kernel / np.sum(kernel)
            
            filtered = ndimage.convolve(field, kernel, mode='constant', cval=0.0)
            
        elif filter_type == 'gaussian':
            # Gaussian filter
            sigma = filter_width / 2.355  # FWHM to sigma conversion
            filtered = ndimage.gaussian_filter(field, sigma=sigma, mode='constant', cval=0.0)
            
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")
        
        return filtered
    
    def compute_strain_rate_tensor(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                                  dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute strain rate tensor and its magnitude.
        
        Args:
            u, v, w: Velocity components (D, H, W)
            dx: Grid spacing
            
        Returns:
            strain_components: Dictionary of strain rate tensor components
            strain_magnitude: Strain rate magnitude |S|
        """
        # Compute velocity gradients
        du_dx, du_dy, du_dz = np.gradient(u, dx)
        dv_dx, dv_dy, dv_dz = np.gradient(v, dx)
        dw_dx, dw_dy, dw_dz = np.gradient(w, dx)
        
        # Strain rate tensor components S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
        S11 = du_dx
        S22 = dv_dy
        S33 = dw_dz
        S12 = 0.5 * (du_dy + dv_dx)
        S13 = 0.5 * (du_dz + dw_dx)
        S23 = 0.5 * (dv_dz + dw_dy)
        
        # Strain rate magnitude |S| = sqrt(2 * S_ij * S_ij)
        strain_magnitude = np.sqrt(2 * (S11**2 + S22**2 + S33**2 + 
                                       2 * (S12**2 + S13**2 + S23**2)))
        
        strain_components = {
            'S11': S11, 'S22': S22, 'S33': S33,
            'S12': S12, 'S13': S13, 'S23': S23
        }
        
        return strain_components, strain_magnitude
    
    def compute_leonard_stress(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                              filter_type: str = 'box') -> Dict[str, np.ndarray]:
        """
        Compute Leonard stress tensor L_ij = û_i û_j - û_i * û_j.
        
        Args:
            u, v, w: Velocity components (D, H, W)
            filter_type: Filter type for test filter
            
        Returns:
            leonard_stress: Dictionary of Leonard stress components
        """
        # Apply test filter to velocity components
        u_test = self.apply_filter_3d(u, self.delta_test, filter_type)
        v_test = self.apply_filter_3d(v, self.delta_test, filter_type)
        w_test = self.apply_filter_3d(w, self.delta_test, filter_type)
        
        # Compute products at base scale, then filter
        uu_filtered = self.apply_filter_3d(u * u, self.delta_test, filter_type)
        uv_filtered = self.apply_filter_3d(u * v, self.delta_test, filter_type)
        uw_filtered = self.apply_filter_3d(u * w, self.delta_test, filter_type)
        vv_filtered = self.apply_filter_3d(v * v, self.delta_test, filter_type)
        vw_filtered = self.apply_filter_3d(v * w, self.delta_test, filter_type)
        ww_filtered = self.apply_filter_3d(w * w, self.delta_test, filter_type)
        
        # Leonard stress: L_ij = û_i û_j - û_i * û_j
        leonard_stress = {
            'L11': uu_filtered - u_test * u_test,
            'L22': vv_filtered - v_test * v_test,
            'L33': ww_filtered - w_test * w_test,
            'L12': uv_filtered - u_test * v_test,
            'L13': uw_filtered - u_test * w_test,
            'L23': vw_filtered - v_test * w_test
        }
        
        return leonard_stress
    
    def compute_model_tensor_m(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                              filter_type: str = 'box', dx: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Compute model tensor M_ij for Germano identity.
        
        Args:
            u, v, w: Velocity components (D, H, W)
            filter_type: Filter type
            dx: Grid spacing
            
        Returns:
            model_tensor: Dictionary of M_ij components
        """
        # Strain rate at base scale
        strain_base, strain_mag_base = self.compute_strain_rate_tensor(u, v, w, dx)
        
        # Apply base filter to get grid-scale fields
        u_base = self.apply_filter_3d(u, self.delta_base, filter_type)
        v_base = self.apply_filter_3d(v, self.delta_base, filter_type)
        w_base = self.apply_filter_3d(w, self.delta_base, filter_type)
        
        # Strain rate at grid scale
        strain_grid, strain_mag_grid = self.compute_strain_rate_tensor(u_base, v_base, w_base, dx)
        
        # Apply test filter to grid-scale strain
        u_test = self.apply_filter_3d(u_base, self.delta_test, filter_type)
        v_test = self.apply_filter_3d(v_base, self.delta_test, filter_type)
        w_test = self.apply_filter_3d(w_base, self.delta_test, filter_type)
        
        # Strain rate at test scale
        strain_test, strain_mag_test = self.compute_strain_rate_tensor(u_test, v_test, w_test, dx)
        
        # Model tensor: M_ij = Δ̂²|Ŝ|Ŝ_ij - Δ²|S̄|S̄_ij (filtered to test scale)
        # Simplified: M_ij ≈ (Δ̂²|Ŝ| - Δ²|S̄|) * Ŝ_ij
        
        delta_base_sq = self.delta_base**2
        delta_test_sq = self.delta_test**2
        
        # Filter grid-scale terms to test scale
        strain_mag_grid_test = self.apply_filter_3d(strain_mag_grid, self.delta_test, filter_type)
        
        for comp in strain_grid:
            strain_grid[comp] = self.apply_filter_3d(strain_grid[comp], self.delta_test, filter_type)
        
        # Compute M_ij components
        model_tensor = {}
        for comp in ['S11', 'S22', 'S33', 'S12', 'S13', 'S23']:
            # M_ij = Δ̂²|Ŝ|Ŝ_ij - Δ²|S̄|S̄_ij
            model_tensor[f'M{comp[1:]}'] = (delta_test_sq * strain_mag_test * strain_test[comp] - 
                                           delta_base_sq * strain_mag_grid_test * strain_grid[comp])
        
        return model_tensor
    
    def compute_dynamic_coefficient(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                                   filter_type: str = 'box', dx: float = 1.0,
                                   averaging: str = 'local3') -> np.ndarray:
        """
        Compute dynamic Smagorinsky coefficient using Germano identity.
        
        Args:
            u, v, w: Velocity components (D, H, W)
            filter_type: Filter type ('box' or 'gaussian')
            dx: Grid spacing
            averaging: Averaging method ('local3', 'plane', 'none')
            
        Returns:
            cs_squared: Dynamic coefficient C_s² field
        """
        # Compute Leonard stress L_ij
        leonard_stress = self.compute_leonard_stress(u, v, w, filter_type)
        
        # Compute model tensor M_ij
        model_tensor = self.compute_model_tensor_m(u, v, w, filter_type, dx)
        
        # Germano identity: L_ij - τ̂_ij ≈ 2 C_s² M_ij
        # Solve: C_s² = <L_ij M_ij> / (2 <M_ij M_ij>)
        
        # Compute numerator: L_ij * M_ij (contracted)
        numerator = np.zeros_like(u)
        denominator = np.zeros_like(u)
        
        stress_comps = ['11', '22', '33', '12', '13', '23']
        
        for comp in stress_comps:
            L_comp = leonard_stress[f'L{comp}']
            M_comp = model_tensor[f'M{comp}']
            
            numerator += L_comp * M_comp
            denominator += M_comp * M_comp
        
        # Apply averaging to stabilize the coefficient
        if averaging == 'local3':
            # Local 3x3x3 averaging
            kernel = np.ones((3, 3, 3)) / 27
            numerator = ndimage.convolve(numerator, kernel, mode='constant')
            denominator = ndimage.convolve(denominator, kernel, mode='constant')
            
        elif averaging == 'plane':
            # Plane averaging (average over x-z planes)
            numerator = np.mean(numerator, axis=(0, 2), keepdims=True)
            denominator = np.mean(denominator, axis=(0, 2), keepdims=True)
            
            # Broadcast back to full shape
            numerator = np.broadcast_to(numerator, u.shape)
            denominator = np.broadcast_to(denominator, u.shape)
        
        # Compute C_s²
        cs_squared = np.zeros_like(u)
        valid_mask = np.abs(denominator) > 1e-12
        
        cs_squared[valid_mask] = numerator[valid_mask] / (2 * denominator[valid_mask])
        
        # Clip negative values (non-physical)
        cs_squared = np.maximum(cs_squared, 0.0)
        
        return cs_squared
    
    def compute_yplus_coordinates(self, grid_shape: Tuple[int, int, int], 
                                 channel_height: float = 2.0) -> np.ndarray:
        """
        Compute Y+ coordinates for channel flow.
        
        Args:
            grid_shape: (D, H, W) shape of velocity grid
            channel_height: Channel half-height in wall units
            
        Returns:
            yplus_coords: Y+ coordinates array
        """
        D, H, W = grid_shape
        
        # Y-coordinate (wall-normal direction)
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
    
    def aggregate_coefficient_by_yplus(self, cs_squared: np.ndarray, yplus_coords: np.ndarray,
                                      yplus_bands: List[Tuple[float, float]] = None) -> Dict:
        """
        Aggregate effective coefficient by Y+ bands.
        
        Args:
            cs_squared: Dynamic coefficient field (D, H, W)
            yplus_coords: Y+ coordinates (D, H, W)
            yplus_bands: Y+ band definitions
            
        Returns:
            aggregated_results: Dictionary with band-wise statistics
        """
        if yplus_bands is None:
            yplus_bands = [(0, 30), (30, 100), (100, 370), (370, 1000)]
        
        results = {
            'yplus_bands': yplus_bands,
            'band_statistics': {},
            'profiles': {
                'yplus_centers': [],
                'cs_eff_mean': [],
                'cs_eff_std': [],
                'cs_eff_median': []
            }
        }
        
        for band_idx, (y_min, y_max) in enumerate(yplus_bands):
            # Find points in this band
            band_mask = (yplus_coords >= y_min) & (yplus_coords < y_max)
            
            if np.any(band_mask):
                band_coeffs = cs_squared[band_mask]
                
                # Remove outliers (clip at 99th percentile)
                p99 = np.percentile(band_coeffs, 99)
                band_coeffs_clipped = band_coeffs[band_coeffs <= p99]
                
                stats = {
                    'yplus_range': [y_min, y_max],
                    'yplus_center': (y_min + y_max) / 2,
                    'n_points': len(band_coeffs),
                    'n_points_clipped': len(band_coeffs_clipped),
                    'cs_eff_mean': float(np.mean(band_coeffs_clipped)),
                    'cs_eff_std': float(np.std(band_coeffs_clipped)),
                    'cs_eff_median': float(np.median(band_coeffs_clipped)),
                    'cs_eff_min': float(np.min(band_coeffs_clipped)),
                    'cs_eff_max': float(np.max(band_coeffs_clipped)),
                    'cs_eff_p25': float(np.percentile(band_coeffs_clipped, 25)),
                    'cs_eff_p75': float(np.percentile(band_coeffs_clipped, 75))
                }
                
                results['band_statistics'][band_idx] = stats
                
                # Add to profiles
                results['profiles']['yplus_centers'].append(stats['yplus_center'])
                results['profiles']['cs_eff_mean'].append(stats['cs_eff_mean'])
                results['profiles']['cs_eff_std'].append(stats['cs_eff_std'])
                results['profiles']['cs_eff_median'].append(stats['cs_eff_median'])
            
            else:
                # Empty band
                results['band_statistics'][band_idx] = {
                    'yplus_range': [y_min, y_max],
                    'yplus_center': (y_min + y_max) / 2,
                    'n_points': 0,
                    'cs_eff_mean': 0.0,
                    'cs_eff_std': 0.0,
                    'cs_eff_median': 0.0
                }
        
        return results
    
    def analyze_coefficient_reliability(self, cs_squared: np.ndarray, 
                                      prediction_error: np.ndarray,
                                      prediction_uncertainty: np.ndarray = None,
                                      yplus_coords: np.ndarray = None) -> Dict:
        """
        Analyze correlation between coefficient drift and prediction error/uncertainty.
        
        Args:
            cs_squared: Dynamic coefficient field (D, H, W)
            prediction_error: Prediction error field (D, H, W)
            prediction_uncertainty: Prediction uncertainty field (D, H, W)
            yplus_coords: Y+ coordinates for band-wise analysis
            
        Returns:
            reliability_analysis: Dictionary with correlation statistics
        """
        # Compute coefficient drift from median
        cs_median = np.median(cs_squared)
        coeff_drift = np.abs(cs_squared - cs_median)
        
        # Flatten for correlation analysis
        coeff_drift_flat = coeff_drift.flatten()
        error_flat = np.abs(prediction_error.flatten())
        
        # Error-coefficient correlation
        error_corr = np.corrcoef(coeff_drift_flat, error_flat)[0, 1]
        
        results = {
            'coefficient_median': float(cs_median),
            'coefficient_drift_std': float(np.std(coeff_drift)),
            'error_coefficient_correlation': float(error_corr) if not np.isnan(error_corr) else 0.0
        }
        
        # Uncertainty correlation if available
        if prediction_uncertainty is not None:
            uncertainty_flat = prediction_uncertainty.flatten()
            uncertainty_corr = np.corrcoef(coeff_drift_flat, uncertainty_flat)[0, 1]
            results['uncertainty_coefficient_correlation'] = float(uncertainty_corr) if not np.isnan(uncertainty_corr) else 0.0
        
        # Band-wise analysis if Y+ coordinates provided
        if yplus_coords is not None:
            yplus_bands = [(0, 30), (30, 100), (100, 370), (370, 1000)]
            band_correlations = {}
            
            for band_idx, (y_min, y_max) in enumerate(yplus_bands):
                band_mask = (yplus_coords >= y_min) & (yplus_coords < y_max)
                
                if np.any(band_mask):
                    band_coeff_drift = coeff_drift[band_mask].flatten()
                    band_error = np.abs(prediction_error[band_mask]).flatten()
                    
                    if len(band_coeff_drift) > 1:
                        band_corr = np.corrcoef(band_coeff_drift, band_error)[0, 1]
                        band_correlations[band_idx] = {
                            'yplus_range': [y_min, y_max],
                            'correlation': float(band_corr) if not np.isnan(band_corr) else 0.0,
                            'n_points': len(band_coeff_drift)
                        }
            
            results['band_correlations'] = band_correlations
        
        return results
    
    def plot_coefficient_profiles(self, aggregated_results: Dict, 
                                 save_path: Optional[str] = None,
                                 re_tau_label: str = "Re_τ=1000") -> None:
        """
        Plot effective Smagorinsky coefficient profiles.
        
        Args:
            aggregated_results: Results from aggregate_coefficient_by_yplus
            save_path: Path to save figure
            re_tau_label: Reynolds number label for plot
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        profiles = aggregated_results['profiles']
        yplus = np.array(profiles['yplus_centers'])
        cs_mean = np.array(profiles['cs_eff_mean'])
        cs_std = np.array(profiles['cs_eff_std'])
        cs_median = np.array(profiles['cs_eff_median'])
        
        # Plot 1: Mean profile with error bars
        ax1.errorbar(yplus, cs_mean, yerr=cs_std, marker='o', capsize=5, 
                    label=f'{re_tau_label} (mean ± std)')
        ax1.plot(yplus, cs_median, 's--', label=f'{re_tau_label} (median)')
        
        ax1.set_xlabel('Y⁺')
        ax1.set_ylabel('C_s^eff')
        ax1.set_title('Effective Smagorinsky Coefficient Profile')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Box plot by band
        band_data = []
        band_labels = []
        
        for band_idx in sorted(aggregated_results['band_statistics'].keys()):
            stats = aggregated_results['band_statistics'][band_idx]
            if stats['n_points'] > 0:
                # Create box plot data from percentiles
                band_data.append([
                    stats['cs_eff_min'],
                    stats['cs_eff_p25'],
                    stats['cs_eff_median'],
                    stats['cs_eff_p75'],
                    stats['cs_eff_max']
                ])
                y_min, y_max = stats['yplus_range']
                band_labels.append(f'[{y_min}, {y_max})')
        
        if band_data:
            bp = ax2.boxplot(band_data, labels=band_labels, patch_artist=True)
            ax2.set_xlabel('Y⁺ Band')
            ax2.set_ylabel('C_s^eff')
            ax2.set_title('Coefficient Distribution by Y⁺ Band')
            ax2.tick_params(axis='x', rotation=45)
            
            # Color boxes
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
                patch.set_alpha(0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
