"""
Physics admissibility gates for turbulence predictions.
Implements strict pass/fail criteria as required for thesis scope.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import signal
from scipy.fft import fftn, fftfreq
import matplotlib.pyplot as plt

class PhysicsGateValidator:
    """Physics admissibility validator with strict pass/fail gates."""
    
    def __init__(self, thresholds: Dict = None):
        """
        Initialize physics gate validator.
        
        Args:
            thresholds: Dictionary of threshold values for pass/fail criteria
        """
        # Default thresholds from thesis scope
        if thresholds is None:
            self.thresholds = {
                'divergence_max_ratio': 1.2,  # ||∇·u||₂ ≤ 1.2× DNS discretization error
                'spectra_deviation_max': 0.10,  # ±10% deviation in premultiplied spectra
                'wall_law_kappa_min': 0.35,   # von Kármán constant bounds
                'wall_law_kappa_max': 0.45,
                'wall_law_B_min': 4.0,        # Additive constant bounds  
                'wall_law_B_max': 6.0,
                'wall_law_max_deviation': 0.15  # Max deviation from canonical wall law
            }
        else:
            self.thresholds = thresholds
    
    def validate_incompressibility(self, velocity_field: np.ndarray, 
                                  reference_divergence: Optional[float] = None,
                                  dx: float = 1.0) -> Dict:
        """
        Validate incompressibility constraint: ∇·u ≈ 0.
        
        Args:
            velocity_field: Velocity field (3, D, H, W) or (N, 3, D, H, W)
            reference_divergence: Reference DNS discretization error
            dx: Grid spacing
            
        Returns:
            incompressibility_results: Pass/fail status and metrics
        """
        # Handle batch dimension
        if velocity_field.ndim == 5:
            # Batch processing
            batch_results = []
            for i in range(velocity_field.shape[0]):
                result = self.validate_incompressibility(velocity_field[i], reference_divergence, dx)
                batch_results.append(result)
            
            # Aggregate results
            divergence_norms = [r['divergence_norm'] for r in batch_results]
            pass_flags = [r['pass'] for r in batch_results]
            
            return {
                'divergence_norm': np.mean(divergence_norms),
                'divergence_norm_std': np.std(divergence_norms),
                'divergence_norms_all': divergence_norms,
                'pass_rate': np.mean(pass_flags),
                'pass': np.mean(pass_flags) > 0.95,  # 95% of samples must pass
                'n_samples': len(batch_results)
            }
        
        # Single sample processing
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        
        # Compute divergence: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
        du_dx = np.gradient(u, dx, axis=0)
        dv_dy = np.gradient(v, dx, axis=1) 
        dw_dz = np.gradient(w, dx, axis=2)
        
        divergence = du_dx + dv_dy + dw_dz
        
        # L2 norm of divergence
        divergence_norm = np.sqrt(np.mean(divergence**2))
        
        # Pass/fail criterion
        if reference_divergence is not None:
            threshold = self.thresholds['divergence_max_ratio'] * reference_divergence
            pass_flag = divergence_norm <= threshold
        else:
            # Use absolute threshold if no reference available
            threshold = 1e-3  # Typical discretization error scale
            pass_flag = divergence_norm <= threshold
        
        return {
            'divergence_norm': float(divergence_norm),
            'threshold': float(threshold) if reference_divergence else 1e-3,
            'pass': bool(pass_flag),
            'divergence_field': divergence,
            'reference_divergence': reference_divergence
        }
    
    def validate_energy_spectrum(self, velocity_field: np.ndarray,
                                reference_spectrum: Optional[np.ndarray] = None,
                                dx: float = 1.0) -> Dict:
        """
        Validate energy spectrum and Kolmogorov scaling.
        
        Args:
            velocity_field: Velocity field (3, D, H, W)
            reference_spectrum: Reference DNS spectrum for comparison
            dx: Grid spacing
            
        Returns:
            spectrum_results: Pass/fail status and spectral metrics
        """
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        
        # Compute 3D energy spectrum
        u_fft = fftn(u)
        v_fft = fftn(v) 
        w_fft = fftn(w)
        
        energy_spectrum = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2 + np.abs(w_fft)**2)
        
        # Wavenumber grid
        kx = fftfreq(u.shape[0], d=dx)
        ky = fftfreq(u.shape[1], d=dx)
        kz = fftfreq(u.shape[2], d=dx)
        
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        k_magnitude = np.sqrt(KX**2 + KY**2 + KZ**2)
        
        # Radially averaged spectrum
        k_max = np.max(k_magnitude)
        k_bins = np.linspace(0, k_max, 50)
        k_centers = (k_bins[:-1] + k_bins[1:]) / 2
        radial_spectrum = np.zeros(len(k_centers))
        
        for i, k_center in enumerate(k_centers):
            mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i+1])
            if np.any(mask):
                radial_spectrum[i] = np.mean(energy_spectrum[mask])
        
        # Premultiplied spectrum: k * E(k)
        premult_spectrum = k_centers * radial_spectrum
        
        results = {
            'wavenumbers': k_centers,
            'energy_spectrum': radial_spectrum,
            'premultiplied_spectrum': premult_spectrum,
            'total_energy': np.sum(radial_spectrum)
        }
        
        # Validate against reference if available
        if reference_spectrum is not None:
            # Compute relative deviation in resolved range
            resolved_range = (k_centers > 1) & (k_centers < k_max/3) & (radial_spectrum > 0) & (reference_spectrum > 0)
            
            if np.any(resolved_range):
                relative_error = np.abs(radial_spectrum[resolved_range] - reference_spectrum[resolved_range]) / reference_spectrum[resolved_range]
                max_deviation = np.max(relative_error)
                mean_deviation = np.mean(relative_error)
                
                pass_flag = max_deviation <= self.thresholds['spectra_deviation_max']
                
                results.update({
                    'reference_spectrum': reference_spectrum,
                    'max_spectral_deviation': float(max_deviation),
                    'mean_spectral_deviation': float(mean_deviation),
                    'pass': bool(pass_flag),
                    'threshold': self.thresholds['spectra_deviation_max']
                })
            else:
                results.update({
                    'max_spectral_deviation': 1.0,
                    'mean_spectral_deviation': 1.0,
                    'pass': False,
                    'threshold': self.thresholds['spectra_deviation_max']
                })
        
        # Validate Kolmogorov -5/3 scaling in inertial range
        inertial_range = (k_centers > 2) & (k_centers < k_max/4) & (radial_spectrum > 0)
        
        if np.sum(inertial_range) > 3:
            log_k = np.log(k_centers[inertial_range])
            log_E = np.log(radial_spectrum[inertial_range])
            
            # Linear fit in log space
            slope, intercept = np.polyfit(log_k, log_E, 1)
            kolmogorov_deviation = abs(slope + 5/3)
            
            results.update({
                'inertial_slope': float(slope),
                'kolmogorov_deviation': float(kolmogorov_deviation),
                'kolmogorov_pass': kolmogorov_deviation < 0.2  # Allow 20% deviation
            })
        
        return results
    
    def validate_wall_law(self, velocity_field: np.ndarray, 
                         yplus_coords: np.ndarray,
                         wall_normal_dir: int = 1) -> Dict:
        """
        Validate wall law: U⁺(y⁺) in viscous and log regions.
        
        Args:
            velocity_field: Velocity field (3, D, H, W)
            yplus_coords: Y⁺ coordinates (D, H, W)
            wall_normal_dir: Wall-normal direction (0=x, 1=y, 2=z)
            
        Returns:
            wall_law_results: Pass/fail status and wall law metrics
        """
        if velocity_field.shape[0] != 3:
            raise ValueError("Expected 3-component velocity field")
        
        # Extract streamwise velocity (assume u is streamwise)
        u_streamwise = velocity_field[0]
        
        # Average over homogeneous directions (x and z for channel flow)
        if wall_normal_dir == 1:  # y is wall-normal
            u_mean = np.mean(u_streamwise, axis=(0, 2))  # Average over x and z
            yplus_mean = np.mean(yplus_coords, axis=(0, 2))
        else:
            raise NotImplementedError("Only y-direction wall-normal currently supported")
        
        # Convert to wall units (U⁺ = u/u_τ, assuming u_τ = 1 for normalized data)
        u_plus = u_mean
        
        # Define regions
        viscous_region = yplus_mean <= 5
        buffer_region = (yplus_mean > 5) & (yplus_mean <= 30)
        log_region = (yplus_mean > 30) & (yplus_mean <= 300)
        
        results = {
            'yplus': yplus_mean,
            'u_plus': u_plus
        }
        
        # Validate viscous sublayer: U⁺ ≈ y⁺
        if np.any(viscous_region):
            yplus_visc = yplus_mean[viscous_region]
            uplus_visc = u_plus[viscous_region]
            
            # Linear fit: U⁺ = a * y⁺
            if len(yplus_visc) > 1:
                slope_visc = np.polyfit(yplus_visc, uplus_visc, 1)[0]
                viscous_deviation = abs(slope_visc - 1.0)
                
                results.update({
                    'viscous_slope': float(slope_visc),
                    'viscous_deviation': float(viscous_deviation),
                    'viscous_pass': viscous_deviation < 0.1  # 10% tolerance
                })
        
        # Validate log region: U⁺ = (1/κ) ln(y⁺) + B
        if np.any(log_region):
            yplus_log = yplus_mean[log_region]
            uplus_log = u_plus[log_region]
            
            if len(yplus_log) > 3:
                # Fit: U⁺ = A ln(y⁺) + B, where A = 1/κ
                log_yplus = np.log(yplus_log)
                coeffs = np.polyfit(log_yplus, uplus_log, 1)
                A, B = coeffs[0], coeffs[1]
                
                kappa = 1.0 / A if A > 0 else 0.0
                
                # Check if κ and B are within canonical bounds
                kappa_pass = (self.thresholds['wall_law_kappa_min'] <= kappa <= 
                             self.thresholds['wall_law_kappa_max'])
                B_pass = (self.thresholds['wall_law_B_min'] <= B <= 
                         self.thresholds['wall_law_B_max'])
                
                # Compute fit quality
                u_plus_fit = A * log_yplus + B
                max_deviation = np.max(np.abs(uplus_log - u_plus_fit))
                
                deviation_pass = max_deviation <= self.thresholds['wall_law_max_deviation']
                
                results.update({
                    'log_slope_A': float(A),
                    'log_intercept_B': float(B),
                    'kappa': float(kappa),
                    'kappa_pass': bool(kappa_pass),
                    'B_pass': bool(B_pass),
                    'max_log_deviation': float(max_deviation),
                    'deviation_pass': bool(deviation_pass),
                    'log_law_pass': bool(kappa_pass and B_pass and deviation_pass)
                })
        
        # Overall wall law pass
        wall_law_pass = results.get('viscous_pass', True) and results.get('log_law_pass', True)
        results['wall_law_pass'] = wall_law_pass
        
        return results
    
    def comprehensive_physics_validation(self, velocity_field: np.ndarray,
                                       yplus_coords: np.ndarray = None,
                                       reference_data: Dict = None,
                                       dx: float = 1.0) -> Dict:
        """
        Run comprehensive physics validation with pass/fail gates.
        
        Args:
            velocity_field: Velocity field (3, D, H, W) or (N, 3, D, H, W)
            yplus_coords: Y⁺ coordinates for wall law validation
            reference_data: Reference DNS data for comparison
            dx: Grid spacing
            
        Returns:
            validation_results: Comprehensive pass/fail results
        """
        print("Running comprehensive physics validation...")
        
        results = {
            'overall_pass': True,
            'failed_gates': [],
            'validation_summary': {}
        }
        
        # 1. Incompressibility validation
        print("  - Validating incompressibility...")
        ref_divergence = reference_data.get('divergence_norm') if reference_data else None
        incomp_results = self.validate_incompressibility(velocity_field, ref_divergence, dx)
        results['incompressibility'] = incomp_results
        
        if not incomp_results['pass']:
            results['overall_pass'] = False
            results['failed_gates'].append('incompressibility')
        
        # 2. Energy spectrum validation
        print("  - Validating energy spectrum...")
        ref_spectrum = reference_data.get('energy_spectrum') if reference_data else None
        spectrum_results = self.validate_energy_spectrum(velocity_field, ref_spectrum, dx)
        results['energy_spectrum'] = spectrum_results
        
        if 'pass' in spectrum_results and not spectrum_results['pass']:
            results['overall_pass'] = False
            results['failed_gates'].append('energy_spectrum')
        
        # 3. Wall law validation (if Y⁺ coordinates provided)
        if yplus_coords is not None:
            print("  - Validating wall law...")
            wall_results = self.validate_wall_law(velocity_field, yplus_coords)
            results['wall_law'] = wall_results
            
            if not wall_results.get('wall_law_pass', False):
                results['overall_pass'] = False
                results['failed_gates'].append('wall_law')
        
        # Create validation summary
        results['validation_summary'] = {
            'total_gates': len([k for k in results.keys() if k not in ['overall_pass', 'failed_gates', 'validation_summary']]),
            'passed_gates': len([k for k in results.keys() if k not in ['overall_pass', 'failed_gates', 'validation_summary']]) - len(results['failed_gates']),
            'failed_gates': results['failed_gates'],
            'pass_rate': 1.0 - len(results['failed_gates']) / max(1, len([k for k in results.keys() if k not in ['overall_pass', 'failed_gates', 'validation_summary']]))
        }
        
        print(f"  Physics validation complete: {results['validation_summary']['passed_gates']}/{results['validation_summary']['total_gates']} gates passed")
        
        return results
    
    def generate_physics_report(self, validation_results: Dict) -> str:
        """
        Generate physics validation report with pass/fail assessment.
        
        Args:
            validation_results: Results from comprehensive_physics_validation
            
        Returns:
            report: Formatted physics validation report
        """
        report = "# Physics Admissibility Report\n\n"
        
        # Overall assessment
        overall_status = "PASS" if validation_results['overall_pass'] else "FAIL"
        report += f"## Overall Status: {overall_status}\n\n"
        
        summary = validation_results['validation_summary']
        report += f"- Gates Passed: {summary['passed_gates']}/{summary['total_gates']}\n"
        report += f"- Pass Rate: {summary['pass_rate']:.1%}\n"
        
        if summary['failed_gates']:
            report += f"- Failed Gates: {', '.join(summary['failed_gates'])}\n"
        
        report += "\n"
        
        # Detailed results
        report += "## Detailed Results\n\n"
        
        # Incompressibility
        if 'incompressibility' in validation_results:
            incomp = validation_results['incompressibility']
            status = "PASS" if incomp['pass'] else "FAIL"
            report += f"### Incompressibility: {status}\n"
            report += f"- Divergence L2 norm: {incomp['divergence_norm']:.6f}\n"
            report += f"- Threshold: {incomp['threshold']:.6f}\n"
            if 'pass_rate' in incomp:
                report += f"- Sample pass rate: {incomp['pass_rate']:.1%}\n"
            report += "\n"
        
        # Energy spectrum
        if 'energy_spectrum' in validation_results:
            spectrum = validation_results['energy_spectrum']
            if 'pass' in spectrum:
                status = "PASS" if spectrum['pass'] else "FAIL"
                report += f"### Energy Spectrum: {status}\n"
                report += f"- Max spectral deviation: {spectrum['max_spectral_deviation']:.1%}\n"
                report += f"- Mean spectral deviation: {spectrum['mean_spectral_deviation']:.1%}\n"
                report += f"- Threshold: {spectrum['threshold']:.1%}\n"
            
            if 'kolmogorov_pass' in spectrum:
                kolm_status = "PASS" if spectrum['kolmogorov_pass'] else "FAIL"
                report += f"- Kolmogorov scaling: {kolm_status}\n"
                report += f"- Inertial slope: {spectrum['inertial_slope']:.3f} (target: -1.667)\n"
            report += "\n"
        
        # Wall law
        if 'wall_law' in validation_results:
            wall = validation_results['wall_law']
            status = "PASS" if wall.get('wall_law_pass', False) else "FAIL"
            report += f"### Wall Law: {status}\n"
            
            if 'kappa' in wall:
                report += f"- von Kármán constant κ: {wall['kappa']:.3f}\n"
                report += f"- Additive constant B: {wall['log_intercept_B']:.3f}\n"
                report += f"- Max log deviation: {wall['max_log_deviation']:.3f}\n"
            
            if 'viscous_pass' in wall:
                visc_status = "PASS" if wall['viscous_pass'] else "FAIL"
                report += f"- Viscous sublayer: {visc_status}\n"
            report += "\n"
        
        return report
