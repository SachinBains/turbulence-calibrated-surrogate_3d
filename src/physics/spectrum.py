"""
Energy spectrum analysis for turbulent flows.
"""
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

def compute_3d_fft(field: np.ndarray) -> np.ndarray:
    """
    Compute 3D FFT of a field.
    
    Args:
        field: 3D field of shape (D, H, W)
        
    Returns:
        3D FFT coefficients
    """
    return np.fft.fftn(field)

def compute_wavenumber_grid(shape: Tuple[int, int, int], dx: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute wavenumber grids for 3D FFT.
    
    Args:
        shape: Grid shape (D, H, W)
        dx: Grid spacing
        
    Returns:
        Tuple of (wavenumber_magnitude, wavenumber_components)
    """
    D, H, W = shape
    
    # Wavenumber components
    kx = np.fft.fftfreq(D, dx) * 2 * np.pi
    ky = np.fft.fftfreq(H, dx) * 2 * np.pi
    kz = np.fft.fftfreq(W, dx) * 2 * np.pi
    
    # 3D grids
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Magnitude
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    
    return K, (KX, KY, KZ)

def compute_energy_spectrum_3d(velocity_field: np.ndarray, dx: float = 1.0,
                              k_bins: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 3D energy spectrum E(k) from velocity field.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components [u, v, w]
        dx: Grid spacing
        k_bins: Optional wavenumber bins
        
    Returns:
        Tuple of (wavenumbers, energy_spectrum)
    """
    if velocity_field.shape[0] != 3:
        raise ValueError("Expected velocity field with 3 components")
    
    D, H, W = velocity_field.shape[1:]
    
    # Compute FFTs of velocity components
    u_hat = np.fft.fftn(velocity_field[0])
    v_hat = np.fft.fftn(velocity_field[1])
    w_hat = np.fft.fftn(velocity_field[2])
    
    # Kinetic energy in Fourier space
    E_k = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2 + np.abs(w_hat)**2)
    
    # Wavenumber magnitude
    K, _ = compute_wavenumber_grid((D, H, W), dx)
    
    # Flatten arrays
    K_flat = K.flatten()
    E_k_flat = E_k.flatten()
    
    # Remove zero wavenumber
    nonzero_mask = K_flat > 0
    K_flat = K_flat[nonzero_mask]
    E_k_flat = E_k_flat[nonzero_mask]
    
    # Define wavenumber bins
    if k_bins is None:
        k_min = np.min(K_flat)
        k_max = np.min([np.max(K_flat), np.pi/dx])  # Nyquist limit
        n_bins = min(50, len(K_flat) // 10)
        k_bins = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    
    # Bin the spectrum
    k_centers = (k_bins[:-1] + k_bins[1:]) / 2
    spectrum = np.zeros(len(k_centers))
    
    for i in range(len(k_centers)):
        mask = (K_flat >= k_bins[i]) & (K_flat < k_bins[i + 1])
        if np.any(mask):
            # Sum energy in this wavenumber shell
            spectrum[i] = np.sum(E_k_flat[mask])
    
    return k_centers, spectrum

def compute_kolmogorov_spectrum(k: np.ndarray, epsilon: float, C_k: float = 1.5) -> np.ndarray:
    """
    Compute theoretical Kolmogorov -5/3 energy spectrum.
    
    Args:
        k: Wavenumber array
        epsilon: Dissipation rate
        C_k: Kolmogorov constant
        
    Returns:
        Theoretical energy spectrum E(k) = C_k * epsilon^(2/3) * k^(-5/3)
    """
    return C_k * (epsilon**(2/3)) * (k**(-5/3))

def fit_inertial_range(k: np.ndarray, spectrum: np.ndarray, 
                      k_range: Optional[Tuple[float, float]] = None) -> Dict[str, float]:
    """
    Fit power law to inertial range of energy spectrum.
    
    Args:
        k: Wavenumber array
        spectrum: Energy spectrum
        k_range: Optional wavenumber range for fitting
        
    Returns:
        Dictionary with fit parameters
    """
    # Remove zeros and invalid values
    valid_mask = (spectrum > 0) & np.isfinite(spectrum) & (k > 0)
    k_valid = k[valid_mask]
    spectrum_valid = spectrum[valid_mask]
    
    if len(k_valid) < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan}
    
    # Define fitting range
    if k_range is None:
        # Use middle portion of spectrum
        k_min = np.percentile(k_valid, 20)
        k_max = np.percentile(k_valid, 80)
    else:
        k_min, k_max = k_range
    
    # Select points in range
    range_mask = (k_valid >= k_min) & (k_valid <= k_max)
    k_fit = k_valid[range_mask]
    spectrum_fit = spectrum_valid[range_mask]
    
    if len(k_fit) < 3:
        return {'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan}
    
    # Log-log fit: log(E) = slope * log(k) + intercept
    log_k = np.log10(k_fit)
    log_E = np.log10(spectrum_fit)
    
    # Linear regression
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    slope, intercept = np.linalg.lstsq(A, log_E, rcond=None)[0]
    
    # R-squared
    log_E_pred = slope * log_k + intercept
    ss_res = np.sum((log_E - log_E_pred)**2)
    ss_tot = np.sum((log_E - np.mean(log_E))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_squared),
        'k_range': (float(k_min), float(k_max))
    }

def compute_integral_length_scale(k: np.ndarray, spectrum: np.ndarray) -> float:
    """
    Compute integral length scale from energy spectrum.
    L = (3π/4) * ∫ E(k)/k dk / ∫ E(k) dk
    
    Args:
        k: Wavenumber array
        spectrum: Energy spectrum
        
    Returns:
        Integral length scale
    """
    # Remove invalid values
    valid_mask = (spectrum > 0) & np.isfinite(spectrum) & (k > 0)
    k_valid = k[valid_mask]
    spectrum_valid = spectrum[valid_mask]
    
    if len(k_valid) < 2:
        return np.nan
    
    # Numerical integration using trapezoidal rule
    numerator = np.trapz(spectrum_valid / k_valid, k_valid)
    denominator = np.trapz(spectrum_valid, k_valid)
    
    if denominator == 0:
        return np.nan
    
    L = (3 * np.pi / 4) * numerator / denominator
    return float(L)

def compute_taylor_microscale(velocity_field: np.ndarray, dx: float = 1.0) -> float:
    """
    Compute Taylor microscale λ from velocity field.
    λ = sqrt(15 * ν * <u²> / ε)
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        dx: Grid spacing
        
    Returns:
        Taylor microscale
    """
    # Compute velocity fluctuations
    u_mean = np.mean(velocity_field, axis=(1, 2, 3), keepdims=True)
    u_prime = velocity_field - u_mean
    
    # Mean square velocity
    u_squared_mean = np.mean(np.sum(u_prime**2, axis=0))
    
    # Compute dissipation rate (simplified)
    # ε ≈ 15ν * <(∂u/∂x)²>
    du_dx = np.gradient(u_prime[0], dx, axis=0)
    dv_dy = np.gradient(u_prime[1], dx, axis=1)
    dw_dz = np.gradient(u_prime[2], dx, axis=2)
    
    dissipation_approx = 15 * 1e-4 * np.mean(du_dx**2 + dv_dy**2 + dw_dz**2)
    
    if dissipation_approx <= 0:
        return np.nan
    
    lambda_taylor = np.sqrt(15 * 1e-4 * u_squared_mean / dissipation_approx)
    return float(lambda_taylor)

def compute_kolmogorov_scale(epsilon: float, nu: float = 1e-4) -> float:
    """
    Compute Kolmogorov length scale η = (ν³/ε)^(1/4).
    
    Args:
        epsilon: Dissipation rate
        nu: Kinematic viscosity
        
    Returns:
        Kolmogorov length scale
    """
    if epsilon <= 0:
        return np.nan
    
    eta = (nu**3 / epsilon)**(1/4)
    return float(eta)

def analyze_energy_spectrum(velocity_field: np.ndarray, dx: float = 1.0,
                          nu: float = 1e-4) -> Dict:
    """
    Comprehensive energy spectrum analysis.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        dx: Grid spacing
        nu: Kinematic viscosity
        
    Returns:
        Dictionary with spectrum analysis results
    """
    results = {}
    
    # Compute energy spectrum
    k, spectrum = compute_energy_spectrum_3d(velocity_field, dx)
    results['wavenumbers'] = k.tolist()
    results['energy_spectrum'] = spectrum.tolist()
    
    # Fit inertial range
    inertial_fit = fit_inertial_range(k, spectrum)
    results['inertial_range'] = inertial_fit
    
    # Length scales
    results['integral_length_scale'] = compute_integral_length_scale(k, spectrum)
    results['taylor_microscale'] = compute_taylor_microscale(velocity_field, dx)
    
    # Estimate dissipation rate from spectrum
    if not np.isnan(inertial_fit['slope']) and inertial_fit['r_squared'] > 0.5:
        # Use fitted spectrum to estimate dissipation
        C_k = 1.5  # Kolmogorov constant
        # From E(k) = C_k * ε^(2/3) * k^(-5/3)
        # ε = (E(k) * k^(5/3) / C_k)^(3/2)
        k_ref = np.median(k[spectrum > 0])
        E_ref = np.interp(k_ref, k, spectrum)
        epsilon_est = (E_ref * k_ref**(5/3) / C_k)**(3/2)
        results['dissipation_rate_spectrum'] = float(epsilon_est)
        
        # Kolmogorov scale
        results['kolmogorov_scale'] = compute_kolmogorov_scale(epsilon_est, nu)
    else:
        results['dissipation_rate_spectrum'] = np.nan
        results['kolmogorov_scale'] = np.nan
    
    # Theoretical Kolmogorov spectrum for comparison
    if not np.isnan(results.get('dissipation_rate_spectrum', np.nan)):
        kolm_spectrum = compute_kolmogorov_spectrum(k, results['dissipation_rate_spectrum'])
        results['kolmogorov_spectrum'] = kolm_spectrum.tolist()
    
    return results
