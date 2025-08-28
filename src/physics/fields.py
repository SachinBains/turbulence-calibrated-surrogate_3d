"""
Physics validation for turbulent flow fields.
"""
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

def compute_divergence_3d(velocity_field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute divergence of 3D velocity field using finite differences.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components [u, v, w]
        dx: Grid spacing
        
    Returns:
        Divergence field of shape (D, H, W)
    """
    if velocity_field.shape[0] != 3:
        raise ValueError("Expected velocity field with 3 components")
    
    u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
    
    # Compute gradients using central differences
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dx, axis=1) 
    dw_dz = np.gradient(w, dx, axis=2)
    
    return du_dx + dv_dy + dw_dz

def compute_vorticity_3d(velocity_field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute vorticity (curl) of 3D velocity field.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components [u, v, w]
        dx: Grid spacing
        
    Returns:
        Vorticity field of shape (3, D, H, W) - [omega_x, omega_y, omega_z]
    """
    if velocity_field.shape[0] != 3:
        raise ValueError("Expected velocity field with 3 components")
    
    u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
    
    # Compute vorticity components
    # omega_x = dw/dy - dv/dz
    dw_dy = np.gradient(w, dx, axis=1)
    dv_dz = np.gradient(v, dx, axis=2)
    omega_x = dw_dy - dv_dz
    
    # omega_y = du/dz - dw/dx  
    du_dz = np.gradient(u, dx, axis=2)
    dw_dx = np.gradient(w, dx, axis=0)
    omega_y = du_dz - dw_dx
    
    # omega_z = dv/dx - du/dy
    dv_dx = np.gradient(v, dx, axis=0)
    du_dy = np.gradient(u, dx, axis=1)
    omega_z = dv_dx - du_dy
    
    return np.stack([omega_x, omega_y, omega_z], axis=0)

def compute_strain_rate_tensor(velocity_field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute strain rate tensor S_ij = 0.5 * (du_i/dx_j + du_j/dx_i).
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components [u, v, w]
        dx: Grid spacing
        
    Returns:
        Strain rate tensor of shape (3, 3, D, H, W)
    """
    if velocity_field.shape[0] != 3:
        raise ValueError("Expected velocity field with 3 components")
    
    u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
    
    # Compute velocity gradients
    du_dx = np.gradient(u, dx, axis=0)
    du_dy = np.gradient(u, dx, axis=1)
    du_dz = np.gradient(u, dx, axis=2)
    
    dv_dx = np.gradient(v, dx, axis=0)
    dv_dy = np.gradient(v, dx, axis=1)
    dv_dz = np.gradient(v, dx, axis=2)
    
    dw_dx = np.gradient(w, dx, axis=0)
    dw_dy = np.gradient(w, dx, axis=1)
    dw_dz = np.gradient(w, dx, axis=2)
    
    # Strain rate tensor components
    S = np.zeros((3, 3) + velocity_field.shape[1:])
    
    # Diagonal terms
    S[0, 0] = du_dx
    S[1, 1] = dv_dy
    S[2, 2] = dw_dz
    
    # Off-diagonal terms (symmetric)
    S[0, 1] = S[1, 0] = 0.5 * (du_dy + dv_dx)
    S[0, 2] = S[2, 0] = 0.5 * (du_dz + dw_dx)
    S[1, 2] = S[2, 1] = 0.5 * (dv_dz + dw_dy)
    
    return S

def compute_enstrophy(velocity_field: np.ndarray, dx: float = 1.0) -> float:
    """
    Compute enstrophy (0.5 * |omega|^2) of velocity field.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        dx: Grid spacing
        
    Returns:
        Mean enstrophy
    """
    vorticity = compute_vorticity_3d(velocity_field, dx)
    enstrophy_field = 0.5 * np.sum(vorticity**2, axis=0)
    return float(np.mean(enstrophy_field))

def compute_kinetic_energy(velocity_field: np.ndarray) -> float:
    """
    Compute mean kinetic energy of velocity field.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        
    Returns:
        Mean kinetic energy
    """
    ke_field = 0.5 * np.sum(velocity_field**2, axis=0)
    return float(np.mean(ke_field))

def compute_dissipation_rate(velocity_field: np.ndarray, nu: float = 1e-4, 
                           dx: float = 1.0) -> float:
    """
    Compute viscous dissipation rate epsilon = 2*nu*S_ij*S_ij.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        nu: Kinematic viscosity
        dx: Grid spacing
        
    Returns:
        Mean dissipation rate
    """
    S = compute_strain_rate_tensor(velocity_field, dx)
    
    # Compute S_ij * S_ij (double contraction)
    S_squared = np.sum(S * S, axis=(0, 1))
    
    # Dissipation rate
    epsilon_field = 2 * nu * S_squared
    return float(np.mean(epsilon_field))

def check_incompressibility(velocity_field: np.ndarray, dx: float = 1.0, 
                          tolerance: float = 1e-6) -> Dict[str, float]:
    """
    Check incompressibility constraint (div(u) = 0).
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        dx: Grid spacing
        tolerance: Tolerance for incompressibility
        
    Returns:
        Dictionary with divergence statistics
    """
    divergence = compute_divergence_3d(velocity_field, dx)
    
    div_mean = float(np.mean(divergence))
    div_std = float(np.std(divergence))
    div_max = float(np.max(np.abs(divergence)))
    div_rms = float(np.sqrt(np.mean(divergence**2)))
    
    # Check if incompressible within tolerance
    is_incompressible = div_rms < tolerance
    
    return {
        'div_mean': div_mean,
        'div_std': div_std,
        'div_max': div_max,
        'div_rms': div_rms,
        'is_incompressible': is_incompressible,
        'tolerance': tolerance
    }

def compute_reynolds_stress_tensor(velocity_field: np.ndarray) -> np.ndarray:
    """
    Compute Reynolds stress tensor R_ij = <u'_i * u'_j> where u' are fluctuations.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        
    Returns:
        Reynolds stress tensor of shape (3, 3)
    """
    # Compute fluctuations (subtract mean)
    u_mean = np.mean(velocity_field, axis=(1, 2, 3), keepdims=True)
    u_prime = velocity_field - u_mean
    
    # Compute Reynolds stress components
    R = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            R[i, j] = np.mean(u_prime[i] * u_prime[j])
    
    return R

def compute_turbulent_kinetic_energy(velocity_field: np.ndarray) -> float:
    """
    Compute turbulent kinetic energy k = 0.5 * <u'_i * u'_i>.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        
    Returns:
        Turbulent kinetic energy
    """
    R = compute_reynolds_stress_tensor(velocity_field)
    tke = 0.5 * np.trace(R)
    return float(tke)

def validate_field_physics(velocity_field: np.ndarray, dx: float = 1.0,
                         nu: float = 1e-4, tolerance: float = 1e-6) -> Dict:
    """
    Comprehensive physics validation of velocity field.
    
    Args:
        velocity_field: Shape (3, D, H, W) - velocity components
        dx: Grid spacing
        nu: Kinematic viscosity
        tolerance: Tolerance for incompressibility
        
    Returns:
        Dictionary with all physics metrics
    """
    results = {}
    
    # Basic field properties
    results['kinetic_energy'] = compute_kinetic_energy(velocity_field)
    results['turbulent_ke'] = compute_turbulent_kinetic_energy(velocity_field)
    results['enstrophy'] = compute_enstrophy(velocity_field, dx)
    results['dissipation_rate'] = compute_dissipation_rate(velocity_field, nu, dx)
    
    # Incompressibility check
    incomp_results = check_incompressibility(velocity_field, dx, tolerance)
    results.update(incomp_results)
    
    # Reynolds stress tensor
    reynolds_stress = compute_reynolds_stress_tensor(velocity_field)
    results['reynolds_stress'] = reynolds_stress.tolist()
    
    # Anisotropy measures
    R_trace = np.trace(reynolds_stress)
    if R_trace > 0:
        # Normalized anisotropy tensor
        b_ij = reynolds_stress / R_trace - np.eye(3) / 3
        results['anisotropy_invariant_2'] = float(np.trace(b_ij @ b_ij))
        results['anisotropy_invariant_3'] = float(np.trace(b_ij @ b_ij @ b_ij))
    else:
        results['anisotropy_invariant_2'] = 0.0
        results['anisotropy_invariant_3'] = 0.0
    
    # Velocity statistics
    for i, component in enumerate(['u', 'v', 'w']):
        comp_field = velocity_field[i]
        results[f'{component}_mean'] = float(np.mean(comp_field))
        results[f'{component}_std'] = float(np.std(comp_field))
        results[f'{component}_skewness'] = float(_compute_skewness(comp_field))
        results[f'{component}_kurtosis'] = float(_compute_kurtosis(comp_field))
    
    return results

def _compute_skewness(field: np.ndarray) -> float:
    """Compute skewness of field."""
    mean_val = np.mean(field)
    std_val = np.std(field)
    if std_val == 0:
        return 0.0
    return np.mean(((field - mean_val) / std_val) ** 3)

def _compute_kurtosis(field: np.ndarray) -> float:
    """Compute kurtosis of field."""
    mean_val = np.mean(field)
    std_val = np.std(field)
    if std_val == 0:
        return 0.0
    return np.mean(((field - mean_val) / std_val) ** 4) - 3.0  # Excess kurtosis
