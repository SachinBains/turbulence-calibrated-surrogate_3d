#!/usr/bin/env python3
"""
Inspect JHTDB HDF5 file structure to understand raw data format.
This script examines the actual structure of JHTDB files to determine
what preprocessing is needed to convert raw indices to velocity fields.
"""

import h5py
import numpy as np
import argparse
from pathlib import Path

def inspect_h5_structure(file_path):
    """Recursively inspect HDF5 file structure."""
    print(f"\n{'='*60}")
    print(f"INSPECTING: {file_path}")
    print(f"{'='*60}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"File size: {Path(file_path).stat().st_size / (1024*1024):.2f} MB")
        print(f"Root keys: {list(f.keys())}")
        
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: {obj.shape} {obj.dtype}")
                if obj.size < 20:  # Print small datasets
                    print(f"{indent}  Values: {obj[...]}")
                else:
                    print(f"{indent}  Min: {obj[...].min():.6f}, Max: {obj[...].max():.6f}")
                    print(f"{indent}  Mean: {obj[...].mean():.6f}, Std: {obj[...].std():.6f}")
            else:
                print(f"{indent}{name}/ (group)")
        
        f.visititems(print_structure)

def analyze_jhtdb_physics(file_path):
    """Analyze what physics transformations are needed."""
    print(f"\n{'='*60}")
    print("PHYSICS ANALYSIS")
    print(f"{'='*60}")
    
    # JHTDB Channel Flow Parameters
    u_tau = 0.0499  # Friction velocity
    nu = 5e-5       # Kinematic viscosity
    domain_size = (8*np.pi, 2, 3*np.pi)  # (Lx, Ly, Lz)
    grid_res = (2048, 512, 1536)         # (Nx, Ny, Nz)
    
    print(f"Domain size: {domain_size}")
    print(f"Grid resolution: {grid_res}")
    print(f"u_tau: {u_tau}")
    print(f"nu: {nu}")
    print(f"Re_tau: {u_tau * 1.0 / nu:.0f}")  # Assuming half-channel height = 1
    
    # Grid spacing
    dx = domain_size[0] / grid_res[0]
    dy = domain_size[1] / grid_res[1] 
    dz = domain_size[2] / grid_res[2]
    
    print(f"Grid spacing: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
    
    with h5py.File(file_path, 'r') as f:
        print(f"\nFile contains these datasets:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                data = f[key]
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
                
                # Check if this looks like indices
                if 'index' in key.lower() or key in ['ix', 'iy', 'iz']:
                    print(f"    -> Likely spatial index, range: {data[...].min()} to {data[...].max()}")
                
                # Check if this looks like velocity
                elif key in ['u', v', 'w'] or 'velocity' in key.lower():
                    print(f"    -> Likely velocity component")
                    print(f"       Raw range: {data[...].min():.6f} to {data[...].max():.6f}")
                    print(f"       Scaled by u_tau: {data[...].min()/u_tau:.6f} to {data[...].max()/u_tau:.6f}")
                
                # Check if this looks like coordinates
                elif key in ['x', 'y', 'z']:
                    print(f"    -> Likely coordinate, range: {data[...].min():.6f} to {data[...].max():.6f}")

def main():
    parser = argparse.ArgumentParser(description="Inspect JHTDB HDF5 file structure")
    parser.add_argument("file_path", help="Path to HDF5 file to inspect")
    args = parser.parse_args()
    
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        return 1
    
    try:
        inspect_h5_structure(file_path)
        analyze_jhtdb_physics(file_path)
        
        print(f"\n{'='*60}")
        print("NEXT STEPS")
        print(f"{'='*60}")
        print("1. Identify which datasets contain velocity components")
        print("2. Determine if coordinate transformation is needed")
        print("3. Check if velocity scaling by u_tau is required")
        print("4. Implement B-spline grid mapping for y-coordinates if needed")
        print("5. Update ChannelDataset to handle raw JHTDB format")
        
    except Exception as e:
        print(f"Error inspecting file: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
