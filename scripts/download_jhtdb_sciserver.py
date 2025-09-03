#!/usr/bin/env python3
"""
Download JHTDB data using SciServer for turbulence-calibrated-surrogate_3d project.
Run this script in SciServer compute container with JH Turbulence DB image.
"""

import pyJHTDB
import numpy as np
import h5py
import os
from pathlib import Path
import argparse
import time

def download_jhtdb_data(dataset='channel', n_cubes=200, cube_size=64, output_dir='jhtdb_data'):
    """
    Download velocity cubes from JHTDB using SciServer.
    
    Args:
        dataset: JHTDB dataset name ('channel' for Re_tau=1000)
        n_cubes: Number of cubes to download
        cube_size: Size of each cube (cube_size^3)
        output_dir: Output directory for HDF5 files
    """
    
    # Set authentication token
    pyJHTDB.dbinfo.auth_token = "uk.ac.manchester.postgrad.sachin.bains-df182d45"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Dataset parameters
    if dataset == 'channel':
        grid_size = (2048, 512, 1536)  # Re_tau = 1000
        domain_size = (8*np.pi, 2, 3*np.pi)
    elif dataset == 'channel5200':
        grid_size = (8192, 1536, 6144)  # Re_tau = 5200
        domain_size = (8*np.pi, 2, 3*np.pi)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    print(f"Downloading {n_cubes} cubes of size {cube_size}³ from {dataset}")
    print(f"Grid size: {grid_size}")
    print(f"Output directory: {output_path}")
    
    # Y+ stratified sampling
    y_plus_bands = [
        (5, 30),    # Near-wall
        (30, 100),  # Buffer layer
        (100, 300), # Log layer
        (300, 800)  # Outer layer
    ]
    
    cubes_per_band = n_cubes // len(y_plus_bands)
    
    cube_idx = 0
    for band_idx, (y_plus_min, y_plus_max) in enumerate(y_plus_bands):
        print(f"\nDownloading band {band_idx+1}: y+ = [{y_plus_min}, {y_plus_max}]")
        
        for i in range(cubes_per_band):
            try:
                # Random spatial locations
                x_start = np.random.randint(0, grid_size[0] - cube_size)
                z_start = np.random.randint(0, grid_size[2] - cube_size)
                
                # Y+ stratified sampling
                y_plus_target = np.random.uniform(y_plus_min, y_plus_max)
                # Convert y+ to grid coordinates (approximate)
                y_start = int((y_plus_target / 1000.0) * grid_size[1])
                y_start = max(0, min(y_start, grid_size[1] - cube_size))
                
                # Random time step
                time_step = np.random.randint(0, 4000)
                
                print(f"  Cube {cube_idx+1}/{n_cubes}: "
                      f"pos=({x_start},{y_start},{z_start}), "
                      f"time={time_step}, y+≈{y_plus_target:.1f}")
                
                # Get velocity cube using pyJHTDB
                velocity = pyJHTDB.getCutout(
                    data_set=dataset,
                    field='u',  # velocity field
                    time_step=time_step,
                    start=np.array([x_start, y_start, z_start]),
                    size=np.array([cube_size, cube_size, cube_size]),
                    step=np.array([1, 1, 1])
                )
                
                # Transpose from pyJHTDB format (3, z, y, x) to (x, y, z, 3)
                velocity = np.transpose(velocity, (3, 2, 1, 0))
                
                # Save to HDF5
                output_file = output_path / f'cube_{cube_idx:04d}.h5'
                with h5py.File(output_file, 'w') as f:
                    f['velocity'] = velocity.astype(np.float32)
                    f['y_plus'] = y_plus_target
                    f['position'] = [x_start, y_start, z_start]
                    f['time_step'] = time_step
                    f['dataset'] = dataset
                
                print(f"    Saved: {output_file} "
                      f"(shape={velocity.shape}, "
                      f"range=[{velocity.min():.3f}, {velocity.max():.3f}])")
                
                cube_idx += 1
                
                # Rate limiting
                time.sleep(1.0)
                
            except Exception as e:
                print(f"    Error downloading cube {cube_idx}: {e}")
                continue
    
    print(f"\nDownload complete! {cube_idx} cubes saved to {output_path}")
    print(f"Total size: ~{cube_idx * cube_size**3 * 3 * 4 / 1e9:.2f} GB")
    
    # Create metadata file
    metadata_file = output_path / 'metadata.txt'
    with open(metadata_file, 'w') as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Number of cubes: {cube_idx}\n")
        f.write(f"Cube size: {cube_size}³\n")
        f.write(f"Grid size: {grid_size}\n")
        f.write(f"Domain size: {domain_size}\n")
        f.write(f"Y+ bands: {y_plus_bands}\n")
        f.write(f"Downloaded using SciServer pyJHTDB\n")
    
    print(f"Metadata saved to {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description='Download JHTDB data using SciServer')
    parser.add_argument('--dataset', default='channel', 
                       choices=['channel', 'channel5200'],
                       help='JHTDB dataset name')
    parser.add_argument('--n_cubes', type=int, default=200,
                       help='Number of cubes to download')
    parser.add_argument('--cube_size', type=int, default=64,
                       help='Size of each cube (cube_size^3)')
    parser.add_argument('--output_dir', default='jhtdb_data',
                       help='Output directory for HDF5 files')
    
    args = parser.parse_args()
    
    download_jhtdb_data(
        dataset=args.dataset,
        n_cubes=args.n_cubes,
        cube_size=args.cube_size,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()
