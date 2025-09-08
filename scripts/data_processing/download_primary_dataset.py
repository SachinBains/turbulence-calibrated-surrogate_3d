#!/usr/bin/env python3
"""
Download Primary Dataset: Channel Flow Reτ=1000
1,600 cubes of 96³ resolution across 4 Y+ bands
"""

import numpy as np
import h5py
import time
from pathlib import Path
from giverny.turbulence_dataset import turb_dataset
import argparse

def calculate_yplus_positions(retau=1000, ny=512):
    """Calculate Y+ positions for channel flow."""
    # Channel half-height in wall units
    delta_plus = retau
    
    # Grid points in wall-normal direction (0 to 1, then mirrored)
    y_grid = np.linspace(0, 1, ny//2 + 1)  # 0 to center
    
    # Convert to Y+ (wall units)
    yplus = y_grid * delta_plus
    
    return yplus

def get_yplus_band_indices(yplus_values, band_ranges):
    """Get grid indices for each Y+ band."""
    bands = {}
    for i, (y_min, y_max) in enumerate(band_ranges):
        # Find indices within this Y+ range
        mask = (yplus_values >= y_min) & (yplus_values <= y_max)
        indices = np.where(mask)[0]
        bands[f'band_{i+1}'] = {
            'range': (y_min, y_max),
            'indices': indices,
            'count': len(indices)
        }
    return bands

def sample_cube_locations(nx, ny, nz, cube_size, n_cubes_per_time, bands):
    """Sample random cube locations for each Y+ band."""
    locations = []
    
    for band_name, band_info in bands.items():
        y_indices = band_info['indices']
        
        for _ in range(n_cubes_per_time):
            # Random x, z positions (ensure cube fits)
            x_start = np.random.randint(0, nx - cube_size + 1)
            z_start = np.random.randint(0, nz - cube_size + 1)
            
            # Random y position within band (ensure cube fits)
            if len(y_indices) >= cube_size:
                y_start_idx = np.random.choice(len(y_indices) - cube_size + 1)
                y_start = y_indices[y_start_idx]
            else:
                # If band is smaller than cube, center it
                y_start = y_indices[0] if len(y_indices) > 0 else 0
            
            locations.append({
                'band': band_name,
                'x': x_start,
                'y': y_start, 
                'z': z_start,
                'yplus_range': band_info['range']
            })
    
    return locations

def download_cube(dataset, x, y, z, cube_size, time_step):
    """Download a single velocity cube."""
    try:
        # Define cube boundaries
        x_end = x + cube_size
        y_end = y + cube_size  
        z_end = z + cube_size
        
        # Download velocity data
        velocity_data = dataset.getData(
            'velocity',
            time_step,
            x, x_end,
            y, y_end, 
            z, z_end
        )
        
        return velocity_data
        
    except Exception as e:
        print(f"Error downloading cube at ({x},{y},{z}): {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./primary_dataset_96', 
                       help='Output directory for cubes')
    parser.add_argument('--cube_size', type=int, default=96,
                       help='Cube size (96³)')
    parser.add_argument('--retau', type=int, default=1000,
                       help='Reynolds number (1000)')
    parser.add_argument('--n_time_points', type=int, default=160,
                       help='Number of time points to sample')
    parser.add_argument('--cubes_per_band_per_time', type=int, default=1,
                       help='Cubes per band per time point')
    parser.add_argument('--start_time', type=int, default=0,
                       help='Starting time step')
    parser.add_argument('--time_stride', type=int, default=25,
                       help='Time stride (every 25th frame)')
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Y+ band definitions
    yplus_bands = [
        (5, 30),     # Near-wall
        (30, 100),   # Buffer layer  
        (100, 300),  # Log layer
        (300, 800)   # Outer layer
    ]
    
    print(f"Downloading Channel Flow Reτ={args.retau} dataset")
    print(f"Target: {len(yplus_bands)} Y+ bands × {args.n_time_points} times × {args.cubes_per_band_per_time} cubes")
    print(f"Total cubes: {len(yplus_bands) * args.n_time_points * args.cubes_per_band_per_time}")
    
    # Initialize dataset
    dataset = turb_dataset()
    dataset.initialize('channel', args.retau)
    
    # Get grid info
    grid_info = dataset.get_grid_info()
    nx, ny, nz = grid_info['nx'], grid_info['ny'], grid_info['nz']
    print(f"Grid size: {nx} × {ny} × {nz}")
    
    # Calculate Y+ positions
    yplus_values = calculate_yplus_positions(args.retau, ny)
    bands = get_yplus_band_indices(yplus_values, yplus_bands)
    
    print("\nY+ Band Analysis:")
    for band_name, band_info in bands.items():
        print(f"  {band_name}: Y+ {band_info['range']} → {band_info['count']} grid points")
    
    # Download cubes
    cube_count = 0
    total_cubes = len(yplus_bands) * args.n_time_points * args.cubes_per_band_per_time
    
    for time_idx in range(args.n_time_points):
        time_step = args.start_time + time_idx * args.time_stride
        print(f"\nTime step {time_step} ({time_idx+1}/{args.n_time_points})")
        
        # Sample cube locations for this time step
        locations = sample_cube_locations(
            nx, ny, nz, args.cube_size, 
            args.cubes_per_band_per_time, bands
        )
        
        for loc in locations:
            cube_count += 1
            
            print(f"  Cube {cube_count}/{total_cubes}: {loc['band']} Y+{loc['yplus_range']} at ({loc['x']},{loc['y']},{loc['z']})")
            
            # Download cube
            velocity_data = download_cube(
                dataset, loc['x'], loc['y'], loc['z'], 
                args.cube_size, time_step
            )
            
            if velocity_data is not None:
                # Save cube
                filename = f"cube_96_{cube_count:04d}_{loc['band']}_t{time_step}.h5"
                filepath = output_dir / filename
                
                with h5py.File(filepath, 'w') as f:
                    f.create_dataset('velocity', data=velocity_data)
                    f.attrs['time_step'] = time_step
                    f.attrs['yplus_band'] = f"{loc['yplus_range'][0]}-{loc['yplus_range'][1]}"
                    f.attrs['position'] = [loc['x'], loc['y'], loc['z']]
                    f.attrs['cube_size'] = args.cube_size
                    f.attrs['retau'] = args.retau
                
                print(f"    Saved: {filename} ({velocity_data.shape})")
            else:
                print(f"    Failed to download cube {cube_count}")
            
            # Rate limiting
            time.sleep(1.2)
    
    print(f"\nDownload complete: {cube_count} cubes saved to {output_dir}")
    print(f"Total storage: ~{cube_count * 10.6:.1f} MB")

if __name__ == '__main__':
    main()
