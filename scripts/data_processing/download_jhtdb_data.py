#!/usr/bin/env python3
"""
JHTDB Data Download Script
Download 3D velocity cubes from JHTDB for turbulence surrogate modeling.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataio.jhtdb_api import JHTDBClient

def main():
    parser = argparse.ArgumentParser(description='Download JHTDB data for turbulence modeling')
    parser.add_argument('--dataset', default='channel', choices=['channel', 'channel5200', 'isotropic1024coarse'],
                       help='JHTDB dataset to download')
    parser.add_argument('--output_dir', required=True, help='Output directory for downloaded data')
    parser.add_argument('--mode', choices=['smoke_test', 'full_scale'], default='smoke_test',
                       help='Data collection mode')
    parser.add_argument('--cube_size', nargs=3, type=int, default=[64, 64, 64],
                       help='Size of velocity cubes to download')
    parser.add_argument('--n_cubes', type=int, default=200,
                       help='Number of cubes for smoke test mode (corrected: 200 cubes = 630MB)')
    parser.add_argument('--max_cubes_per_band', type=int, default=400,
                       help='Maximum cubes per y+ band for full scale mode (corrected: 400 per band)')
    parser.add_argument('--token', help='JHTDB API token')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum concurrent downloads')
    parser.add_argument('--rate_limit', type=float, default=0.1,
                       help='Rate limit between requests (seconds)')
    
    args = parser.parse_args()
    
    # Initialize JHTDB client (with conservative limits)
    client = JHTDBClient(max_workers=min(args.max_workers, 10), rate_limit=max(args.rate_limit, 1.0))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {args.dataset} data in {args.mode} mode")
    print(f"Output directory: {output_dir}")
    print(f"Cube size: {args.cube_size}")
    
    print(f"\nGenerating {args.mode} configuration...")
    if args.mode == 'smoke_test':
        cube_configs = client.create_smoke_test_config(
            dataset=args.dataset,
            cube_size=tuple(args.cube_size),
            n_cubes=args.n_cubes
        )
        expected_size = len(cube_configs) * (args.cube_size[0]**3 * 3 * 4) / (1024**2)
        print(f"Expected download size: {expected_size:.1f} MB")
        
        print(f"Generated {len(cube_configs)} cube configurations")
        if args.dataset == 'channel_5200':
            print(f"Note: Channel Re_tau=5200 limited to 11 temporal frames")
        elif args.dataset == 'channel':
            print(f"Channel Re_tau=1000 with ~4000 temporal frames available")
            
            # Save configuration for reference
            config_file = output_dir / 'download_config.json'
            with open(config_file, 'w') as f:
                json.dump(cube_configs, f, indent=2, default=str)
    else:
        cube_configs = client.create_full_scale_config(
            dataset=args.dataset,
            cube_size=tuple(args.cube_size),
            max_cubes_per_band=args.max_cubes_per_band
        )
        print(f"Generated {len(cube_configs)} cube configurations for full scale")
        
        # Save configuration for reference
        config_file = output_dir / 'download_config.json'
        with open(config_file, 'w') as f:
            json.dump(cube_configs, f, indent=2, default=str)
        print(f"Saved configuration to: {config_file}")
    
    # Download data
    print(f"Starting download with {args.max_workers} workers...")
    
    downloaded_files = client.download_dataset_batch(
        dataset=args.dataset,
        output_dir=str(output_dir),
        cube_configs=cube_configs
    )
    
    print(f"\nDownload completed!")
    print(f"Downloaded {len(downloaded_files)} files")
    print(f"Total size: {sum(Path(f).stat().st_size for f in downloaded_files if Path(f).exists()) / 1e9:.2f} GB")
    
    # Save download summary
    summary = {
        'dataset': args.dataset,
        'mode': args.mode,
        'cube_size': args.cube_size,
        'n_files': len(downloaded_files),
        'total_size_gb': sum(Path(f).stat().st_size for f in downloaded_files if Path(f).exists()) / 1e9,
        'files': downloaded_files
    }
    
    summary_file = output_dir / 'download_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Download summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
