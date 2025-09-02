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
                       help='Download mode')
    parser.add_argument('--cube_size', nargs=3, type=int, default=[64, 64, 64],
                       help='Cube size (x y z)')
    parser.add_argument('--n_cubes', type=int, default=100,
                       help='Number of cubes for smoke test')
    parser.add_argument('--max_cubes_per_band', type=int, default=1000,
                       help='Maximum cubes per y+ band for full scale')
    parser.add_argument('--token', help='JHTDB API token')
    parser.add_argument('--max_workers', type=int, default=4,
                       help='Maximum concurrent downloads')
    parser.add_argument('--rate_limit', type=float, default=0.1,
                       help='Rate limit between requests (seconds)')
    
    args = parser.parse_args()
    
    # Initialize JHTDB client
    client = JHTDBClient(
        token=args.token,
        max_workers=args.max_workers,
        rate_limit=args.rate_limit
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading {args.dataset} data in {args.mode} mode")
    print(f"Output directory: {output_dir}")
    print(f"Cube size: {args.cube_size}")
    
    # Generate download configuration
    if args.mode == 'smoke_test':
        configs = client.create_smoke_test_config(
            dataset=args.dataset,
            cube_size=tuple(args.cube_size),
            n_cubes=args.n_cubes
        )
        print(f"Generated {len(configs)} cube configurations for smoke test")
        
    else:  # full_scale
        # Channel flow y+ bands
        y_plus_bands = [(1, 5), (5, 15), (15, 50), (50, 150), (150, 500)]
        
        configs = client.create_full_scale_config(
            dataset=args.dataset,
            cube_size=tuple(args.cube_size),
            y_plus_bands=y_plus_bands,
            max_cubes_per_band=args.max_cubes_per_band
        )
        print(f"Generated {len(configs)} cube configurations for full scale")
        
        # Save configuration for reference
        config_file = output_dir / 'download_config.json'
        with open(config_file, 'w') as f:
            json.dump(configs, f, indent=2, default=str)
        print(f"Saved configuration to: {config_file}")
    
    # Download data
    print(f"Starting download with {args.max_workers} workers...")
    
    downloaded_files = client.download_dataset_batch(
        dataset=args.dataset,
        output_dir=str(output_dir),
        cube_configs=configs
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
