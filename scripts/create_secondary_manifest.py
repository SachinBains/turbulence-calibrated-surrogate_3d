#!/usr/bin/env python3
"""
Create manifest for Re_τ=5200 secondary dataset.
Scans the three batch directories and creates a unified manifest with y+ band assignments.
"""

import os
import h5py
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import re

def extract_metadata_from_filename(filename):
    """Extract metadata from secondary dataset filename format."""
    # Format: 5200_96_sample105_t5_ix6773_iy123_iz4197_yplus1155.h5
    pattern = r'5200_96_sample(\d+)_t(\d+)_ix(\d+)_iy(\d+)_iz(\d+)_yplus(\d+)\.h5'
    match = re.match(pattern, filename)
    
    if not match:
        raise ValueError(f"Filename {filename} doesn't match expected pattern")
    
    return {
        'sample_id': int(match.group(1)),
        'time_index': int(match.group(2)),
        'ix': int(match.group(3)),
        'iy': int(match.group(4)),
        'iz': int(match.group(5)),
        'yplus': int(match.group(6))
    }

def assign_yplus_band(yplus_value):
    """Assign y+ value to wall-normal band."""
    # Fixed bands: B1:[0,30), B2:[30,100), B3:[100,370), B4:[370,1000)
    if 0 <= yplus_value < 30:
        return 1
    elif 30 <= yplus_value < 100:
        return 2
    elif 100 <= yplus_value < 370:
        return 3
    elif 370 <= yplus_value <= 1000:
        return 4
    else:
        return None  # Outside expected range

def scan_secondary_directories(data_dirs):
    """Scan all secondary dataset directories and create manifest entries."""
    manifest_entries = []
    
    for data_dir in data_dirs:
        data_path = Path(data_dir)
        if not data_path.exists():
            print(f"Warning: Directory {data_dir} does not exist, skipping...")
            continue
            
        print(f"Scanning directory: {data_dir}")
        h5_files = list(data_path.glob("*.h5"))
        print(f"Found {len(h5_files)} HDF5 files")
        
        for h5_file in h5_files:
            try:
                # Extract metadata from filename
                metadata = extract_metadata_from_filename(h5_file.name)
                
                # Assign y+ band
                yplus_band = assign_yplus_band(metadata['yplus'])
                if yplus_band is None:
                    print(f"Warning: y+ value {metadata['yplus']} outside expected range for {h5_file.name}")
                    continue
                
                # Verify file can be opened and has correct structure
                with h5py.File(h5_file, 'r') as f:
                    if 'u' not in f:
                        print(f"Warning: No 'u' dataset in {h5_file.name}, skipping...")
                        continue
                    
                    velocity_shape = f['u'].shape
                    if velocity_shape != (96, 96, 96, 3):
                        print(f"Warning: Unexpected velocity shape {velocity_shape} in {h5_file.name}")
                        continue
                
                # Create manifest entry
                entry = {
                    'file_path': str(h5_file.absolute()),
                    'sample_id': metadata['sample_id'],
                    'time_index': metadata['time_index'],
                    'ix': metadata['ix'],
                    'iy': metadata['iy'],
                    'iz': metadata['iz'],
                    'yplus': metadata['yplus'],
                    'yplus_band': yplus_band,
                    're_tau': 5200,
                    'resolution': '96x96x96'
                }
                
                manifest_entries.append(entry)
                
            except Exception as e:
                print(f"Error processing {h5_file.name}: {e}")
                continue
    
    return manifest_entries

def create_secondary_manifest(data_dirs, output_path):
    """Create complete secondary dataset manifest."""
    print("="*60)
    print("CREATING RE_τ=5200 SECONDARY DATASET MANIFEST")
    print("="*60)
    
    # Scan all directories
    manifest_entries = scan_secondary_directories(data_dirs)
    
    if not manifest_entries:
        raise ValueError("No valid files found in any directory")
    
    # Convert to DataFrame
    df = pd.DataFrame(manifest_entries)
    
    # Sort by sample_id for consistency
    df = df.sort_values('sample_id').reset_index(drop=True)
    
    # Print summary statistics
    print(f"\nMANIFEST SUMMARY:")
    print(f"Total files: {len(df)}")
    print(f"Sample ID range: {df['sample_id'].min()} - {df['sample_id'].max()}")
    print(f"Y+ range: {df['yplus'].min()} - {df['yplus'].max()}")
    
    print(f"\nY+ BAND DISTRIBUTION:")
    band_counts = df['yplus_band'].value_counts().sort_index()
    for band, count in band_counts.items():
        band_ranges = {1: "[0,30)", 2: "[30,100)", 3: "[100,370)", 4: "[370,1000)"}
        print(f"  Band {band} {band_ranges[band]}: {count} cubes")
    
    # Save manifest
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nManifest saved to: {output_path}")
    
    # Also save as JSON for easy loading
    json_path = output_path.with_suffix('.json')
    df.to_json(json_path, orient='records', indent=2)
    print(f"JSON version saved to: {json_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Create Re_τ=5200 secondary dataset manifest")
    parser.add_argument("--output", required=True, help="Output manifest file path")
    parser.add_argument("--data-dirs", nargs='+', required=True, 
                       help="Secondary dataset directories")
    
    args = parser.parse_args()
    
    try:
        manifest_df = create_secondary_manifest(args.data_dirs, args.output)
        
        print(f"\n{'='*60}")
        print("SECONDARY MANIFEST CREATION COMPLETED")
        print(f"{'='*60}")
        print(f"Ready for zero-shot evaluation on {len(manifest_df)} cubes")
        
        return 0
        
    except Exception as e:
        print(f"Error creating manifest: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
