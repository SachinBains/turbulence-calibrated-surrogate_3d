import argparse, yaml, sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dataio.hit_dataset import inspect_shapes, compute_channel_stats, save_stats

def main():
    ap = argparse.ArgumentParser(description="Inspect HIT HDF5 data and compute statistics")
    ap.add_argument("--config", required=True, help="Path to config YAML file")
    args = ap.parse_args()

    # Load config
    cfg = yaml.safe_load(open(args.config, "r"))
    
    print("="*60)
    print("HIT DATA INSPECTION")
    print("="*60)
    
    print("\nCONFIG:")
    print(f"  velocity_h5: {cfg['paths']['velocity_h5']}")
    print(f"  pressure_h5: {cfg['paths']['pressure_h5']}")
    print(f"  velocity_key: {cfg['data']['velocity_key']}")
    print(f"  pressure_key: {cfg['data']['pressure_key']}")
    print(f"  cube_size: {cfg['data']['block_size']}")
    
    # Inspect shapes and basic info
    print("\n" + "="*60)
    print("DATASET INSPECTION")
    print("="*60)
    
    try:
        results = inspect_shapes(cfg)
        
        print("\nVELOCITY:")
        vel_info = results['velocity']
        print(f"  File keys: {vel_info['file_keys']}")
        print(f"  Shape: {vel_info['shape']}")
        print(f"  Dtype: {vel_info['dtype']}")
        print(f"  Min: {vel_info['min']:.6f}")
        print(f"  Max: {vel_info['max']:.6f}")
        
        print("\nPRESSURE:")
        prs_info = results['pressure']
        print(f"  File keys: {prs_info['file_keys']}")
        print(f"  Shape: {prs_info['shape']}")
        print(f"  Dtype: {prs_info['dtype']}")
        print(f"  Min: {prs_info['min']:.6f}")
        print(f"  Max: {prs_info['max']:.6f}")
        
    except Exception as e:
        print(f"Error during inspection: {e}")
        return 1
    
    # Compute channel statistics
    print("\n" + "="*60)
    print("COMPUTING CHANNEL STATISTICS")
    print("="*60)
    
    try:
        cube_size = cfg['data']['block_size']
        stats = compute_channel_stats(cfg, cube_size=cube_size, max_samples=100)
        
        print(f"\nVELOCITY STATISTICS (from {stats['num_samples']} training cubes):")
        vel_stats = stats['velocity']
        for i, channel in enumerate(vel_stats['channels']):
            print(f"  {channel}: mean={vel_stats['mean'][i]:.6f}, std={vel_stats['std'][i]:.6f}")
        
        print(f"\nPRESSURE STATISTICS (from {stats['num_samples']} training cubes):")
        prs_stats = stats['pressure']
        for i, channel in enumerate(prs_stats['channels']):
            print(f"  {channel}: mean={prs_stats['mean'][i]:.6f}, std={prs_stats['std'][i]:.6f}")
        
        # Save statistics
        results_dir = Path(cfg['paths']['results_dir'])
        output_path = results_dir / 'data_stats.npz'
        save_dict = save_stats(stats, output_path)
        
        print(f"\nSUCCESS: Statistics saved to {output_path}")
        print(f"Total voxels processed: {vel_stats['shape'][0]:,}")
        
    except Exception as e:
        print(f"Error computing statistics: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

