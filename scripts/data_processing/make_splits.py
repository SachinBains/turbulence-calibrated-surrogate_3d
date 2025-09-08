import argparse, yaml, sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from dataio.splits import make_splits_from_h5

def main():
    ap = argparse.ArgumentParser(description="Create train/val/test splits for HIT dataset")
    ap.add_argument('--config', required=True, help='Path to config YAML file')
    ap.add_argument('--mode', choices=['id', 'ab'], default='id', 
                    help='Split mode: "id" for in-domain, "ab" for A→B spatial splits')
    args = ap.parse_args()
    
    # Load config
    cfg = yaml.safe_load(open(args.config, 'r'))
    
    print(f"Creating {args.mode.upper()} splits...")
    print(f"Config: {args.config}")
    print(f"Block size: {cfg['data']['block_size']}")
    print(f"Stride: {cfg['data']['stride']}")
    print(f"Seed: {cfg.get('seed', 42)}")
    
    # Create splits
    train_idx, val_idx, test_idx = make_splits_from_h5(cfg, mode=args.mode)
    
    return 0

if __name__ == '__main__':
    exit(main())
