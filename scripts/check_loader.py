import argparse
import yaml
import torch
from torch.utils.data import DataLoader
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dataio.hit_dataset import HITDataset

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"], help="Which split to test")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    ds = HITDataset(cfg, split=args.split)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    for i, (X, y) in enumerate(loader):
        print(f"Batch {i+1}:")
        print(f"  X shape: {tuple(X.shape)}")
        print(f"  y shape: {tuple(y.shape)}")
        # Print per-channel mean/std for X and y
        X_np = X.numpy()
        y_np = y.numpy()
        print(f"  X mean: {X_np.mean(axis=(0,2,3,4))}")
        print(f"  X std:  {X_np.std(axis=(0,2,3,4))}")
        print(f"  y mean: {y_np.mean(axis=(0,2,3,4))}")
        print(f"  y std:  {y_np.std(axis=(0,2,3,4))}")
        if i == 1:
            break
