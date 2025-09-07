import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import typing
import glob

class ChannelDataset(Dataset):
    """Dataset for channel flow velocity cubes - compatible with HITDataset interface."""
    
    def __init__(self, cfg, split: str = 'train', eval_mode: bool = False):
        """
        Initialize Channel Dataset.
        
        Args:
            cfg: Configuration dictionary with data paths
            split: Dataset split ('train', 'val', 'test')
            eval_mode: Whether in evaluation mode (affects normalization)
        """
        # Extract data directory from config
        if 'dataset' in cfg and 'data_dir' in cfg['dataset']:
            self.data_dir = Path(cfg['dataset']['data_dir'])
        elif 'data' in cfg and 'data_dir' in cfg['data']:
            self.data_dir = Path(cfg['data']['data_dir'])
        elif 'paths' in cfg and 'data_dir' in cfg['paths']:
            self.data_dir = Path(cfg['paths']['data_dir'])
        else:
            raise ValueError("Config must contain data_dir path in dataset, data, or paths section")
            
        self.split = split
        self.eval_mode = eval_mode
        self.cfg = cfg
        
        # Find all cube files - handle both single directory and batch structure
        cube_files = []
        
        # Check if we have batch directories (Re1000_Batch1_Data2, etc.)
        batch_dirs = sorted(self.data_dir.glob("Re1000_Batch*_Data*"))
        
        if batch_dirs:
            print(f"Found {len(batch_dirs)} batch directories for Y+ bands")
            for batch_dir in batch_dirs:
                batch_files = sorted(glob.glob(str(batch_dir / "ret1000_cube_*.h5")))
                if not batch_files:
                    batch_files = sorted(glob.glob(str(batch_dir / "cube_*.h5")))
                if not batch_files:
                    batch_files = sorted(glob.glob(str(batch_dir / "chan96_*.h5")))
                cube_files.extend(batch_files)
                print(f"  {batch_dir.name}: {len(batch_files)} files")
        else:
            # Single directory structure (fallback)
            cube_files = sorted(glob.glob(str(self.data_dir / "ret1000_cube_*.h5")))
            if not cube_files:
                cube_files = sorted(glob.glob(str(self.data_dir / "cube_64_*.h5")))
            if not cube_files:
                cube_files = sorted(glob.glob(str(self.data_dir / "cube_*.h5")))
            if not cube_files:
                cube_files = sorted(glob.glob(str(self.data_dir / "chan96_*.h5")))
        
        if not cube_files:
            raise ValueError(f"No cube files found in {self.data_dir} or batch subdirectories")
        
        # Use stratified splits if available, otherwise fall back to simple splitting
        splits_dir = Path("splits")
        train_split_file = splits_dir / "channel_train_idx.npy"
        
        if train_split_file.exists():
            # Load stratified splits
            if split == 'train':
                indices = np.load(splits_dir / "channel_train_idx.npy")
            elif split == 'val':
                indices = np.load(splits_dir / "channel_val_idx.npy")
            elif split == 'test':
                indices = np.load(splits_dir / "channel_test_idx.npy")
            elif split == 'cal':
                indices = np.load(splits_dir / "channel_cal_idx.npy")
            else:
                raise ValueError(f"Unknown split: {split}")
            
            self.cube_files = [cube_files[i] for i in indices]
            print(f"ChannelDataset {split}: {len(self.cube_files)} files (stratified Y+ splits)")
        else:
            # Fallback to simple splitting
            print("Warning: No stratified splits found, using simple filename-based splitting")
            n_total = len(cube_files)
            n_train = int(0.7 * n_total)
            n_val = int(0.15 * n_total)
            
            if split == 'train':
                self.cube_files = cube_files[:n_train]
            elif split == 'val':
                self.cube_files = cube_files[n_train:n_train+n_val]
            else:  # test
                self.cube_files = cube_files[n_train+n_val:]
            
            print(f"ChannelDataset {split}: {len(self.cube_files)} files (simple splits)")
        
        
        # Compute normalization stats
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute normalization statistics following thesis methodology."""
        
        # Check if we should compute stats or load pre-computed ones
        stats_file = Path("splits") / "channel_normalization_stats.npz"
        
        if self.split == 'train' or not stats_file.exists():
            # Compute stats from ALL training files (thesis methodology)
            print("Computing normalization statistics from all training files...")
            velocities = []
            
            # Use all training files for comprehensive statistics
            max_files = len(self.cube_files) if self.split == 'train' else min(50, len(self.cube_files))
            
            for i in range(max_files):
                try:
                    with h5py.File(self.cube_files[i], 'r') as f:
                        # Handle different possible keys
                        if 'velocity' in f:
                            velocity = f['velocity'][:]
                        elif 'u' in f:
                            # Combine u, v, w components
                            u = f['u'][:]
                            v = f['v'][:]
                            w = f['w'][:]
                            velocity = np.stack([u, v, w], axis=-1)
                        else:
                            print(f"Warning: No velocity data found in {self.cube_files[i]}")
                            continue
                            
                        velocities.append(velocity)
                        
                        # Progress indicator for large datasets
                        if (i + 1) % 100 == 0:
                            print(f"  Processed {i + 1}/{max_files} files...")
                            
                except Exception as e:
                    print(f"Warning: Could not load {self.cube_files[i]}: {e}")
                    continue
            
            if not velocities:
                raise ValueError("No valid velocity data found for computing statistics")
            
            # Stack all velocities and compute stats
            all_velocities = np.stack(velocities, axis=0)
            
            # Compute per-component statistics (thesis: μ_train, σ_train ∈ R³)
            self.mean = np.mean(all_velocities, axis=(0, 1, 2, 3))  # Shape: (3,)
            self.std = np.std(all_velocities, axis=(0, 1, 2, 3))    # Shape: (3,)
            
            # Add small epsilon for numerical stability (thesis: σ + 10⁻⁸)
            self.std = self.std + 1e-8
            
            # Save stats for other splits (thesis: "frozen for validation, test")
            if self.split == 'train':
                stats_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(stats_file, mean=self.mean, std=self.std)
                print(f"Saved normalization stats to {stats_file}")
            
            print(f"Computed normalization stats from {len(velocities)} files:")
            print(f"  Mean: {self.mean}")
            print(f"  Std:  {self.std}")
            
        else:
            # Load frozen stats for val/test (thesis methodology)
            print("Loading frozen normalization statistics from training...")
            stats = np.load(stats_file)
            self.mean = stats['mean']
            self.std = stats['std']
            
            print(f"Loaded frozen stats: mean={self.mean}, std={self.std}")
    
    def __len__(self):
        return len(self.cube_files)
    
    def __getitem__(self, idx):
        """Load a velocity cube - returns (x, y) tuple like HITDataset."""
        cube_file = self.cube_files[idx]
        
        with h5py.File(cube_file, 'r') as f:
            # Try different velocity keys based on file format
            if 'velocity' in f:
                velocity = f['velocity'][:]  # Original format
            elif 'u' in f:
                velocity = f['u'][:]  # JHTDB format: (96, 96, 96, 3)
            else:
                raise KeyError(f"No velocity data found in {cube_file}. Available keys: {list(f.keys())}")
        
        # Normalize
        velocity = (velocity - self.velocity_mean) / (self.velocity_std + 1e-8)
        
        # Convert to torch tensor: (3, 64, 64, 64)
        velocity_tensor = torch.from_numpy(velocity).float().permute(3, 0, 1, 2)
        
        # Return (x, y) tuple for compatibility with HITDataset interface
        if self.eval_mode:
            # In eval mode, return denormalized data for ground truth comparison
            velocity_denorm = velocity * (self.velocity_std + 1e-8) + self.velocity_mean
            velocity_denorm_tensor = torch.from_numpy(velocity_denorm).float().permute(3, 0, 1, 2)
            return velocity_tensor, velocity_denorm_tensor
        else:
            return velocity_tensor, velocity_tensor
