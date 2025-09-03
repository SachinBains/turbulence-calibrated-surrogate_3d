import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from typing import List, Tuple
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
        if 'data' in cfg and 'data_dir' in cfg['data']:
            self.data_dir = Path(cfg['data']['data_dir'])
        elif 'paths' in cfg and 'data_dir' in cfg['paths']:
            self.data_dir = Path(cfg['paths']['data_dir'])
        else:
            raise ValueError("Config must contain data_dir path")
            
        self.split = split
        self.eval_mode = eval_mode
        self.cfg = cfg
        
        # Find all cube files
        cube_files = sorted(glob.glob(str(self.data_dir / "cube_64_*.h5")))
        
        if not cube_files:
            raise ValueError(f"No cube_64_*.h5 files found in {self.data_dir}")
        
        # Split data: 70% train, 15% val, 15% test
        n_total = len(cube_files)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        if split == 'train':
            self.cube_files = cube_files[:n_train]
        elif split == 'val':
            self.cube_files = cube_files[n_train:n_train+n_val]
        else:  # test
            self.cube_files = cube_files[n_train+n_val:]
        
        print(f"ChannelDataset {split}: {len(self.cube_files)} files")
        
        # Compute normalization stats
        self._compute_stats()
    
    def _compute_stats(self):
        """Compute normalization statistics from first few files."""
        velocities = []
        
        # Use first 5 files for stats
        for i in range(min(5, len(self.cube_files))):
            try:
                with h5py.File(self.cube_files[i], 'r') as f:
                    velocity = f['velocity'][:]  # Shape: (64, 64, 64, 3)
                    velocities.append(velocity.reshape(-1, 3))
            except Exception as e:
                print(f"Warning: Could not load {self.cube_files[i]}: {e}")
                continue
        
        if velocities:
            all_vel = np.concatenate(velocities, axis=0)
            self.velocity_mean = np.mean(all_vel, axis=0)
            self.velocity_std = np.std(all_vel, axis=0)
        else:
            # Fallback
            self.velocity_mean = np.array([0.0, 0.0, 0.0])
            self.velocity_std = np.array([1.0, 1.0, 1.0])
        
        print(f"Velocity stats - Mean: {self.velocity_mean}, Std: {self.velocity_std}")
    
    def __len__(self):
        return len(self.cube_files)
    
    def __getitem__(self, idx):
        """Load a velocity cube - returns (x, y) tuple like HITDataset."""
        cube_file = self.cube_files[idx]
        
        with h5py.File(cube_file, 'r') as f:
            velocity = f['velocity'][:]  # Shape: (64, 64, 64, 3)
        
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
