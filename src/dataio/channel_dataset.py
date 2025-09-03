import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from typing import List, Tuple
import glob

class ChannelDataset(Dataset):
    """Simple dataset for channel flow velocity cubes."""
    
    def __init__(self, data_dir: str, split: str = 'train'):
        """
        Initialize Channel Dataset.
        
        Args:
            data_dir: Directory containing cube_64_*.h5 files
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        
        # Find all cube files
        cube_files = sorted(glob.glob(str(self.data_dir / "cube_64_*.h5")))
        
        if not cube_files:
            raise ValueError(f"No cube_64_*.h5 files found in {data_dir}")
        
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
        """Load a velocity cube."""
        cube_file = self.cube_files[idx]
        
        with h5py.File(cube_file, 'r') as f:
            velocity = f['velocity'][:]  # Shape: (64, 64, 64, 3)
        
        # Normalize
        velocity = (velocity - self.velocity_mean) / (self.velocity_std + 1e-8)
        
        # Convert to torch tensor: (3, 64, 64, 64)
        velocity_tensor = torch.from_numpy(velocity).float().permute(3, 0, 1, 2)
        
        return {
            'input': velocity_tensor,
            'target': velocity_tensor,  # For now, same as input (autoencoder style)
            'file': str(cube_file)
        }
