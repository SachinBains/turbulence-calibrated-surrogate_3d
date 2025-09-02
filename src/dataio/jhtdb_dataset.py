import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import requests
import json
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging

class JHTDBChannelDataset(Dataset):
    """JHTDB Channel Flow Dataset for 3D turbulence surrogate modeling."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 cube_size: Tuple[int, int, int] = (96, 96, 96),
                 y_plus_bands: List[Tuple[float, float]] = [(1, 5), (5, 15), (15, 50), (50, 150), (150, 500)],
                 reynolds_tau: int = 1000,
                 temporal_stride: int = 5,
                 spatial_stride: int = 1,
                 normalize: bool = True,
                 cache_data: bool = False):
        """
        Initialize JHTDB Channel Flow Dataset.
        
        Args:
            data_dir: Directory containing JHTDB data
            split: Dataset split ('train', 'val', 'test')
            cube_size: Size of 3D velocity cubes (x, y, z)
            y_plus_bands: List of y+ ranges for stratified sampling
            reynolds_tau: Reynolds number based on friction velocity
            temporal_stride: Stride for temporal sampling
            spatial_stride: Stride for spatial sampling
            normalize: Whether to normalize velocity fields
            cache_data: Whether to cache data in memory
        """
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.cube_size = cube_size
        self.y_plus_bands = y_plus_bands
        self.reynolds_tau = reynolds_tau
        self.temporal_stride = temporal_stride
        self.spatial_stride = spatial_stride
        self.normalize = normalize
        self.cache_data = cache_data
        
        # Channel flow parameters
        self.channel_height = 2.0  # Channel half-height
        self.utau = 1.0  # Friction velocity (normalized)
        self.nu = 1.0 / reynolds_tau  # Kinematic viscosity
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load dataset metadata
        self._load_metadata()
        
        # Create sample indices
        self._create_sample_indices()
        
        # Cache for loaded data
        self._data_cache = {} if cache_data else None
        
        # Normalization statistics
        self._compute_normalization_stats()
    
    def _load_metadata(self):
        """Load dataset metadata and grid information."""
        
        metadata_path = self.data_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            # Default metadata for channel flow
            self.metadata = {
                'grid_size': [2048, 512, 1536],
                'domain_size': [8*np.pi, 2.0, 3*np.pi],
                'time_steps': 4000,
                'dt': 0.0065,
                'reynolds_tau': self.reynolds_tau
            }
            
            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        
        self.grid_size = self.metadata['grid_size']
        self.domain_size = self.metadata['domain_size']
        self.time_steps = self.metadata['time_steps']
        
        # Compute grid spacing
        self.dx = self.domain_size[0] / self.grid_size[0]
        self.dy = self.domain_size[1] / self.grid_size[1]
        self.dz = self.domain_size[2] / self.grid_size[2]
        
        # Compute y+ coordinates
        y_coords = np.linspace(-1, 1, self.grid_size[1])
        y_wall_dist = np.minimum(1 + y_coords, 1 - y_coords)  # Distance from nearest wall
        self.y_plus = y_wall_dist * self.reynolds_tau
    
    def _create_sample_indices(self):
        """Create indices for samples based on y+ bands and temporal/spatial sampling."""
        
        self.sample_indices = []
        
        # Split ratios
        split_ratios = {'train': 0.6, 'val': 0.2, 'test': 0.2}
        
        for band_idx, (y_plus_min, y_plus_max) in enumerate(self.y_plus_bands):
            # Find y indices in this band
            band_mask = (self.y_plus >= y_plus_min) & (self.y_plus <= y_plus_max)
            y_indices = np.where(band_mask)[0]
            
            if len(y_indices) == 0:
                continue
            
            # Ensure cube fits in y direction
            valid_y_indices = y_indices[
                (y_indices >= self.cube_size[1]//2) & 
                (y_indices < self.grid_size[1] - self.cube_size[1]//2)
            ]
            
            # Temporal sampling
            time_indices = np.arange(0, self.time_steps, self.temporal_stride)
            
            # Spatial sampling in x and z
            x_indices = np.arange(
                self.cube_size[0]//2, 
                self.grid_size[0] - self.cube_size[0]//2, 
                self.cube_size[0] * self.spatial_stride
            )
            z_indices = np.arange(
                self.cube_size[2]//2, 
                self.grid_size[2] - self.cube_size[2]//2, 
                self.cube_size[2] * self.spatial_stride
            )
            
            # Create all combinations
            for t in time_indices:
                for x in x_indices:
                    for z in z_indices:
                        for y in valid_y_indices[::len(valid_y_indices)//min(len(valid_y_indices), 8)]:
                            self.sample_indices.append({
                                'time': t,
                                'x': x,
                                'y': y,
                                'z': z,
                                'y_plus_band': band_idx,
                                'y_plus': self.y_plus[y]
                            })
        
        # Shuffle and split
        np.random.seed(42)
        indices = np.random.permutation(len(self.sample_indices))
        
        # Compute split boundaries
        n_total = len(indices)
        n_train = int(n_total * split_ratios['train'])
        n_val = int(n_total * split_ratios['val'])
        
        if self.split == 'train':
            self.sample_indices = [self.sample_indices[i] for i in indices[:n_train]]
        elif self.split == 'val':
            self.sample_indices = [self.sample_indices[i] for i in indices[n_train:n_train+n_val]]
        else:  # test
            self.sample_indices = [self.sample_indices[i] for i in indices[n_train+n_val:]]
        
        self.logger.info(f"Created {len(self.sample_indices)} samples for {self.split} split")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics from a subset of data."""
        
        stats_path = self.data_dir / f'norm_stats_{self.reynolds_tau}.json'
        
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.velocity_mean = np.array(stats['velocity_mean'])
                self.velocity_std = np.array(stats['velocity_std'])
        else:
            # Compute from subset of training data
            self.logger.info("Computing normalization statistics...")
            
            velocities = []
            n_samples = min(100, len(self.sample_indices))
            
            for i in range(n_samples):
                try:
                    velocity_cube = self._load_velocity_cube(self.sample_indices[i])
                    velocities.append(velocity_cube.flatten())
                except:
                    continue
            
            if velocities:
                all_velocities = np.concatenate(velocities)
                self.velocity_mean = np.mean(all_velocities.reshape(-1, 3), axis=0)
                self.velocity_std = np.std(all_velocities.reshape(-1, 3), axis=0)
            else:
                # Fallback values
                self.velocity_mean = np.array([0.0, 0.0, 0.0])
                self.velocity_std = np.array([1.0, 1.0, 1.0])
            
            # Save statistics
            stats = {
                'velocity_mean': self.velocity_mean.tolist(),
                'velocity_std': self.velocity_std.tolist()
            }
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
    
    def _load_velocity_cube(self, sample_info: Dict) -> np.ndarray:
        """Load a velocity cube from JHTDB data."""
        
        # Check cache first
        cache_key = f"{sample_info['time']}_{sample_info['x']}_{sample_info['y']}_{sample_info['z']}"
        if self._data_cache is not None and cache_key in self._data_cache:
            return self._data_cache[cache_key]
        
        # Load from file
        data_file = self.data_dir / f"velocity_t{sample_info['time']:06d}.h5"
        
        if not data_file.exists():
            # Generate synthetic data for testing
            velocity_cube = self._generate_synthetic_cube(sample_info)
        else:
            with h5py.File(data_file, 'r') as f:
                # Extract cube region
                x_start = sample_info['x'] - self.cube_size[0] // 2
                x_end = x_start + self.cube_size[0]
                y_start = sample_info['y'] - self.cube_size[1] // 2
                y_end = y_start + self.cube_size[1]
                z_start = sample_info['z'] - self.cube_size[2] // 2
                z_end = z_start + self.cube_size[2]
                
                velocity_cube = f['velocity'][x_start:x_end, y_start:y_end, z_start:z_end, :]
        
        # Cache if enabled
        if self._data_cache is not None:
            self._data_cache[cache_key] = velocity_cube
        
        return velocity_cube
    
    def _generate_synthetic_cube(self, sample_info: Dict) -> np.ndarray:
        """Generate synthetic velocity cube for testing."""
        
        # Create realistic channel flow profile
        y_center = sample_info['y']
        y_wall_dist = min(y_center, self.grid_size[1] - y_center) / (self.grid_size[1] / 2)
        
        # Log-law velocity profile
        y_plus_local = sample_info['y_plus']
        if y_plus_local < 5:
            u_mean = y_plus_local  # Viscous sublayer
        else:
            u_mean = 2.5 * np.log(y_plus_local) + 5.5  # Log layer
        
        # Generate turbulent fluctuations
        np.random.seed(hash(str(sample_info)) % 2**32)
        
        # Base velocity field
        velocity_cube = np.zeros((*self.cube_size, 3))
        
        # Streamwise velocity (u)
        velocity_cube[:, :, :, 0] = u_mean + 0.1 * u_mean * np.random.randn(*self.cube_size)
        
        # Wall-normal velocity (v) - smaller fluctuations
        velocity_cube[:, :, :, 1] = 0.05 * u_mean * np.random.randn(*self.cube_size)
        
        # Spanwise velocity (w)
        velocity_cube[:, :, :, 2] = 0.08 * u_mean * np.random.randn(*self.cube_size)
        
        # Add some spatial correlation
        from scipy.ndimage import gaussian_filter
        for i in range(3):
            velocity_cube[:, :, :, i] = gaussian_filter(velocity_cube[:, :, :, i], sigma=2.0)
        
        return velocity_cube
    
    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.sample_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        
        sample_info = self.sample_indices[idx]
        
        # Load velocity cube
        velocity_cube = self._load_velocity_cube(sample_info)
        
        # Normalize if requested
        if self.normalize:
            velocity_cube = (velocity_cube - self.velocity_mean) / (self.velocity_std + 1e-8)
        
        # Convert to torch tensor and rearrange dimensions
        # From (x, y, z, components) to (components, x, y, z)
        velocity_tensor = torch.from_numpy(velocity_cube).float().permute(3, 0, 1, 2)
        
        return {
            'velocity': velocity_tensor,
            'y_plus': torch.tensor(sample_info['y_plus'], dtype=torch.float32),
            'y_plus_band': torch.tensor(sample_info['y_plus_band'], dtype=torch.long),
            'reynolds_tau': torch.tensor(self.reynolds_tau, dtype=torch.float32),
            'sample_info': sample_info
        }
    
    def get_physics_properties(self, idx: int) -> Dict[str, float]:
        """Get physics properties for a sample."""
        
        sample_info = self.sample_indices[idx]
        
        return {
            'y_plus': sample_info['y_plus'],
            'reynolds_tau': self.reynolds_tau,
            'y_wall_distance': sample_info['y_plus'] / self.reynolds_tau,
            'grid_spacing': [self.dx, self.dy, self.dz],
            'domain_size': self.domain_size
        }
    
    def get_dataset_stats(self) -> Dict[str, Union[int, float, List]]:
        """Get dataset statistics."""
        
        y_plus_values = [s['y_plus'] for s in self.sample_indices]
        
        return {
            'n_samples': len(self.sample_indices),
            'cube_size': self.cube_size,
            'y_plus_range': [min(y_plus_values), max(y_plus_values)],
            'y_plus_bands': self.y_plus_bands,
            'reynolds_tau': self.reynolds_tau,
            'grid_size': self.grid_size,
            'domain_size': self.domain_size,
            'normalization_stats': {
                'velocity_mean': self.velocity_mean.tolist(),
                'velocity_std': self.velocity_std.tolist()
            }
        }
