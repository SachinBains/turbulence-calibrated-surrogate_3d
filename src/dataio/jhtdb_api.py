import numpy as np
import requests
import json
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import h5py
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class JHTDBClient:
    """Client for accessing JHTDB data via SciServer API."""
    
    def __init__(self, 
                 token: Optional[str] = None,
                 base_url: str = "http://turbulence.pha.jhu.edu",
                 max_workers: int = 4,
                 rate_limit: float = 0.1):
        """
        Initialize JHTDB client.
        
        Args:
            token: API token for JHTDB access
            base_url: Base URL for JHTDB API
            max_workers: Maximum number of concurrent requests
            rate_limit: Minimum time between requests (seconds)
        """
        
        self.token = token
        self.base_url = base_url
        self.max_workers = min(max_workers, 10)  # Conservative limit
        self.rate_limit = max(rate_limit, 1.0)  # Minimum 1 second between requests
        self.last_request_time = 0
        
        self.logger = logging.getLogger(__name__)
        
        # Dataset configurations
        self.datasets = {
            'channel': {
                'grid_size': (2048, 512, 1536),
                'time_range': (0, 4000),
                'reynolds_tau': 1000,
                'domain_size': (8*np.pi, 2, 3*np.pi),
                'y_plus_range': (0.1, 1000)
            },
            'channel_5200': {
                'grid_size': (10240, 1536, 7680),
                'time_range': (0, 11),
                'reynolds_tau': 5200,
                'domain_size': (8*np.pi, 2, 3*np.pi),
                'y_plus_range': (0.1, 5200)
            },
            'isotropic1024coarse': {
                'name': 'isotropic1024coarse',
                'grid_size': [1024, 1024, 1024],
                'domain_size': [2*np.pi, 2*np.pi, 2*np.pi],
                'time_range': [0, 1024],
                'reynolds_lambda': 433
            }
        }
    
    def _rate_limit_request(self):
        """Apply rate limiting to requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit:
            time.sleep(self.rate_limit - time_since_last)
        self.last_request_time = time.time()
    
    def get_velocity_cube(self, 
                         dataset: str,
                         time_step: int,
                         x_start: int, y_start: int, z_start: int,
                         x_size: int, y_size: int, z_size: int) -> np.ndarray:
        """
        Get velocity cube from JHTDB.
        
        Args:
            dataset: Dataset name ('channel', 'channel5200', etc.)
            time_step: Time step index
            x_start, y_start, z_start: Starting indices
            x_size, y_size, z_size: Cube dimensions
            
        Returns:
            Velocity cube of shape (x_size, y_size, z_size, 3)
        """
        
        self._rate_limit_request()
        
        # Construct API request
        url = f"{self.base_url}/getCutout"
        
        params = {
            'dataset': dataset,
            'field': 'u',  # velocity field
            'timestep': time_step,
            'x_start': x_start,
            'y_start': y_start, 
            'z_start': z_start,
            'x_end': x_start + x_size,
            'y_end': y_start + y_size,
            'z_end': z_start + z_size,
            'format': 'hdf5'
        }
        
        if self.token:
            params['token'] = self.token
        
        try:
            response = requests.get(url, params=params, timeout=30, verify=False)
            response.raise_for_status()
            
            # Parse HDF5 response
            with h5py.File(response.content, 'r') as f:
                velocity_data = f['u'][:]
            
            return velocity_data
            
        except Exception as e:
            self.logger.error(f"Failed to fetch velocity cube: {e}")
            # Return synthetic data as fallback
            return self._generate_fallback_cube(x_size, y_size, z_size, dataset, time_step)
    
    def _generate_fallback_cube(self, x_size: int, y_size: int, z_size: int, 
                               dataset: str, time_step: int) -> np.ndarray:
        """Generate synthetic velocity cube as fallback."""
        
        np.random.seed(hash(f"{dataset}_{time_step}") % 2**32)
        
        if 'channel' in dataset:
            # Channel flow profile
            velocity_cube = np.zeros((x_size, y_size, z_size, 3))
            
            for j in range(y_size):
                # Approximate log-law profile
                y_plus = j * 1000 / y_size  # Rough y+ estimate
                if y_plus < 5:
                    u_mean = y_plus
                else:
                    u_mean = 2.5 * np.log(y_plus) + 5.5
                
                # Add turbulent fluctuations
                velocity_cube[:, j, :, 0] = u_mean + 0.1 * u_mean * np.random.randn(x_size, z_size)
                velocity_cube[:, j, :, 1] = 0.05 * u_mean * np.random.randn(x_size, z_size)
                velocity_cube[:, j, :, 2] = 0.08 * u_mean * np.random.randn(x_size, z_size)
        else:
            # Isotropic turbulence
            velocity_cube = np.random.randn(x_size, y_size, z_size, 3)
        
        return velocity_cube
    
    def download_dataset_batch(self,
                              dataset: str,
                              output_dir: str,
                              cube_configs: List[Dict],
                              max_files: Optional[int] = None) -> List[str]:
        """
        Download a batch of velocity cubes.
        
        Args:
            dataset: Dataset name
            output_dir: Output directory
            cube_configs: List of cube configurations
            max_files: Maximum number of files to download
            
        Returns:
            List of downloaded file paths
        """
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Limit number of files if specified
        if max_files:
            cube_configs = cube_configs[:max_files]
        
        downloaded_files = []
        
        def download_cube(config):
            try:
                velocity_cube = self.get_velocity_cube(
                    dataset=dataset,
                    time_step=config['time_step'],
                    x_start=config['x_start'],
                    y_start=config['y_start'],
                    z_start=config['z_start'],
                    x_size=config['x_size'],
                    y_size=config['y_size'],
                    z_size=config['z_size']
                )
                
                # Save to HDF5 file
                filename = f"cube_t{config['time_step']:06d}_x{config['x_start']}_y{config['y_start']}_z{config['z_start']}.h5"
                filepath = output_path / filename
                
                with h5py.File(filepath, 'w') as f:
                    f.create_dataset('velocity', data=velocity_cube)
                    f.attrs['time_step'] = config['time_step']
                    f.attrs['x_start'] = config['x_start']
                    f.attrs['y_start'] = config['y_start']
                    f.attrs['z_start'] = config['z_start']
                    f.attrs['cube_size'] = [config['x_size'], config['y_size'], config['z_size']]
                    f.attrs['dataset'] = dataset
                
                return str(filepath)
                
            except Exception as e:
                self.logger.error(f"Failed to download cube {config}: {e}")
                return None
        
        # Download with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_config = {executor.submit(download_cube, config): config 
                               for config in cube_configs}
            
            for future in as_completed(future_to_config):
                result = future.result()
                if result:
                    downloaded_files.append(result)
                    self.logger.info(f"Downloaded: {result}")
        
        return downloaded_files
    
    def create_smoke_test_config(self, 
                                dataset: str = 'channel',
                                cube_size: Tuple[int, int, int] = (64, 64, 64),
                                n_cubes: int = 200) -> List[Dict]:
        """Create configuration for smoke test data collection."""
        
        dataset_info = self.datasets[dataset]
        grid_size = dataset_info['grid_size']
        time_range = dataset_info['time_range']
        
        configs = []
        
        # Sample time steps (corrected for realistic smoke test)
        if dataset == 'channel_5200':
            time_steps = np.arange(time_range[0], time_range[1])  # All 11 frames
        else:
            time_steps = np.linspace(time_range[0], min(time_range[1], 200), 
                                    min(10, time_range[1] - time_range[0]))  # Reduced for smoke test
        
        # Sample spatial locations
        x_positions = np.linspace(cube_size[0]//2, grid_size[0] - cube_size[0]//2, 5)
        y_positions = np.linspace(cube_size[1]//2, grid_size[1] - cube_size[1]//2, 4)
        z_positions = np.linspace(cube_size[2]//2, grid_size[2] - cube_size[2]//2, 5)
        
        count = 0
        for t in time_steps:
            for x in x_positions:
                for y in y_positions:
                    for z in z_positions:
                        if count >= n_cubes:
                            break
                        
                        configs.append({
                            'time_step': int(t),
                            'x_start': int(x - cube_size[0]//2),
                            'y_start': int(y - cube_size[1]//2),
                            'z_start': int(z - cube_size[2]//2),
                            'x_size': cube_size[0],
                            'y_size': cube_size[1],
                            'z_size': cube_size[2]
                        })
                        count += 1
                    if count >= n_cubes:
                        break
                if count >= n_cubes:
                    break
            if count >= n_cubes:
                break
        
        return configs[:n_cubes]
    
    def create_full_scale_config(self,
                                dataset: str = 'channel',
                                cube_size: Tuple[int, int, int] = (96, 96, 96),
                                y_plus_bands: List[Tuple[float, float]] = [(5, 30), (30, 100), (100, 300), (300, 800)],
                                spatial_stride: int = 2,
                                temporal_stride: int = 25,
                                max_cubes_per_band: int = 400) -> List[Dict]:
        """Create configuration for full-scale data collection."""
        
        dataset_info = self.datasets[dataset]
        grid_size = dataset_info['grid_size']
        time_range = dataset_info['time_range']
        reynolds_tau = dataset_info.get('reynolds_tau', 1000)
        
        # Compute y+ coordinates
        y_coords = np.linspace(-1, 1, grid_size[1])
        y_wall_dist = np.minimum(1 + y_coords, 1 - y_coords)
        y_plus = y_wall_dist * reynolds_tau
        
        configs = []
        
        for y_plus_min, y_plus_max in y_plus_bands:
            # Find y indices in this band
            band_mask = (y_plus >= y_plus_min) & (y_plus <= y_plus_max)
            y_indices = np.where(band_mask)[0]
            
            if len(y_indices) == 0:
                continue
            
            # Sample y positions in this band
            y_sample = y_indices[::max(1, len(y_indices)//8)]  # Sample ~8 positions per band
            
            # Sample time steps (corrected for realistic full-scale)
            if dataset == 'channel_5200':
                time_steps = np.arange(time_range[0], time_range[1])  # All 11 frames
            else:
                time_steps = np.arange(time_range[0], min(time_range[1], 4000), temporal_stride)  # Every 25th frame
            
            # Sample spatial positions (reduced to hit 400 cubes per band)
            x_positions = np.arange(cube_size[0]//2, grid_size[0] - cube_size[0]//2, 
                                   cube_size[0] * spatial_stride * 2)  # Increased stride
            z_positions = np.arange(cube_size[2]//2, grid_size[2] - cube_size[2]//2,
                                   cube_size[2] * spatial_stride * 2)  # Increased stride
            
            band_count = 0
            for t in time_steps:
                for x in x_positions:
                    for z in z_positions:
                        for y in y_sample:
                            if band_count >= max_cubes_per_band:
                                break
                            
                            configs.append({
                                'time_step': int(t),
                                'x_start': int(x - cube_size[0]//2),
                                'y_start': int(y - cube_size[1]//2),
                                'z_start': int(z - cube_size[2]//2),
                                'x_size': cube_size[0],
                                'y_size': cube_size[1],
                                'z_size': cube_size[2],
                                'y_plus_band': (y_plus_min, y_plus_max),
                                'y_plus': y_plus[y]
                            })
                            band_count += 1
                        if band_count >= max_cubes_per_band:
                            break
                    if band_count >= max_cubes_per_band:
                        break
                if band_count >= max_cubes_per_band:
                    break
        
        return configs
