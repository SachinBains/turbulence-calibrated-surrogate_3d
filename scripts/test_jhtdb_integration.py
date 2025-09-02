#!/usr/bin/env python3
"""
Test JHTDB Integration Script
Validate JHTDB dataset loading and basic pipeline functionality.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.dataio.jhtdb_dataset import JHTDBChannelDataset
from src.dataio.jhtdb_api import JHTDBClient

def test_dataset_creation():
    """Test JHTDB dataset creation and basic functionality."""
    
    print("Testing JHTDB dataset creation...")
    
    # Create temporary data directory
    data_dir = Path("./test_data_jhtdb")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize dataset
        dataset = JHTDBChannelDataset(
            data_dir=str(data_dir),
            split='train',
            cube_size=(64, 64, 64),
            y_plus_bands=[(1, 15), (50, 150), (200, 500)],
            reynolds_tau=1000,
            temporal_stride=10,
            spatial_stride=2,
            normalize=True,
            cache_data=True
        )
        
        print(f"Dataset created successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Cube size: {dataset.cube_size}")
        print(f"  - Reynolds tau: {dataset.reynolds_tau}")
        
        # Test sample loading
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample loaded successfully")
            print(f"  - Velocity shape: {sample['velocity'].shape}")
            print(f"  - Y+ value: {sample['y_plus'].item():.2f}")
            print(f"  - Y+ band: {sample['y_plus_band'].item()}")
            
            # Test physics properties
            physics = dataset.get_physics_properties(0)
            print(f"Physics properties extracted")
            print(f"  - Y+ value: {physics['y_plus']:.2f}")
            print(f"  - Grid spacing: {physics['grid_spacing']}")
        
        # Test dataset statistics
        stats = dataset.get_dataset_stats()
        print(f"Dataset statistics computed")
        print(f"  - Y+ range: [{stats['y_plus_range'][0]:.1f}, {stats['y_plus_range'][1]:.1f}]")
        print(f"  - Normalization mean: {stats['normalization_stats']['velocity_mean']}")
        
        return True
        
    except Exception as e:
        print(f"X Dataset test failed: {e}")
        return False
    
    finally:
        # Cleanup
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)

def test_api_client():
    """Test JHTDB API client functionality."""
    
    print("\nTesting JHTDB API client...")
    
    try:
        # Initialize client
        client = JHTDBClient(max_workers=2, rate_limit=0.5)
        
        print(f"API client created successfully")
        
        # Test smoke test configuration
        smoke_config = client.create_smoke_test_config(
            dataset='channel',
            cube_size=(32, 32, 32),
            n_cubes=5
        )
        
        print(f"Smoke test config generated")
        print(f"  - Number of cubes: {len(smoke_config)}")
        print(f"  - Sample config: {smoke_config[0] if smoke_config else 'None'}")
        
        # Test full scale configuration
        full_config = client.create_full_scale_config(
            dataset='channel',
            cube_size=(64, 64, 64),
            max_cubes_per_band=10
        )
        
        print(f"Full scale config generated")
        print(f"  - Number of cubes: {len(full_config)}")
        
        return True
        
    except Exception as e:
        print(f"X API client test failed: {e}")
        return False

def test_dataloader():
    """Test PyTorch DataLoader integration."""
    
    print("\nTesting PyTorch DataLoader integration...")
    
    # Create temporary data directory
    data_dir = Path("./test_data_loader")
    data_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize dataset
        dataset = JHTDBChannelDataset(
            data_dir=str(data_dir),
            split='train',
            cube_size=(32, 32, 32),
            y_plus_bands=[(1, 50), (100, 300)],
            reynolds_tau=1000,
            cache_data=True
        )
        
        # Create DataLoader
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # Use 0 for testing
            pin_memory=False
        )
        
        print(f"DataLoader created successfully")
        print(f"  - Batch size: {dataloader.batch_size}")
        print(f"  - Number of batches: {len(dataloader)}")
        
        # Test batch loading
        if len(dataloader) > 0:
            batch = next(iter(dataloader))
            print(f"Batch loaded successfully")
            print(f"  - Velocity batch shape: {batch['velocity'].shape}")
            print(f"  - Y+ batch shape: {batch['y_plus'].shape}")
            print(f"  - Device: {batch['velocity'].device}")
        
        return True
        
    except Exception as e:
        print(f"X DataLoader test failed: {e}")
        return False
    
    finally:
        # Cleanup
        import shutil
        if data_dir.exists():
            shutil.rmtree(data_dir)

def test_config_loading():
    """Test configuration file loading."""
    
    print("\nTesting configuration loading...")
    
    try:
        import yaml
        
        # Test smoke test config
        smoke_config_path = Path("../configs/channel_flow_3d_smoke.yaml")
        if smoke_config_path.exists():
            with open(smoke_config_path, 'r') as f:
                smoke_config = yaml.safe_load(f)
            
            print(f"Smoke test config loaded")
            print(f"  - Experiment: {smoke_config['experiment']['name']}")
            print(f"  - Dataset: {smoke_config['dataset']['type']}")
            print(f"  - Cube size: {smoke_config['dataset']['cube_size']}")
        
        # Test full config
        full_config_path = Path("../configs/channel_flow_3d.yaml")
        if full_config_path.exists():
            with open(full_config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            print(f"Full config loaded")
            print(f"  - Experiment: {full_config['experiment']['name']}")
            print(f"  - Model: {full_config['model']['type']}")
            print(f"  - UQ method: {full_config['model']['uncertainty_method']}")
        
        return True
        
    except Exception as e:
        print(f"X Config loading test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test JHTDB integration')
    parser.add_argument('--test', choices=['all', 'dataset', 'api', 'dataloader', 'config'], 
                       default='all', help='Which test to run')
    
    args = parser.parse_args()
    
    print("JHTDB Integration Test Suite")
    print("=" * 40)
    
    results = {}
    
    if args.test in ['all', 'dataset']:
        results['dataset'] = test_dataset_creation()
    
    if args.test in ['all', 'api']:
        results['api'] = test_api_client()
    
    if args.test in ['all', 'dataloader']:
        results['dataloader'] = test_dataloader()
    
    if args.test in ['all', 'config']:
        results['config'] = test_config_loading()
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "X FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! JHTDB integration is ready.")
        return 0
    else:
        print("Some tests failed. Check the output above.")
        return 1

if __name__ == '__main__':
    exit(main())
