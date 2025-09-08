#!/usr/bin/env python3
"""
Update dataset path in channel_dataset.py to point to correct smoke test location
"""

import os
from pathlib import Path

def update_dataset_path():
    """Update the dataset loading logic to use correct smoke test path"""
    
    dataset_file = Path(__file__).parent.parent / "src/dataio/channel_dataset.py"
    
    # Read the current file
    with open(dataset_file, 'r') as f:
        content = f.read()
    
    # Replace the data directory extraction logic
    old_logic = '''        # Extract data directory from config
        if 'dataset' in cfg and 'data_dir' in cfg['dataset']:
            self.data_dir = Path(cfg['dataset']['data_dir'])
        elif 'data' in cfg and 'data_dir' in cfg['data']:
            self.data_dir = Path(cfg['data']['data_dir'])
        elif 'paths' in cfg and 'data_dir' in cfg['paths']:
            self.data_dir = Path(cfg['paths']['data_dir'])
        else:
            raise ValueError("Config must contain data_dir path in dataset, data, or paths section")'''
    
    new_logic = '''        # Extract data directory from config
        if 'dataset' in cfg and 'data_dir' in cfg['dataset']:
            self.data_dir = Path(cfg['dataset']['data_dir'])
        elif 'data' in cfg and 'data_dir' in cfg['data']:
            self.data_dir = Path(cfg['data']['data_dir'])
        elif 'paths' in cfg and 'data_dir' in cfg['paths']:
            self.data_dir = Path(cfg['paths']['data_dir'])
        else:
            raise ValueError("Config must contain data_dir path in dataset, data, or paths section")
        
        # For smoke test, ensure we use the correct path structure
        if not self.data_dir.exists():
            # Try alternative smoke test path
            alt_path = Path(str(self.data_dir).replace('/p78669sb/', '/n63719vm/'))
            if alt_path.exists():
                self.data_dir = alt_path
                print(f"Using alternative path: {self.data_dir}")'''
    
    # Replace in content
    if old_logic in content:
        content = content.replace(old_logic, new_logic)
        
        # Write back
        with open(dataset_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated dataset path logic in {dataset_file}")
        return True
    else:
        print(f"❌ Could not find expected logic in {dataset_file}")
        return False

if __name__ == "__main__":
    update_dataset_path()
