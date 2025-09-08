#!/usr/bin/env python3
"""
Script to update all config paths for friend's HPC (n63719vm)
Run this on friend's HPC after cloning repo and transferring data
"""
import os
import glob
import yaml

def update_config_paths(config_path, friend_username="n63719vm"):
    """Update all paths in config file for friend's HPC"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Base paths for friend's HPC
    base_path = f"/mnt/iusers01/fse-ugpgt01/mace01/{friend_username}"
    
    # Update dataset paths
    if 'dataset' in config and 'data_dir' in config['dataset']:
        old_path = config['dataset']['data_dir']
        # Replace your username with friend's
        new_path = old_path.replace("p78669sb", friend_username)
        config['dataset']['data_dir'] = new_path
        print(f"Updated data_dir: {old_path} -> {new_path}")
    
    # Update all paths in paths section
    if 'paths' in config:
        for key, old_path in config['paths'].items():
            if isinstance(old_path, str) and old_path.startswith('/mnt/iusers01'):
                new_path = old_path.replace("p78669sb", friend_username)
                config['paths'][key] = new_path
                print(f"Updated {key}: {old_path} -> {new_path}")
    
    # Write updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    return config

def main():
    """Update all config files for friend's HPC"""
    
    # Find all config files
    config_patterns = [
        "configs/3d/*.yaml",
        "configs/3d_primary/*.yaml", 
        "configs/3d_primary_final/*.yaml",
        "configs/3d_secondary/*.yaml"
    ]
    
    updated_count = 0
    
    for pattern in config_patterns:
        config_files = glob.glob(pattern)
        for config_file in config_files:
            if os.path.exists(config_file):
                print(f"\nUpdating {config_file}...")
                update_config_paths(config_file)
                updated_count += 1
    
    print(f"\n✅ Updated {updated_count} config files for friend's HPC")
    print("Ready to run analysis pipeline!")

if __name__ == "__main__":
    main()
