#!/usr/bin/env python3
"""
Fix ensemble directory structure for analysis compatibility
Moves ensemble results from nested C3D*/C3D*/members/ to flat C3D*/ structure
"""

import os
import shutil
from pathlib import Path
import argparse

def fix_ensemble_structure(artifacts_root):
    """Fix ensemble directory structure to be compatible with analysis scripts"""
    results_dir = Path(artifacts_root) / "results"
    
    # Find all ensemble experiment directories
    ensemble_dirs = [d for d in results_dir.glob("C3D*_ensemble_*") if d.is_dir()]
    
    for ensemble_dir in ensemble_dirs:
        print(f"Processing {ensemble_dir.name}...")
        
        # Check for nested structure: C3D3_ensemble/C3D3_ensemble/members/
        nested_dir = ensemble_dir / ensemble_dir.name
        members_dir = nested_dir / "members"
        
        if members_dir.exists():
            print(f"  Found nested structure: {members_dir}")
            
            # Move members directory up to ensemble_dir level
            target_members = ensemble_dir / "members"
            if target_members.exists():
                print(f"  Removing existing {target_members}")
                shutil.rmtree(target_members)
            
            print(f"  Moving {members_dir} -> {target_members}")
            shutil.move(str(members_dir), str(target_members))
            
            # Move any other files from nested_dir to ensemble_dir
            for item in nested_dir.iterdir():
                target_item = ensemble_dir / item.name
                if target_item.exists():
                    if target_item.is_dir():
                        shutil.rmtree(target_item)
                    else:
                        target_item.unlink()
                
                print(f"  Moving {item} -> {target_item}")
                shutil.move(str(item), str(target_item))
            
            # Remove empty nested directory
            if nested_dir.exists() and not any(nested_dir.iterdir()):
                print(f"  Removing empty {nested_dir}")
                nested_dir.rmdir()
        
        # Verify structure
        final_members = ensemble_dir / "members"
        if final_members.exists():
            member_count = len([d for d in final_members.iterdir() if d.is_dir()])
            print(f"  ✅ Fixed: {member_count} ensemble members in {final_members}")
        else:
            print(f"  ⚠️  No members directory found after fix")

def main():
    parser = argparse.ArgumentParser(description="Fix ensemble directory structure")
    parser.add_argument("--artifacts_dir", required=True, help="Path to artifacts directory")
    args = parser.parse_args()
    
    print("=== Fixing Ensemble Directory Structure ===")
    fix_ensemble_structure(args.artifacts_dir)
    print("✅ Ensemble structure fix complete!")

if __name__ == "__main__":
    main()
