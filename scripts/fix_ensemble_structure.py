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
    """Fix ALL nested directory structures to be compatible with analysis scripts"""
    results_dir = Path(artifacts_root) / "results"
    
    # Find ALL experiment directories (not just ensemble)
    experiment_dirs = [d for d in results_dir.glob("C3D*") if d.is_dir()]
    
    for exp_dir in experiment_dirs:
        print(f"Processing {exp_dir.name}...")
        
        # Check for nested structure: C3D*/C3D*/
        nested_dir = exp_dir / exp_dir.name
        
        if nested_dir.exists():
            print(f"  Found nested structure: {nested_dir}")
            
            # Move all contents up one level
            if "ensemble" in exp_dir.name:
                # Handle ensemble members
                members_dir = nested_dir / "members"
                if members_dir.exists():
                    target_members = exp_dir / "members"
                    if target_members.exists():
                        print(f"  Removing existing {target_members}")
                        shutil.rmtree(target_members)
                    
                    print(f"  Moving {members_dir} -> {target_members}")
                    shutil.move(str(members_dir), str(target_members))
            
            # Move all other files/dirs from nested to parent
            for item in nested_dir.iterdir():
                if item.name != "members":  # Already handled above
                    target_item = exp_dir / item.name
                    if target_item.exists():
                        if target_item.is_dir():
                            shutil.rmtree(target_item)
                        else:
                            target_item.unlink()
                    print(f"  Moving {item} -> {target_item}")
                    shutil.move(str(item), str(target_item))
            
            # Remove the now-empty nested directory
            if nested_dir.exists():
                print(f"  Removing nested directory: {nested_dir}")
                shutil.rmtree(nested_dir)
        else:
            print(f"  No nested structure found for {exp_dir.name}")
        
        # Verify structure for ensemble
        if "ensemble" in exp_dir.name:
            final_members = exp_dir / "members"
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
