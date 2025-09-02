#!/usr/bin/env python3
"""
Validate 3D Track Setup and Path Separation
Ensures complete isolation from original artifacts and proper 3D configuration
"""

import os
import sys
from pathlib import Path
import yaml
import json

def check_path_separation():
    """Verify no references to original artifacts directory"""
    
    print("=== Path Separation Validation ===")
    
    base_dir = Path.cwd()
    original_artifacts = "/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts"
    artifacts_3d = "/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"
    
    violations = []
    
    # Check all YAML configs
    config_files = list(base_dir.glob("configs/**/*.yaml"))
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                if original_artifacts in content:
                    violations.append(f"Config {config_file} references original artifacts")
                elif artifacts_3d not in content and "artifacts" in content:
                    violations.append(f"Config {config_file} has ambiguous artifacts reference")
        except Exception as e:
            print(f"Warning: Could not read {config_file}: {e}")
    
    # Check Python scripts for hardcoded paths
    script_files = list(base_dir.glob("scripts/**/*.py"))
    for script_file in script_files:
        try:
            with open(script_file, 'r') as f:
                content = f.read()
                if original_artifacts in content:
                    violations.append(f"Script {script_file} references original artifacts")
        except Exception as e:
            print(f"Warning: Could not read {script_file}: {e}")
    
    if violations:
        print("❌ Path separation violations found:")
        for violation in violations:
            print(f"  - {violation}")
        return False
    else:
        print("✅ Path separation validated - no references to original artifacts")
        return True

def check_experiment_ids():
    """Verify no collision with original experiment IDs (E1-E6)"""
    
    print("\n=== Experiment ID Validation ===")
    
    base_dir = Path.cwd()
    original_ids = ["E1", "E2", "E3", "E4", "E5", "E6"]
    violations = []
    
    # Check config files
    config_files = list(base_dir.glob("configs/**/*.yaml"))
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                for orig_id in original_ids:
                    if f"{orig_id}_" in content or f"experiment_id: {orig_id}" in content:
                        violations.append(f"Config {config_file} uses original experiment ID {orig_id}")
        except Exception as e:
            print(f"Warning: Could not read {config_file}: {e}")
    
    # Check experiment manifest
    manifest_file = base_dir / "experiments" / "3d" / "C3D_experiments_manifest.csv"
    if manifest_file.exists():
        try:
            with open(manifest_file, 'r') as f:
                content = f.read()
                for orig_id in original_ids:
                    if orig_id in content:
                        violations.append(f"Experiment manifest references original ID {orig_id}")
        except Exception as e:
            print(f"Warning: Could not read manifest: {e}")
    
    if violations:
        print("❌ Experiment ID collisions found:")
        for violation in violations:
            print(f"  - {violation}")
        return False
    else:
        print("✅ Experiment IDs validated - no collisions with E1-E6")
        return True

def check_3d_configs():
    """Validate 3D-specific configurations"""
    
    print("\n=== 3D Configuration Validation ===")
    
    base_dir = Path.cwd()
    config_dir = base_dir / "configs" / "3d"
    
    if not config_dir.exists():
        print("❌ 3D configs directory not found")
        return False
    
    required_configs = [
        "C3D1_channel_baseline_128.yaml",
        "C3D2_channel_mc_dropout_128.yaml", 
        "C3D3_channel_ensemble_128.yaml"
    ]
    
    missing_configs = []
    invalid_configs = []
    
    for config_name in required_configs:
        config_path = config_dir / config_name
        if not config_path.exists():
            missing_configs.append(config_name)
        else:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Validate required fields
                if 'experiment_id' not in config:
                    invalid_configs.append(f"{config_name}: missing experiment_id")
                elif not config['experiment_id'].startswith('C3D'):
                    invalid_configs.append(f"{config_name}: invalid experiment_id format")
                    
                if 'paths' not in config or 'artifacts_root' not in config['paths']:
                    invalid_configs.append(f"{config_name}: missing artifacts_root path")
                elif 'artifacts_3d' not in config['paths']['artifacts_root']:
                    invalid_configs.append(f"{config_name}: artifacts_root not pointing to artifacts_3d")
                    
            except Exception as e:
                invalid_configs.append(f"{config_name}: parse error - {e}")
    
    if missing_configs or invalid_configs:
        print("❌ 3D configuration issues found:")
        for missing in missing_configs:
            print(f"  - Missing: {missing}")
        for invalid in invalid_configs:
            print(f"  - Invalid: {invalid}")
        return False
    else:
        print("✅ 3D configurations validated")
        return True

def check_directory_structure():
    """Validate required directory structure"""
    
    print("\n=== Directory Structure Validation ===")
    
    base_dir = Path.cwd()
    required_dirs = [
        "configs/3d",
        "experiments/3d", 
        "docs"
    ]
    
    required_files = [
        "docs/3d_track.md",
        "experiments/3d/C3D_experiments_manifest.csv",
        "scripts/setup_3d_artifacts.sh"
    ]
    
    missing_items = []
    
    for req_dir in required_dirs:
        if not (base_dir / req_dir).exists():
            missing_items.append(f"Directory: {req_dir}")
    
    for req_file in required_files:
        if not (base_dir / req_file).exists():
            missing_items.append(f"File: {req_file}")
    
    if missing_items:
        print("❌ Missing required items:")
        for item in missing_items:
            print(f"  - {item}")
        return False
    else:
        print("✅ Directory structure validated")
        return True

def check_git_setup():
    """Validate git configuration and upstream setup"""
    
    print("\n=== Git Configuration Validation ===")
    
    try:
        import subprocess
        
        # Check remote configuration
        result = subprocess.run(['git', 'remote', '-v'], 
                              capture_output=True, text=True, cwd=Path.cwd())
        
        if result.returncode != 0:
            print("❌ Git remote check failed")
            return False
        
        remotes = result.stdout
        
        # Validate origin points to 3d repo
        if 'turbulence-calibrated-surrogate_3d' not in remotes:
            print("❌ Origin remote not pointing to 3D repository")
            return False
        
        # Validate upstream points to original repo
        if 'upstream' not in remotes or 'turbulence-calibrated-surrogate_full' not in remotes:
            print("❌ Upstream remote not configured or not pointing to original repository")
            return False
        
        print("✅ Git configuration validated")
        return True
        
    except Exception as e:
        print(f"❌ Git validation failed: {e}")
        return False

def generate_validation_report():
    """Generate comprehensive validation report"""
    
    print("\n" + "="*60)
    print("3D TRACK SETUP VALIDATION REPORT")
    print("="*60)
    
    checks = [
        ("Path Separation", check_path_separation),
        ("Experiment IDs", check_experiment_ids), 
        ("3D Configurations", check_3d_configs),
        ("Directory Structure", check_directory_structure),
        ("Git Setup", check_git_setup)
    ]
    
    results = {}
    all_passed = True
    
    for check_name, check_func in checks:
        try:
            results[check_name] = check_func()
            all_passed = all_passed and results[check_name]
        except Exception as e:
            print(f"❌ {check_name} validation failed with error: {e}")
            results[check_name] = False
            all_passed = False
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check_name:.<30} {status}")
    
    print("\n" + "="*60)
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED")
        print("3D Track setup is complete and properly isolated!")
        print("\nNext steps:")
        print("1. Run setup_3d_artifacts.sh on CSF3")
        print("2. Download JHTDB datasets")
        print("3. Start first 3D training experiment")
    else:
        print("⚠️  VALIDATION FAILURES DETECTED")
        print("Please fix the issues above before proceeding.")
        return False
    
    return all_passed

if __name__ == "__main__":
    success = generate_validation_report()
    sys.exit(0 if success else 1)
