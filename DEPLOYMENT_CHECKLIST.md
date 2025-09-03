# TCS_3D Deployment Checklist

## Pre-Deployment Verification ✅

### Repository Status
- [x] All HIT references removed and replaced with ChannelDataset
- [x] 6 experiment configs (C3D1-C3D6) created and validated
- [x] 6 SLURM job scripts generated with correct CSF3 settings
- [x] All scripts updated to use correct config structure
- [x] Artifacts directory structure created and configured
- [x] Git workflow documented

### Data Verification
- [x] Data location: `/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/data_3d/channel_flow_smoke/`
- [x] 200 velocity cubes (64³) confirmed as `cube_64_*.h5` files
- [x] Dataset splits: 140 train / 30 val / 30 test

### Environment Setup
- [x] CSF3 partition: `gpuV`
- [x] Required modules: `python/3.9`, `cuda/11.8`, `gcc/9.3.0`
- [x] Virtual environment: `turbml`
- [x] Python path configured in SLURM scripts

## Deployment Steps

### 1. Push to GitHub
```bash
cd "c:\Users\Sachi\OneDrive\Desktop\Dissertation\turbulence-calibrated-surrogate_3d"
git add .
git commit -m "TCS_3D: Complete pipeline cleanup and 6 experiment configs ready"
git push origin main
```

### 2. Sync to CSF3
```bash
# SSH to CSF3
ssh p78669sb@csf3.itservices.manchester.ac.uk

# Navigate and pull latest
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/turbulence-calibrated-surrogate_3d
git pull origin main

# Activate environment
source /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/venvs/turbml/bin/activate
```

### 3. Verification Tests
```bash
# Test data loading
python scripts/check_loader.py --config configs/3d/C3D1_channel_baseline_128.yaml --split train

# Test config parsing
python -c "
import yaml
cfg = yaml.safe_load(open('configs/3d/C3D1_channel_baseline_128.yaml'))
print(f'Data dir: {cfg[\"dataset\"][\"data_dir\"]}')
print(f'Channels: {cfg[\"model\"][\"in_channels\"]} -> {cfg[\"model\"][\"out_channels\"]}')
"
```

### 4. Launch Experiments
```bash
# Submit all experiments
bash job/submit_all_experiments.sh

# Monitor progress
squeue -u p78669sb
```

## Expected Outputs

### Job Submission
```
Submitting all C3D experiments...
Submitting C3D1 (Baseline)...
C3D1 Job ID: XXXXXXX
Submitting C3D2 (MC Dropout)...
C3D2 Job ID: XXXXXXX
...
```

### Directory Structure After Training
```
artifacts_3d/
├── checkpoints/
│   ├── C3D1_channel_baseline_128/
│   │   ├── best_model.pth
│   │   └── final_model.pth
│   └── ...
├── results/
│   ├── C3D1_channel_baseline_128/
│   │   ├── metrics.json
│   │   └── predictions.npz
│   └── ...
└── logs/
    └── training/
        ├── C3D1_JOBID.out
        └── C3D1_JOBID.err
```

## Troubleshooting

### Common Issues
1. **Module not found**: Check if correct modules loaded in SLURM script
2. **Data path error**: Verify data directory exists on CSF3
3. **GPU allocation**: Ensure `gpuV` partition available
4. **Memory issues**: Reduce batch size if OOM errors occur

### Quick Fixes
```bash
# Check data directory
ls -la /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/data_3d/channel_flow_smoke/

# Check modules
module avail python
module avail cuda

# Check GPU availability
sinfo -p gpuV
```

## Success Criteria
- [ ] All 6 jobs submitted successfully
- [ ] No immediate SLURM errors in first 5 minutes
- [ ] Training logs show data loading without errors
- [ ] Model checkpoints being saved to artifacts_3d/

## Next Steps After Training
1. Monitor job completion
2. Sync results back to local: `scp -r artifacts_3d/ local_path/`
3. Analyze results and generate comparison plots
4. Prepare findings for dissertation chapter
