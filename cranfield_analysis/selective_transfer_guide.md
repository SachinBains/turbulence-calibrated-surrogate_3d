# Selective Data Transfer Guide for Analysis

## Essential Files Only - Optimized Transfer

Based on analysis pipeline requirements, you only need these files:

### 1. Model Checkpoints (Required)
```bash
# Only transfer best_*.pth files (NOT epoch_*.pth or checkpoint_epoch_*.pth)
rsync -avz --include="best_*.pth" --exclude="epoch_*.pth" --exclude="checkpoint_epoch_*.pth" \
  /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/results/ \
  n63719vm@csf3.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/results/
```

### 2. Training Metrics (Required)
```bash
# Transfer training logs and metrics
rsync -avz --include="*_metrics.json" --include="training_log.json" --include="*.csv" \
  /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/results/ \
  n63719vm@csf3.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/results/
```

### 3. Complete Selective Transfer Command
```bash
# Single command to transfer only essential files
rsync -avz \
  --include="*/" \
  --include="best_*.pth" \
  --include="*_metrics.json" \
  --include="training_log.json" \
  --include="*.csv" \
  --exclude="epoch_*.pth" \
  --exclude="checkpoint_epoch_*.pth" \
  --exclude="*" \
  /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d/results/ \
  n63719vm@csf3.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/results/
```

## File Size Comparison

### Before (Full Transfer):
- Each experiment: ~2-4GB (with all epoch checkpoints)
- 6 experiments: ~12-24GB total

### After (Selective Transfer):
- Each experiment: ~200-400MB (best models + metrics only)
- 6 experiments: ~1.2-2.4GB total

**Transfer time reduction: ~90%**

## What Analysis Scripts Actually Need

All analysis scripts only look for:
- `best_*.pth` - Best model checkpoint
- `*_metrics.json` - Training metrics
- `training_log.json` - Training history

**Never used by analysis:**
- `epoch_*.pth` - Training resumption only
- `checkpoint_epoch_*.pth` - Training resumption only

## Verification Command

After transfer, verify essential files:
```bash
for exp in C3D1_channel_baseline_128 C3D2_channel_mc_dropout_128 C3D3_channel_ensemble_128 C3D4_channel_variational_128 C3D5_channel_swag_128 C3D6_channel_physics_informed_128; do
    echo "=== $exp ==="
    ls -la /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/results/$exp/best_*.pth
    ls -la /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/results/$exp/*_metrics.json
done
```
