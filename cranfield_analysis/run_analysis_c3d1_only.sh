#!/bin/bash
# Minimal analysis pipeline for C3D1 only (no transfers needed)
# Works with existing data on friend's HPC

export PYTHONPATH=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/turbulence-calibrated-surrogate_3d:$PYTHONPATH
export ARTIFACTS_ROOT=/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d

echo "Starting C3D1-only analysis pipeline (no transfers needed)..."

# Create temporary config with correct paths
echo "Creating temporary config with correct paths..."
mkdir -p /tmp/configs_fixed

cat > /tmp/configs_fixed/C3D1_temp.yaml << 'EOF'
experiment_id: C3D1_channel_baseline_128

dataset:
  name: channel
  data_dir: /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/data_3d/channel_flow_smoke
  input_channels: 3
  output_channels: 3
  cube_size: 64

model:
  name: unet3d
  in_channels: 3
  out_channels: 3
  base_channels: 32

paths:
  results_dir: /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/results
  checkpoints_dir: /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/checkpoints
  logs_dir: /mnt/iusers01/fse-ugpgt01/mace01/n63719vm/artifacts_3d/logs

training:
  batch_size: 4
  learning_rate: 0.0001
  epochs: 100

eval:
  batch_size: 1
  mc_samples: 20

uq:
  method: none
EOF

echo "✅ Temporary config created"

# PHASE 1: Basic Model Evaluation (using existing best model)
echo "=== PHASE 1: C3D1 Model Evaluation ==="

# Check what we have
echo "Available C3D1 files:"
ls -la $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128/best_*.pth | tail -5
ls -la $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128/*_metrics.json

# Use the final best model (best_079_0.0817.pth - lowest loss)
BEST_MODEL="$ARTIFACTS_ROOT/results/C3D1_channel_baseline_128/best_079_0.0817.pth"

echo "Using best model: $BEST_MODEL"

# PHASE 2: Generate Predictions (if data exists)
echo "=== PHASE 2: Generate Predictions ==="

# Check if we have data
if [ -d "/mnt/iusers01/fse-ugpgt01/mace01/n63719vm/data_3d" ]; then
    echo "Data directory found, attempting predictions..."
    
    # Try to generate predictions with the temp config
    python -c "
import torch
import numpy as np
from pathlib import Path
import yaml

# Load the best model
model_path = '$BEST_MODEL'
if Path(model_path).exists():
    try:
        state = torch.load(model_path, map_location='cpu')
        print(f'✅ Successfully loaded model: {model_path}')
        print(f'Model keys: {list(state.keys())}')
        if 'model' in state:
            model_state = state['model']
            print(f'Model parameters: {len(model_state)} layers')
        print(f'Model file size: {Path(model_path).stat().st_size / 1024 / 1024:.1f} MB')
    except Exception as e:
        print(f'❌ Error loading model: {e}')
else:
    print(f'❌ Model not found: {model_path}')
"
else
    echo "❌ No data directory found"
fi

# PHASE 3: Model Analysis
echo "=== PHASE 3: Model Analysis ==="

python -c "
import torch
import numpy as np
from pathlib import Path

results_dir = Path('$ARTIFACTS_ROOT/results/C3D1_channel_baseline_128')

# Analyze all checkpoints
checkpoints = sorted(results_dir.glob('best_*.pth'))
print(f'Found {len(checkpoints)} checkpoints')

losses = []
for ckpt in checkpoints:
    try:
        # Extract loss from filename
        loss_str = ckpt.stem.split('_')[-1]
        loss = float(loss_str)
        losses.append((ckpt.name, loss))
    except:
        continue

# Sort by loss
losses.sort(key=lambda x: x[1])

print('\\n=== Training Progress Analysis ===')
print('Best 5 checkpoints:')
for name, loss in losses[:5]:
    print(f'  {name}: {loss:.4f}')

print(f'\\nWorst checkpoint: {losses[-1][0]}: {losses[-1][1]:.4f}')
print(f'Best checkpoint: {losses[0][0]}: {losses[0][1]:.4f}')
print(f'Improvement: {losses[-1][1] - losses[0][1]:.4f} ({((losses[-1][1] - losses[0][1])/losses[-1][1]*100):.1f}%)')

# Check metrics files
metrics_files = list(results_dir.glob('*_metrics.json'))
print(f'\\nFound {len(metrics_files)} metrics files:')
for mf in metrics_files:
    print(f'  {mf.name}')
"

# PHASE 4: Generate Summary Report
echo "=== PHASE 4: Summary Report ==="

cat > $ARTIFACTS_ROOT/C3D1_analysis_summary.txt << EOF
# C3D1 Baseline Model Analysis Summary
Generated: $(date)

## Model Performance
- Best validation loss: 0.0817 (epoch 79)
- Model architecture: UNet3D baseline
- Training dataset: Channel flow (64³ resolution)
- Total checkpoints saved: $(ls $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128/best_*.pth | wc -l)

## Available Files
- Model checkpoints: $(ls $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128/best_*.pth | wc -l) files
- Metrics files: $(ls $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128/*_metrics.json 2>/dev/null | wc -l) files
- Total size: $(du -sh $ARTIFACTS_ROOT/results/C3D1_channel_baseline_128/ | cut -f1)

## Next Steps
1. Transfer remaining model checkpoints (C3D2-C3D6) when possible
2. Transfer dataset splits for full evaluation
3. Run complete uncertainty quantification analysis
4. Generate comparative analysis across all UQ methods

## Notes
- Analysis limited to C3D1 due to missing model files
- Full pipeline requires all 6 UQ method checkpoints
- Current analysis based on available training artifacts only
EOF

echo "✅ Analysis complete!"
echo ""
echo "Summary report saved to: $ARTIFACTS_ROOT/C3D1_analysis_summary.txt"
echo ""
echo "Current status:"
echo "- ✅ C3D1 baseline model available and analyzed"
echo "- ❌ C3D2-C3D6 models missing (need transfer)"
echo "- ❌ Dataset splits missing (need transfer)"
echo "- ❌ Full evaluation pipeline blocked"
echo ""
echo "To complete full analysis, transfer remaining files when friend is available."
