#!/bin/bash

# Essential Results Download Script
# Downloads only key files from CSF3 results directories to local machine
# Avoids downloading large prediction arrays and focuses on analysis outputs

# Set paths
REMOTE_USER="p78669sb"
REMOTE_HOST="csf3.itservices.manchester.ac.uk"
REMOTE_ARTIFACTS="/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"
LOCAL_RESULTS="./downloaded_results"

# Create local directory structure
mkdir -p $LOCAL_RESULTS

echo "Downloading essential results from CSF3..."
echo "Target models: C3D1, C3D2, C3D3, C3D6 (primary_final)"

# Essential files to download from each model:
# 1. Training logs and metrics
# 2. Final model checkpoints (best.pth only)
# 3. Evaluation metrics and reports
# 4. Calibration plots and analysis
# 5. Physics validation results
# 6. Summary statistics (not raw predictions)

MODELS=("C3D1_channel_primary_final_1000" "C3D2_channel_primary_final_1000" "C3D3_channel_primary_final_1000" "C3D6_channel_primary_final_1000")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "=== Downloading essential files for $MODEL ==="
    
    # Create model directory
    mkdir -p "$LOCAL_RESULTS/$MODEL"
    
    # 1. Training logs and metrics
    echo "Downloading training logs..."
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/logs/training/${MODEL}*" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No training logs found"
    
    # 2. Best model checkpoint only (not all epochs)
    echo "Downloading best checkpoint..."
    scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/checkpoints/$MODEL/best.pth" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No best checkpoint found"
    
    # 3. Evaluation results (metrics, not raw predictions)
    echo "Downloading evaluation metrics..."
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/metrics/" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No metrics found"
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/reports/" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No reports found"
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/analysis/" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No analysis found"
    
    # 4. Calibration results
    echo "Downloading calibration analysis..."
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/calibration/" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No calibration found"
    
    # 5. Physics validation results
    echo "Downloading physics validation..."
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/physics/" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No physics validation found"
    
    # 6. Figures and plots (compressed)
    echo "Downloading figures..."
    scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/figures/$MODEL/" "$LOCAL_RESULTS/figures/" 2>/dev/null || echo "No figures found"
    
    # 7. Configuration files used
    echo "Downloading config..."
    scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/config.yaml" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No config found"
    
    # 8. Summary statistics (JSON/CSV files only)
    echo "Downloading summary statistics..."
    scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/*.json" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No JSON summaries found"
    scp "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/$MODEL/*.csv" "$LOCAL_RESULTS/$MODEL/" 2>/dev/null || echo "No CSV summaries found"
    
done

# Download global comparison results
echo ""
echo "=== Downloading global comparison results ==="
mkdir -p "$LOCAL_RESULTS/comparison"
scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/comparison/" "$LOCAL_RESULTS/" 2>/dev/null || echo "No comparison results found"
scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/figures/comparison/" "$LOCAL_RESULTS/figures/" 2>/dev/null || echo "No comparison figures found"

# Download aggregated analysis
echo "Downloading aggregated analysis..."
scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/results/summary/" "$LOCAL_RESULTS/" 2>/dev/null || echo "No summary results found"

# Download calibration analysis for all models
echo "Downloading calibration analysis..."
scp -r "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ARTIFACTS/figures/calibration_primary/" "$LOCAL_RESULTS/figures/" 2>/dev/null || echo "No calibration figures found"

echo ""
echo "=== Download Summary ==="
echo "Essential results downloaded to: $LOCAL_RESULTS"
echo ""
echo "EXCLUDED (too large):"
echo "- Raw prediction arrays (predictions/)"
echo "- All model checkpoints except best.pth"
echo "- Intermediate training checkpoints"
echo "- Large HDF5 prediction files"
echo ""
echo "INCLUDED (essential for analysis):"
echo "- Training logs and loss curves"
echo "- Best model checkpoints"
echo "- Evaluation metrics and reports"
echo "- Calibration analysis and plots"
echo "- Physics validation results"
echo "- Summary statistics (JSON/CSV)"
echo "- All figures and visualizations"
echo "- Configuration files"
echo ""

# Calculate download size
echo "Calculating download size..."
du -sh "$LOCAL_RESULTS" 2>/dev/null || echo "Size calculation unavailable"

echo "Download complete!"
