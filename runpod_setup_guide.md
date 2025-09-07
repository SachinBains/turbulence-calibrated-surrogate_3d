# 🚀 RunPod 8x A100 SXM Setup Guide - URGENT THESIS DEPLOYMENT

## Step 1: Account Setup (5 minutes)
1. Go to **runpod.io**
2. Click **"Sign Up"** → **"Continue with GitHub"**
3. Authorize GitHub access
4. Add payment method (credit card)
5. Add $60-80 credit (safety buffer)

## Step 2: Find 8x A100 SXM Pod (2 minutes)
1. Click **"Pods"** → **"+ GPU Pod"**
2. **Filter**: GPU Type = "A100 SXM"
3. **Filter**: Quantity = "8x" 
4. **Sort**: By price (should be ~$11.12/hour)
5. Select **US-East** or **US-West** region
6. Click **"Deploy"** on available 8x A100 SXM pod

## Step 3: Pod Configuration (3 minutes)
**Template**: Select **"PyTorch 2.1"**
**Container Image**: `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
**Volume**: 
- **Container Disk**: 50GB
- **Volume Disk**: 200GB (for dataset + results)
**Environment Variables**:
- `PYTHONPATH=/workspace`
- `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
**Ports**: 
- 8888 (Jupyter)
- 22 (SSH)
- 6006 (TensorBoard)

## Step 4: Upload Your Code (10 minutes)
Once pod starts, use Jupyter terminal:

```bash
# Clone your repo
git clone https://github.com/YOUR_USERNAME/turbulence-calibrated-surrogate_3d.git
cd turbulence-calibrated-surrogate_3d

# Install requirements
pip install -r requirements.txt

# Install additional dependencies for multi-GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Step 5: Upload Dataset (15 minutes)
**Option A: Direct Upload (if small)**
- Use Jupyter file browser
- Upload compressed dataset

**Option B: Download from CSF3 (if accessible)**
```bash
# If you have CSF3 access from RunPod
scp -r YOUR_CSF3_USERNAME@csf3.itservices.manchester.ac.uk:/path/to/dataset ./data/
```

**Option C: Use your backup/cloud storage**
- Upload to Google Drive/Dropbox first
- Download to RunPod

## Step 6: Test Multi-GPU Setup (2 minutes)
```bash
# Check GPUs
nvidia-smi

# Test PyTorch multi-GPU
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

## Step 7: Start Batch Training (1 minute)
```bash
# Make scripts executable
chmod +x scripts/*.py

# Start all 5 models in sequence
python scripts/runpod_batch_train.py
```

## Expected Timeline:
- **C3D2 (MC Dropout)**: ~20 minutes
- **C3D5 (Deep Ensemble)**: ~45 minutes  
- **C3D6 (Evidential)**: ~30 minutes
- **C3D3 (Ensemble)**: ~1.5 hours
- **C3D4 (Variational)**: ~1 hour

**Total: ~4 hours, Cost: ~$45**

## Monitoring:
- Watch terminal output for progress
- Use `nvidia-smi` to monitor GPU usage
- Check `/workspace/artifacts_3d/results/` for saved models

## When Complete:
1. Download all results to local machine
2. Stop the pod to avoid charges
3. Run secondary evaluation on completed models

---
**🎯 This setup will complete ALL remaining models before CSF3 maintenance!**
