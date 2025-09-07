# 🔥 URGENT: 8x H100 SXM RunPod Setup - 2.9 Hours to Complete ALL Models

## IMMEDIATE ACTION PLAN:

### Step 1: RunPod Account (3 minutes)
1. **runpod.io** → **"Sign Up"** → **"Continue with GitHub"**
2. Add credit card + **$80 credit** (safety buffer)
3. Verify account

### Step 2: Deploy 8x H100 SXM Pod (2 minutes)
1. **"Pods"** → **"+ GPU Pod"**
2. **Filter**: GPU = "H100 SXM", Quantity = "8x"
3. **Template**: "PyTorch 2.1"
4. **Configuration**:
   - Container: `runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
   - Volume: 250GB
   - Ports: 8888, 22, 6006
5. **DEPLOY NOW**

### Step 3: Environment Setup (5 minutes)
```bash
# In Jupyter terminal
git clone https://github.com/YOUR_USERNAME/turbulence-calibrated-surrogate_3d.git
cd turbulence-calibrated-surrogate_3d

# Install requirements
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify 8 H100s
nvidia-smi
python -c "import torch; print(f'H100s: {torch.cuda.device_count()}')"
```

### Step 4: Upload Dataset (10 minutes)
**FASTEST METHOD**: Compress and upload your dataset
```bash
# On your local machine
tar -czf dataset.tar.gz /path/to/your/channel_flow_1000/
# Upload via Jupyter file browser
```

### Step 5: Execute Training (1 minute)
```bash
# Update batch training script for H100
python scripts/runpod_batch_train.py
```

## EXPECTED TIMELINE (8x H100 SXM):
- **Start**: 6:20 PM
- **C3D3 (Ensemble)**: 6:20-7:20 PM (1 hour)
- **C3D4 (Variational)**: 7:20-8:08 PM (48 min)
- **C3D5 (Deep Ensemble)**: 8:08-8:44 PM (36 min)
- **C3D6 (Evidential)**: 8:44-9:13 PM (29 min)
- **Complete**: 9:15 PM

## COST BREAKDOWN:
- **Runtime**: 2.9 hours
- **Rate**: $21.52/hour
- **Total**: **$62**
- **Per model**: ~$15.50

## SUCCESS CRITERIA:
✅ All 4 models trained and saved
✅ Checkpoints downloaded to local machine
✅ Ready for secondary evaluation
✅ Complete thesis pipeline

---
**🎯 START RUNPOD SIGNUP NOW - EVERY MINUTE COUNTS!**
