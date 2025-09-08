# Data Transfer Guide for Cranfield HPC

## Step 1: Create Data Package on CSF3

```bash
# SSH to CSF3
ssh p78669sb@csf3.itservices.manchester.ac.uk

# Navigate to data directory
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/

# Create compressed data package (~12.7GB)
tar -czf cranfield_c3d4_data.tar.gz \
  data_3d/channel_flow_1000/jhtdb_96cubed_production_structured/ \
  turbulence-calibrated-surrogate_3d/splits/meta.json

# Check package size
ls -lh cranfield_c3d4_data.tar.gz
```

## Step 2: Transfer to Cranfield HPC

```bash
# From CSF3, transfer to Cranfield
scp cranfield_c3d4_data.tar.gz [username]@[cranfield_hpc_address]:/home/[username]/

# Alternative: Download to local machine first, then upload
# From local machine:
scp p78669sb@csf3.itservices.manchester.ac.uk:/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/cranfield_c3d4_data.tar.gz ./
scp cranfield_c3d4_data.tar.gz [username]@[cranfield_hpc_address]:/home/[username]/
```

## Step 3: Verify Data Contents

The package contains:
- **1,200 HDF5 files**: `chan96_*.h5` (each ~10.6MB)
- **Dataset splits**: `meta.json` for train/val/test splits
- **Total size**: ~12.7GB compressed

## Step 4: Run Setup Script

```bash
# On Cranfield HPC
cd /home/[username]/turbulence-calibrated-surrogate_3d
chmod +x cranfield/setup_cranfield.sh
./cranfield/setup_cranfield.sh
```

This will automatically extract data to the correct locations and update all config paths.
