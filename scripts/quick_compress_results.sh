#!/bin/bash

# Quick compress and download essential results
REMOTE_USER="p78669sb"
REMOTE_HOST="csf3.itservices.manchester.ac.uk"
REMOTE_BASE="/mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d"

echo "Creating compressed archive on CSF3..."

# SSH and create compressed archive on remote
ssh ${REMOTE_USER}@${REMOTE_HOST} << 'EOF'
cd /mnt/iusers01/fse-ugpgt01/mace01/p78669sb/artifacts_3d

# Create compressed archive of essential files
tar -czf essential_results.tar.gz \
  checkpoints/C3D1_channel_primary_final_1000/best.pth \
  checkpoints/C3D2_channel_primary_final_1000/best.pth \
  checkpoints/C3D3_channel_primary_final_1000/best.pth \
  checkpoints/C3D6_channel_primary_final_1000/best.pth \
  results/C3D1_channel_primary_final_1000/ \
  results/C3D2_channel_primary_final_1000/ \
  results/C3D3_channel_primary_final_1000/ \
  results/C3D6_channel_primary_final_1000/ \
  figures/ \
  logs/training/C3D*_primary_final* \
  2>/dev/null

echo "Archive created: essential_results.tar.gz"
ls -lh essential_results.tar.gz
EOF

echo "Downloading compressed archive..."
scp ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/essential_results.tar.gz ./

echo "Extracting locally..."
tar -xzf essential_results.tar.gz

echo "Done! Files extracted to current directory"
ls -la
