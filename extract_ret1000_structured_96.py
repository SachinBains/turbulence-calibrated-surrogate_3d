import pyJHTDB
import numpy as np
import h5py
import math
import os
import time
from datetime import datetime

# Create organized folder structure
os.makedirs('jhtdb_96cubed_production_structured_2', exist_ok=True)

# Set token
pyJHTDB.dbinfo.auth_token = "uk.ac.manchester.postgrad.sachin.bains-df182d45"

# Initialize libJHTDB
lTDB = pyJHTDB.libJHTDB()
lTDB.initialize()

print(f"Starting structured 96³ velocity cube downloads at {datetime.now()}")

# Y+ band sampling strategy
y_plus_bands = [
    (5, 30),      # Band 1: Near-wall
    (30, 100),    # Band 2: Buffer layer  
    (100, 300),   # Band 3: Log layer
    (300, 800)    # Band 4: Outer layer
]

# Temporal sampling: 160 time points (every 25th frame)
time_points = [0.002 * (i * 25) for i in range(160)]

# Target: 400 cubes per band = 1600 total
# 160 time points × 2.5 spatial samples per time = 400 per band
cubes_per_time_per_band = 2.5  # Will alternate between 2 and 3

start_time = time.time()
successful_downloads = 0
failed_downloads = 0
cube_idx = 0

for band_idx, (y_min, y_max) in enumerate(y_plus_bands):
    print(f"\n🎯 Processing Y+ Band {band_idx+1}: [{y_min}-{y_max}]")
    band_cubes = 0
    
    for time_idx, time_val in enumerate(time_points):
        # Alternate between 2 and 3 spatial samples per time point
        n_spatial = 3 if time_idx % 2 == 0 else 2
        
        for spatial_idx in range(n_spatial):
            cube_start = time.time()
            cube_idx += 1
            
            print(f"Band {band_idx+1}, Time {time_idx+1}/160, Spatial {spatial_idx+1}/{n_spatial} (Cube {cube_idx}/1600)")
            
            try:
                cube_size = 96
                n_points = cube_size**3
                
                # Y+ constrained sampling
                y_plus_target = np.random.uniform(y_min, y_max)
                y_center = (y_plus_target / 1000 - 0.5) * 2 * math.pi
                
                # Random X and Z sampling
                x_center = 2*math.pi*np.random.random()
                z_center = 2*math.pi*np.random.random()
                
                cube_extent = 0.3
                x = np.linspace(x_center - cube_extent/2, x_center + cube_extent/2, cube_size)
                y = np.linspace(y_center - cube_extent/2, y_center + cube_extent/2, cube_size)
                z = np.linspace(z_center - cube_extent/2, z_center + cube_extent/2, cube_size)
                
                # Generate 96³ point grid
                points = np.empty((n_points, 3), dtype='float32')
                idx = 0
                for i in range(cube_size):
                    for j in range(cube_size):
                        for k in range(cube_size):
                            points[idx] = [x[i], y[j], z[k]]
                            idx += 1
                
                # Extract velocity data
                velocity_data = lTDB.getData(
                    time_val, 
                    points,
                    sinterp=6,
                    tinterp=0,
                    data_set='channel',
                    getFunction='getVelocity'
                )
                
                velocity_cube = velocity_data.reshape(cube_size, cube_size, cube_size, 3)
                
                # Basic validation - only reject if ALL values are exactly -999
                if np.all(velocity_cube == -999.0):
                    print(f"  ✗ All velocity values are -999, skipping cube")
                    raise ValueError("Complete data failure")
                
                # Log velocity statistics for monitoring
                vel_mean = np.mean(velocity_cube)
                vel_std = np.std(velocity_cube)
                print(f"  📊 Velocity stats: mean={vel_mean:.3f}, std={vel_std:.3f}")
                
                # Save with structured naming
                filepath = f'jhtdb_96cubed_production_structured_2/ret1000_cube_{cube_idx}.h5'
                with h5py.File(filepath, 'w') as f:
                    f['velocity'] = velocity_cube.astype(np.float32)
                    f['y_plus'] = y_plus_target
                    f['y_plus_band'] = band_idx + 1
                    f['y_plus_range'] = [y_min, y_max]
                    f['center_position'] = [x_center, y_center, z_center]
                    f['time'] = time_val
                    f['time_index'] = time_idx
                    f['spatial_index'] = spatial_idx
                    f['dataset'] = 'channel'
                    f['resolution'] = 96
                    f['extraction_method'] = 'structured_sampling'
                
                cube_time = time.time() - cube_start
                successful_downloads += 1
                band_cubes += 1
                
                print(f"  ✓ Y+={y_plus_target:.1f} ({cube_time:.1f}s)")
                
                # Progress reporting every 50 cubes
                if cube_idx % 50 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / cube_idx
                    remaining = (1600 - cube_idx) * avg_time
                    print(f"\n📊 Progress: {cube_idx}/1600 ({100*cube_idx/1600:.1f}%)")
                    print(f"   Average: {avg_time:.1f}s/cube")
                    print(f"   Estimated remaining: {remaining/3600:.1f} hours")
                    print(f"   Band {band_idx+1} progress: {band_cubes}/400\n")
                
            except Exception as e:
                failed_downloads += 1
                print(f"  ✗ Failed: {e}")
    
    print(f"✅ Band {band_idx+1} complete: {band_cubes} cubes")

total_time = time.time() - start_time
print(f"\n🎉 Structured download complete!")
print(f"   Total time: {total_time/3600:.1f} hours")
print(f"   Successful: {successful_downloads}/1600")
print(f"   Failed: {failed_downloads}/1600")
print(f"   Y+ Band distribution:")
for i, (y_min, y_max) in enumerate(y_plus_bands):
    print(f"     Band {i+1} [{y_min}-{y_max}]: ~400 cubes")
print(f"   All cubes saved in: jhtdb_96cubed_production_structured_2/")