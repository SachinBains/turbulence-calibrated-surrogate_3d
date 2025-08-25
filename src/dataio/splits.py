import h5py, numpy as np, json
from pathlib import Path

def tile_indices(shape, block, stride):
  nx,ny,nz=shape; xs=range(0,nx-block+1,stride); ys=range(0,ny-block+1,stride); zs=range(0,nz-block+1,stride)
  starts=[(x,y,z) for x in xs for y in ys for z in zs]; return np.array(starts,dtype=np.int32)

def make_id_splits(shape, block_size, stride, seed=42):
  """Create non-overlapping train/val/test splits for in-domain evaluation."""
  np.random.seed(seed)
  starts = tile_indices(shape, block_size, stride)
  N = len(starts)
  
  # Shuffle indices for random splits
  indices = np.arange(N)
  np.random.shuffle(indices)
  
  # Default ratios: 70% train, 15% val, 15% test
  train_end = int(0.7 * N)
  val_end = train_end + int(0.15 * N)
  
  train_idx = indices[:train_end]
  val_idx = indices[train_end:val_end]
  test_idx = indices[val_end:]
  
  return train_idx, val_idx, test_idx

def make_ab_splits(shape, block_size, stride, holdout_frac=0.25):
  """Create A→B splits where Region B is last 25% in x-direction."""
  nx, ny, nz = shape
  
  # Calculate split point in x-direction
  split_x = int(nx * (1 - holdout_frac))
  
  # Generate all possible cube starts
  xs = range(0, nx - block_size + 1, stride)
  ys = range(0, ny - block_size + 1, stride)
  zs = range(0, nz - block_size + 1, stride)
  
  # Separate Region A and Region B based on x-coordinate
  region_a_starts = []
  region_b_starts = []
  
  for i, (x, y, z) in enumerate([(x, y, z) for x in xs for y in ys for z in zs]):
    if x < split_x:
      region_a_starts.append(i)
    else:
      region_b_starts.append(i)
  
  region_a_starts = np.array(region_a_starts, dtype=np.int32)
  region_b_starts = np.array(region_b_starts, dtype=np.int32)
  
  # If no cubes in region B (can happen with small domains),
  # move some cubes from region A to region B
  if len(region_b_starts) == 0:
    # Move last few cubes to test set
    region_b_starts = region_a_starts[-2:]  # Last 2 cubes for test
    region_a_starts = region_a_starts[:-2]  # Remaining cubes for train/val
  
  # Split Region A into train/val (70%/30% of Region A)
  np.random.seed(42)  # Fixed seed for reproducibility
  np.random.shuffle(region_a_starts)
  
  train_end = int(0.7 * len(region_a_starts))
  train_idx = region_a_starts[:train_end]
  val_idx = region_a_starts[train_end:]
  
  # Region B becomes test set
  test_idx = region_b_starts
  
  return train_idx, val_idx, test_idx
def make_splits_from_h5(cfg, mode='id'):
  """Create splits from HDF5 data.
  
  Args:
    cfg: Configuration dictionary
    mode: 'id' for in-domain splits, 'ab' for A→B spatial splits
  """
  p = cfg['paths']
  d = cfg['data']
  s = cfg.get('splits', {})
  
  b = int(d['block_size'])
  st = int(d['stride'])
  vkey = d.get('velocity_key', 'velocity')
  seed = cfg.get('seed', 42)
  
  # Get shape from HDF5 file
  with h5py.File(p['velocity_h5'], 'r') as f:
    arr = f[vkey]
    if arr.shape[-1] == 3:
      nx, ny, nz = arr.shape[:3]
    elif arr.shape[0] == 3:
      nx, ny, nz = arr.shape[1:4]
    else:
      raise ValueError(f'Unexpected vel shape {arr.shape}')
  
  shape = (nx, ny, nz)
  
  # Create splits based on mode
  if mode == 'id':
    train_idx, val_idx, test_idx = make_id_splits(shape, b, st, seed)
    suffix = ''
  elif mode == 'ab':
    train_idx, val_idx, test_idx = make_ab_splits(shape, b, st, holdout_frac=0.25)
    suffix = '_ab'
  else:
    raise ValueError(f"Unknown mode: {mode}. Use 'id' or 'ab'")
  
  # Save splits
  out = Path(p['splits_dir'])
  out.mkdir(parents=True, exist_ok=True)
  
  np.save(out / f'hit{suffix}_train_idx.npy', train_idx)
  np.save(out / f'hit{suffix}_val_idx.npy', val_idx)
  np.save(out / f'hit{suffix}_test_idx.npy', test_idx)
  
  # Calculate total number of possible cube starts
  total_starts = len(tile_indices(shape, b, st))
  
  # Create metadata
  meta = {
    'shape': [int(nx), int(ny), int(nz)],
    'block_size': int(b),
    'stride': int(st),
    'n_starts': int(total_starts),
    'mode': mode,
    'train_val_test': [int(len(train_idx)), int(len(val_idx)), int(len(test_idx))]
  }
  
  if mode == 'ab':
    meta['holdout_frac'] = 0.25
    meta['split_x'] = int(nx * 0.75)
  
  json.dump(meta, open(out / 'meta.json', 'w'), indent=2)
  
  print(f'Wrote {mode.upper()} splits to {out}')
  print(f'  Train: {len(train_idx)} cubes')
  print(f'  Val: {len(val_idx)} cubes')
  print(f'  Test: {len(test_idx)} cubes')
  print(f'  Total possible: {total_starts} cubes')
  
  # Verify no overlap
  all_indices = np.concatenate([train_idx, val_idx, test_idx])
  if len(np.unique(all_indices)) != len(all_indices):
    raise ValueError("Overlapping indices detected in splits!")
  
  print(f'No overlap confirmed')
  
  return train_idx, val_idx, test_idx
