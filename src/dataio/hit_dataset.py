import h5py, numpy as np, torch, json
from torch.utils.data import Dataset
from pathlib import Path

def _read_h5(path, key):
  with h5py.File(path,'r') as f:
    if key not in f: raise KeyError(f"Key '{key}' not in {path}. Available: {list(f.keys())[:8]}")
    return f[key][...]

def open_h5(path):
  """Open HDF5 file and return file handle for inspection."""
  return h5py.File(path, 'r')

def inspect_shapes(cfg):
  """Inspect HDF5 files and return shape information."""
  paths = cfg['paths']
  data = cfg['data']
  
  results = {}
  
  with h5py.File(paths['velocity_h5'], 'r') as f:
    vel_key = data['velocity_key']
    results['velocity'] = {
      'file_keys': list(f.keys()),
      'shape': f[vel_key].shape,
      'dtype': str(f[vel_key].dtype),
      'min': float(f[vel_key][...].min()),
      'max': float(f[vel_key][...].max())
    }
  
  with h5py.File(paths['pressure_h5'], 'r') as f:
    prs_key = data['pressure_key']
    results['pressure'] = {
      'file_keys': list(f.keys()),
      'shape': f[prs_key].shape,
      'dtype': str(f[prs_key].dtype),
      'min': float(f[prs_key][...].min()),
      'max': float(f[prs_key][...].max())
    }
  
  return results

def compute_channel_stats(cfg, cube_size=64, max_samples=100):
  """Compute per-channel mean/std statistics on training data only."""
  paths = cfg['paths']
  data = cfg['data']
  
  # Load velocity and pressure data
  vel = _read_h5(paths['velocity_h5'], data['velocity_key'])
  prs = _read_h5(paths['pressure_h5'], data['pressure_key'])
  
  # Handle shape normalization (same logic as HITDataset)
  if vel.shape[-1] == 3:
    pass  # Already in correct format (H, W, D, 3)
  elif vel.shape[0] == 3:
    vel = np.moveaxis(vel, 0, -1)  # Move channels to last dim
  else:
    raise ValueError(f'Unexpected velocity shape {vel.shape}')
  
  if prs.ndim == 4 and prs.shape[-1] == 1:
    pass  # Already correct (H, W, D, 1)
  elif prs.ndim == 3:
    prs = prs[..., None]  # Add channel dim
  elif prs.ndim == 4 and prs.shape[0] == 1:
    prs = np.moveaxis(prs, 0, -1)  # Move channels to last dim
  else:
    raise ValueError(f'Unexpected pressure shape {prs.shape}')
  
  # Load training indices
  splits_dir = Path(paths['splits_dir'])
  train_idx = np.load(splits_dir / 'hit_train_idx.npy')
  meta = json.load(open(splits_dir / 'meta.json'))
  
  # Generate cube starts (same logic as HITDataset._starts)
  nx, ny, nz = meta['shape']
  b = meta['block_size']
  s = meta['stride']
  xs = range(0, nx - b + 1, s)
  ys = range(0, ny - b + 1, s)
  zs = range(0, nz - b + 1, s)
  all_starts = np.array([(x, y, z) for x in xs for y in ys for z in zs], dtype=np.int32)
  
  # Get training cube starts
  train_starts = all_starts[train_idx]
  
  # Sample cubes for statistics (to avoid memory issues)
  step = max(1, len(train_starts) // max_samples)
  sample_starts = train_starts[::step]
  
  print(f"Computing stats from {len(sample_starts)} training cubes...")
  
  # Collect samples
  vel_samples = []
  prs_samples = []
  
  for i, (x, y, z) in enumerate(sample_starts):
    if i % 20 == 0:
      print(f"Processing cube {i+1}/{len(sample_starts)}")
    
    vel_cube = vel[x:x+b, y:y+b, z:z+b, :].astype('float32')
    prs_cube = prs[x:x+b, y:y+b, z:z+b, :].astype('float32')
    
    vel_samples.append(vel_cube.reshape(-1, vel_cube.shape[-1]))
    prs_samples.append(prs_cube.reshape(-1, prs_cube.shape[-1]))
  
  # Concatenate all samples
  vel_data = np.concatenate(vel_samples, axis=0)
  prs_data = np.concatenate(prs_samples, axis=0)
  
  # Compute statistics
  vel_mean = vel_data.mean(axis=0)
  vel_std = vel_data.std(axis=0) + 1e-8  # Add small epsilon for numerical stability
  prs_mean = prs_data.mean(axis=0)
  prs_std = prs_data.std(axis=0) + 1e-8
  
  stats = {
    'velocity': {
      'mean': vel_mean,
      'std': vel_std,
      'shape': vel_data.shape,
      'channels': ['u', 'v', 'w']
    },
    'pressure': {
      'mean': prs_mean,
      'std': prs_std,
      'shape': prs_data.shape,
      'channels': ['p']
    },
    'num_samples': len(sample_starts),
    'cube_size': b
  }
  
  return stats

def save_stats(stats, output_path):
  """Save statistics to .npz file."""
  output_path = Path(output_path)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  
  # Prepare data for saving
  save_dict = {
    'vel_mean': stats['velocity']['mean'],
    'vel_std': stats['velocity']['std'],
    'prs_mean': stats['pressure']['mean'],
    'prs_std': stats['pressure']['std'],
    'num_samples': stats['num_samples'],
    'cube_size': stats['cube_size']
  }
  
  np.savez(output_path, **save_dict)
  print(f"Statistics saved to {output_path}")
  
  return save_dict
class HITDataset(Dataset):
  def __init__(self, cfg, split='train', eval_mode=False):
    self.cfg=cfg; self.split=split; p=cfg['paths']; d=cfg['data']
    self.vel=_read_h5(p['velocity_h5'], d['velocity_key']); self.prs=_read_h5(p['pressure_h5'], d['pressure_key'])
    if self.vel.shape[-1]==3: pass
    elif self.vel.shape[0]==3: self.vel=np.moveaxis(self.vel,0,-1)
    else: raise ValueError(f'Unexpected velocity shape {self.vel.shape}')
    if self.prs.ndim==4 and self.prs.shape[-1]==1: pass
    elif self.prs.ndim==3: self.prs=self.prs[...,None]
    elif self.prs.ndim==4 and self.prs.shape[0]==1: self.prs=np.moveaxis(self.prs,0,-1)
    else: raise ValueError(f'Unexpected pressure shape {self.prs.shape}')
    
    # Support split tags (e.g. 'ab' for A→B)
    tag = cfg.get('splits', {}).get('tag', '').strip()
    sdir=Path(p['splits_dir']); meta=json.load(open(sdir/'meta.json'))
    
    def idx_name(split):
        return f"hit_{tag+'_' if tag else ''}{split}_idx.npy"
    
    # Load index with explicit check
    idx_path = sdir / idx_name(split)
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing split file: {idx_path}")
    idx = np.load(idx_path)
    
    self.block=int(d['block_size']); self.starts=self._starts(meta)[idx]; self.eval_mode=eval_mode; self.eval_stride=int(d.get('eval_stride',1))
    
    # For validation in A→B, fall back gracefully if ab_val missing
    if tag == 'ab' and split == 'val':
        try:
            t_idx_path = sdir / idx_name('train')
            t_idx = np.load(t_idx_path)
        except FileNotFoundError:
            t_idx_path = sdir / 'hit_train_idx.npy'
            t_idx = np.load(t_idx_path)
    else:
        t_idx = np.load(sdir / idx_name('train'))
    
    self.stats=self._stats(self._starts(meta)[t_idx])
  def _starts(self, meta):
    nx,ny,nz=meta['shape']; b=meta['block_size']; s=meta['stride']
    xs=range(0,nx-b+1,s); ys=range(0,ny-b+1,s); zs=range(0,nz-b+1,s)
    return np.array([(x,y,z) for x in xs for y in ys for z in zs], dtype=np.int32)
  def _stats(self, starts):
    step=max(1,len(starts)//64); sample=starts[::step]; b=self.block; vs=[]; ps=[]
    for (x,y,z) in sample:
      vs.append(self.vel[x:x+b,y:y+b,z:z+b,:].reshape(-1,3)); ps.append(self.prs[x:x+b,y:y+b,z:z+b,:].reshape(-1,1))
    v=np.concatenate(vs,0); p=np.concatenate(ps,0); return {'vel':(v.mean(0), v.std(0)+1e-8), 'prs':(p.mean(0), p.std(0)+1e-8)}
  def __len__(self): return len(self.starts[::self.eval_stride]) if self.eval_mode else len(self.starts)
  def __getitem__(self,i):
    i=i if not self.eval_mode else i*self.eval_stride; x,y,z=self.starts[i]; b=self.block
    v=self.vel[x:x+b,y:y+b,z:z+b,:].astype('float32'); p=self.prs[x:x+b,y:y+b,z:z+b,:].astype('float32')
    v=(v-self.stats['vel'][0])/(self.stats['vel'][1]); p=(p-self.stats['prs'][0])/(self.stats['prs'][1])
    v=np.moveaxis(v,-1,0); p=np.moveaxis(p,-1,0); return torch.from_numpy(v), torch.from_numpy(p)
