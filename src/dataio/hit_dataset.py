import h5py, numpy as np, torch, json
from torch.utils.data import Dataset
from pathlib import Path
def _read_h5(path, key):
  with h5py.File(path,'r') as f:
    if key not in f: raise KeyError(f"Key '{key}' not in {path}. Available: {list(f.keys())[:8]}")
    return f[key][...]
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
    sdir=Path(p['splits_dir']); idx=np.load(sdir/f'hit_{split}_idx.npy'); meta=json.load(open(sdir/'meta.json'))
    self.block=int(d['block_size']); self.starts=self._starts(meta)[idx]; self.eval_mode=eval_mode; self.eval_stride=int(d.get('eval_stride',1))
    t_idx=np.load(sdir/'hit_train_idx.npy'); self.stats=self._stats(self._starts(meta)[t_idx])
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
