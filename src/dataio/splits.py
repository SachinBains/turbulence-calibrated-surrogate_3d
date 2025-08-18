import h5py, numpy as np, json
from pathlib import Path
def tile_indices(shape, block, stride):
  nx,ny,nz=shape; xs=range(0,nx-block+1,stride); ys=range(0,ny-block+1,stride); zs=range(0,nz-block+1,stride)
  starts=[(x,y,z) for x in xs for y in ys for z in zs]; return np.array(starts,dtype=np.int32)
def make_splits_from_h5(cfg):
  p=cfg['paths']; d=cfg['data']; s=cfg.get('splits',{})
  b=int(d['block_size']); st=int(d['stride']); vkey=d.get('velocity_key','velocity')
  with h5py.File(p['velocity_h5'],'r') as f:
    arr=f[vkey]; 
    if arr.shape[-1]==3: nx,ny,nz=arr.shape[:3]
    elif arr.shape[0]==3: nx,ny,nz=arr.shape[1:4]
    else: raise ValueError(f'Unexpected vel shape {arr.shape}')
  starts=tile_indices((nx,ny,nz),b,st); N=len(starts)
  tr=int(s.get('train_ratio',0.7)*N); va=int(s.get('val_ratio',0.15)*N)
  idx=np.arange(N); train_idx=idx[:tr]; val_idx=idx[tr:tr+va]; test_idx=idx[tr+va:]
  out=Path(p['splits_dir']); out.mkdir(parents=True,exist_ok=True)
  np.save(out/'hit_train_idx.npy',train_idx); np.save(out/'hit_val_idx.npy',val_idx); np.save(out/'hit_test_idx.npy',test_idx)
  meta={'shape':[int(nx),int(ny),int(nz)],'block_size':int(b),'stride':int(st),'n_starts':int(N),'train_val_test':[int(len(train_idx)),int(len(val_idx)),int(len(test_idx))]}
  json.dump(meta, open(out/'meta.json','w'), indent=2); print('Wrote splits to', out)
