import torch, torch.nn as nn
class GaussianNLL(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, mu, y, logvar):
    var=torch.exp(logvar).clamp_min(1e-6); return 0.5*((y-mu)**2/var + logvar).mean()
def spectral_mse(pred, target):
  P=torch.fft.fftn(pred,dim=(-3,-2,-1)); T=torch.fft.fftn(target,dim=(-3,-2,-1))
  return ((P.abs()-T.abs())**2).mean()
def make_loss(cfg):
  spectral=cfg['model'].get('spectral_loss',False); w=float(cfg['model'].get('spectral_weight',0.0))
  name=cfg['train'].get('loss','mse').lower()
  if name=='gaussian_nll':
    base=GaussianNLL()
    def fn(pred,y): return base(pred[:,:1], y, pred[:,1:])
  else:
    base=nn.MSELoss()
    def fn(pred,y):
      l=base(pred,y)
      if spectral and w>0: l=l+w*spectral_mse(pred,y)
      return l
  return fn
