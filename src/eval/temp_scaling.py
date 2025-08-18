import numpy as np
from scipy.optimize import minimize
from .metrics import nll_gaussian
class TemperatureScaler:
  def __init__(self): self.t=1.0
  def fit(self, val_loader, model, logger):
    import torch
    model.eval(); mus=[]; vars=[]; ys=[]
    with torch.no_grad():
      for xb,yb in val_loader:
        xb=xb.cuda() if torch.cuda.is_available() else xb
        pred=model(xb); mu=pred.cpu().numpy(); var=np.full_like(mu,1e-2,dtype=np.float32)
        mus.append(mu); vars.append(var); ys.append(yb.numpy())
    mu=np.concatenate(mus,0); var=np.concatenate(vars,0); y=np.concatenate(ys,0)
    def obj(logt): t=np.exp(logt[0]); return nll_gaussian(mu, var*(t**2), y)
    res=minimize(obj,x0=[0.0],method='Nelder-Mead'); self.t=float(np.exp(res.x[0])); logger.info(f'Temperature {self.t:.3f}'); return self.t
