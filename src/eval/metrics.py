import numpy as np
from scipy.stats import norm
from sklearn.metrics import auc
def rmse(mu,y): return float(np.sqrt(np.mean((mu-y)**2)))
def mae(mu,y): return float(np.mean(np.abs(mu-y)))
def nll_gaussian(mu,var,y):
  var=np.maximum(var,1e-8); return float(0.5*np.mean(np.log(2*np.pi*var)+(y-mu)**2/var))
def crps_gaussian(mu,var,y):
  sigma=np.sqrt(np.maximum(var,1e-8)); z=(y-mu)/sigma
  return float(np.mean(sigma*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/np.sqrt(np.pi))))
def coverage(mu,var,y,alpha=0.9):
  sigma=np.sqrt(np.maximum(var,1e-8))
  lo=mu+norm.ppf((1-alpha)/2.0)*sigma; hi=mu+norm.ppf(1-(1-alpha)/2.0)*sigma
  c=((y>=lo)&(y<=hi)); return float(np.mean(c)), int(c.sum()), int(c.size)
def ece_regression(mu,var,y,bins=10,alpha=0.9):
  sigma=np.sqrt(np.maximum(var,1e-8)); width=2*norm.ppf(1-(1-alpha)/2.0)*sigma
  qs=np.quantile(width,np.linspace(0,1,bins+1)); ece=0.0; n=0
  for i in range(bins):
    m=(width>=qs[i])&(width<(qs[i+1] if i<bins-1 else qs[i+1]))
    if m.sum()==0: continue
    cov,_c,_N=coverage(mu[m],var[m],y[m],alpha); ece+=abs(cov-alpha)*m.sum(); n+=m.sum()
  return float(ece/max(n,1))
def variance_error_correlation(mu,var,y):
  err=np.abs(mu-y).reshape(-1); v=var.reshape(-1)
  if err.size<2: return 0.0
  return float(np.corrcoef(err,v)[0,1])
def aurc(mu,var,y,steps=20):
  err=np.abs(mu-y).reshape(-1); v=var.reshape(-1); order=np.argsort(-v)
  risks=[]; covers=[]; N=err.size
  for k in range(1,steps+1):
    keep=order[k*N//steps:]; risks.append(np.mean(err[keep])); covers.append(keep.size/N)
  return float(auc(covers,risks))
