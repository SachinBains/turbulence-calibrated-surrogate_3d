import numpy as np
from scipy.stats import norm
def reliability_points(mu,var,y,levels=(0.5,0.8,0.9,0.95)):
  out=[]
  for a in levels:
    sigma=np.sqrt(np.maximum(var,1e-8))
    lo=mu+norm.ppf((1-a)/2.0)*sigma; hi=mu+norm.ppf(1-(1-a)/2.0)*sigma
    cov=((y>=lo)&(y<=hi)).mean(); out.append((a,float(cov)))
  return out
