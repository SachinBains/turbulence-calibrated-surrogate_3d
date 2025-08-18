import numpy as np
def radial_spectrum_3d(field):
  F=np.fft.fftn(field); E=np.abs(F)**2
  nx,ny,nz=field.shape; kx=np.fft.fftfreq(nx); ky=np.fft.fftfreq(ny); kz=np.fft.fftfreq(nz)
  KX,KY,KZ=np.meshgrid(kx,ky,kz,indexing='ij'); kr=np.sqrt(KX**2+KY**2+KZ**2)
  bins=(kr*min(nx,ny,nz)).astype(int); kmax=bins.max()
  spec=np.zeros(kmax+1); 
  for i in range(kmax+1):
    m=(bins==i); spec[i]=E[m].mean() if m.any() else 0.0
  return spec
def spectral_error(mu,y):
  if mu.ndim==5: mu=mu[0]; y=y[0]
  if mu.ndim==4:
    errs=[]
    for c in range(mu.shape[0]):
      s1=radial_spectrum_3d(mu[c]); s2=radial_spectrum_3d(y[c]); n=min(len(s1),len(s2))
      errs.append(np.mean(np.abs(s1[:n]-s2[:n])/(np.abs(s2[:n])+1e-8)))
    return float(np.mean(errs))
  s1=radial_spectrum_3d(mu); s2=radial_spectrum_3d(y); n=min(len(s1),len(s2))
  return float(np.mean(np.abs(s1[:n]-s2[:n])/(np.abs(s2[:n])+1e-8)))
