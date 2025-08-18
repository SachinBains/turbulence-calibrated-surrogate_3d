import torch, torch.nn as nn, torch.fft
class SpectralMixer3D(nn.Module):
  def __init__(self,channels): 
    super().__init__(); self.weight=nn.Parameter(torch.randn(channels,channels)*0.02)
  def forward(self,x):
    X=torch.fft.fftn(x,dim=(-3,-2,-1))
    X=torch.einsum('bc,bcdhw->bdhw', self.weight, X) if X.ndim==5 else X
    y=torch.fft.ifftn(X,dim=(-3,-2,-1)).real; return y
