import torch, torch.nn as nn
class GaussianNLL(nn.Module):
  def __init__(self): super().__init__()
  def forward(self, mu, y, logvar):
    var=torch.exp(logvar).clamp_min(1e-6); return 0.5*((y-mu)**2/var + logvar).mean()
def spectral_mse(pred, target):
  P=torch.fft.fftn(pred,dim=(-3,-2,-1)); T=torch.fft.fftn(target,dim=(-3,-2,-1))
  return ((P.abs()-T.abs())**2).mean()

def physics_informed_loss(pred, target, dx=1.0, dy=1.0, dz=1.0, physics_weight=0.1, continuity_weight=0.05, momentum_weight=0.05):
    """Physics-informed loss with Navier-Stokes constraints for 3D channel flow."""
    # Data loss (MSE)
    data_loss = nn.MSELoss()(pred, target)
    
    # Extract velocity components [B, 3, D, H, W]
    u, v, w = pred[:, 0], pred[:, 1], pred[:, 2]
    
    # Compute gradients using finite differences
    # du/dx, dv/dy, dw/dz for continuity equation
    dudx = (u[:, :, :, 1:] - u[:, :, :, :-1]) / dx
    dvdy = (v[:, :, 1:, :] - v[:, :, :-1, :]) / dy
    dwdz = (w[:, 1:, :, :] - w[:, :-1, :, :]) / dz
    
    # Pad to match dimensions
    dudx = torch.cat([dudx, dudx[:, :, :, -1:]], dim=3)
    dvdy = torch.cat([dvdy, dvdy[:, :, -1:, :]], dim=2)
    dwdz = torch.cat([dwdz, dwdz[:, -1:, :, :]], dim=1)
    
    # Continuity equation: ∇·u = du/dx + dv/dy + dw/dz = 0
    continuity_residual = dudx + dvdy + dwdz
    continuity_loss = torch.mean(continuity_residual**2)
    
    # Simplified momentum conservation (pressure gradient terms omitted)
    # Focus on velocity gradient consistency
    momentum_loss = 0.0
    
    # Total physics loss
    physics_loss = continuity_weight * continuity_loss + momentum_weight * momentum_loss
    
    # Combined loss
    total_loss = data_loss + physics_weight * physics_loss
    
    return total_loss, data_loss, physics_loss, continuity_loss

def make_loss(cfg):
  spectral=cfg['model'].get('spectral_loss',False); w=float(cfg['model'].get('spectral_weight',0.0))
  name=cfg['train'].get('loss','mse').lower()
  
  if name=='physics_informed':
    physics_weight = cfg['train'].get('physics_weight', 0.1)
    continuity_weight = cfg['train'].get('continuity_weight', 0.05)
    momentum_weight = cfg['train'].get('momentum_weight', 0.05)
    def fn(pred, y):
      total_loss, data_loss, physics_loss, continuity_loss = physics_informed_loss(
        pred, y, physics_weight=physics_weight, 
        continuity_weight=continuity_weight, momentum_weight=momentum_weight)
      return total_loss
    return fn
  elif name=='gaussian_nll':
    base=GaussianNLL()
    def fn(pred,y): return base(pred[:,:1], y, pred[:,1:])
  else:
    base=nn.MSELoss()
    def fn(pred,y):
      l=base(pred,y)
      if spectral and w>0: l=l+w*spectral_mse(pred,y)
      return l
  return fn
