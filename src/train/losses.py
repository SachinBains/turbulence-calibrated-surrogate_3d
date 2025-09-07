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

def variational_elbo_loss(pred, target, model, kl_weight=1e-5):
    """
    Variational ELBO loss = reconstruction loss + weighted KL divergence
    
    Args:
        pred: Model predictions [B, C, D, H, W]
        target: Ground truth [B, C, D, H, W] 
        model: Variational model with raw_kl_divergence() method
        kl_weight: Weight for KL divergence term
    
    Returns:
        total_loss: ELBO loss
        recon_loss: Reconstruction loss component
        kl_loss: KL divergence component
    """
    # Reconstruction loss (MSE between prediction and target)
    recon_loss = nn.MSELoss()(pred, target)
    
    # KL divergence from variational model (unweighted)
    kl_div = model.raw_kl_divergence()
    
    # ELBO = reconstruction_loss + kl_weight * KL_divergence
    total_loss = recon_loss + kl_weight * kl_div
    
    return total_loss, recon_loss, kl_div

def make_loss(cfg):
  spectral=cfg['model'].get('spectral_loss',False); w=float(cfg['model'].get('spectral_weight',0.0))
  name=cfg['loss'].get('name','mse').lower()  # Updated to use cfg['loss']['name']
  
  if name=='variational_elbo':
    kl_weight = cfg['model'].get('kl_weight', 1e-5)
    def fn(pred, y, model=None):
      if model is None:
        raise ValueError("Variational ELBO loss requires model parameter")
      total_loss, recon_loss, kl_div = variational_elbo_loss(pred, y, model, kl_weight)
      return total_loss
    return fn
  elif name=='physics_informed':
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
