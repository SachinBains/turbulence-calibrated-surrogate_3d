import math
import torch
import os

def evaluate_baseline(model, loader, device, save_dir=None, cfg=None):
    import matplotlib.pyplot as plt
    import numpy as np
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    nvox = 0
    first_pred = None
    first_y = None
    with torch.no_grad():
        for i, (X, y) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(X)
            if i == 0:
                first_pred = pred.detach().cpu().numpy()
                first_y = y.detach().cpu().numpy()
            mse_sum += torch.sum((pred - y) ** 2).item()
            mae_sum += torch.sum(torch.abs(pred - y)).item()
            nvox += y.numel()
    rmse = math.sqrt(mse_sum / nvox)
    mae  = mae_sum / nvox
    # Save 3-panel central slice PNG
    if save_dir is not None and first_pred is not None and first_y is not None:
        out_dir = os.path.join('figures', cfg['experiment_id'] if cfg and 'experiment_id' in cfg else 'baseline')
        os.makedirs(out_dir, exist_ok=True)
        yp = first_pred[0,0] if first_pred.shape[1] == 1 else first_pred[0]
        yt = first_y[0,0] if first_y.shape[1] == 1 else first_y[0]
        err = np.abs(yp - yt)
        c = yp.shape[-1] // 2
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(yt[...,c], cmap='viridis'); plt.title('True')
        plt.subplot(1,3,2)
        plt.imshow(yp[...,c], cmap='viridis'); plt.title('Pred')
        plt.subplot(1,3,3)
        plt.imshow(err[...,c], cmap='magma'); plt.title('|Error|')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'central_slice.png'))
        plt.close()
    # Warn if spectral requested but not implemented
    if cfg and 'spectral' in cfg.get('eval',{}).get('metrics',[]) and not hasattr(model, 'spectral_error'):
        print('Warning: spectral_error requested but not implemented.')
    # Skip UQ metrics if uq.method==none
    if cfg and cfg.get('uq',{}).get('method','none')=='none':
        return {"rmse": rmse, "mae": mae}
    # (UQ metrics would go here)
    return {"rmse": rmse, "mae": mae}
