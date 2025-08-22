import math
import torch

def evaluate_baseline(model, loader, device, save_dir=None):
    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    nvox = 0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(X)
            mse_sum += torch.sum((pred - y) ** 2).item()
            mae_sum += torch.sum(torch.abs(pred - y)).item()
            nvox += y.numel()
    rmse = math.sqrt(mse_sum / nvox)
    mae  = mae_sum / nvox
    return {"rmse": rmse, "mae": mae}
