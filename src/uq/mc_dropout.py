import torch
import numpy as np
from torch import nn
from tqdm import tqdm

def mc_predict(model, loader, device, T=32, eps=1e-6):
    model.eval()
    # Enable dropout during MC: set train mode for dropout, eval for batchnorm (if any)
    def apply_dropout(m):
        if isinstance(m, nn.Dropout3d):
            m.train()
    model.apply(apply_dropout)
    ys, mus, vars = [], [], []
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="MC Predict"):
            xb = xb.to(device)
            yb = yb.to(device)
            mc_samples = []
            for _ in range(T):
                out = model(xb)
                mc_samples.append(out.detach().cpu().numpy())
            mc_samples = np.stack(mc_samples, axis=0)  # [T, B, ...]
            mu = np.mean(mc_samples, axis=0)
            var = np.var(mc_samples, axis=0) + eps
            mus.append(mu)
            vars.append(var)
            ys.append(yb.detach().cpu().numpy())
    mu = np.concatenate(mus, axis=0)
    var = np.concatenate(vars, axis=0)
    y = np.concatenate(ys, axis=0)
    return mu, var, y

def gaussian_nll(y_true, mu, var):
    # NLL for Gaussian: 0.5*log(2pi*var) + (y-mu)^2/(2*var)
    nll = 0.5 * np.log(2 * np.pi * var) + 0.5 * (y_true - mu) ** 2 / var
    return np.mean(nll)

def gaussian_coverage(y_true, mu, var, alpha):
    # alpha: e.g. 0.8, 0.9, 0.95
    from scipy.stats import norm
    z = norm.ppf(0.5 + alpha / 2)
    lower = mu - z * np.sqrt(var)
    upper = mu + z * np.sqrt(var)
    covered = (y_true >= lower) & (y_true <= upper)
    return np.mean(covered)
