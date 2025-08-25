import os, glob, json, argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.uq.mc_dropout import mc_predict, gaussian_nll, gaussian_coverage

def load_cfg(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def find_ckpt(cfg):
    exp = cfg['experiment_id']
    res_dir = cfg['paths']['results_dir']
    base = os.path.join(res_dir, exp)
    cand = os.path.join(base, 'best_model.pth')
    if os.path.exists(cand):
        return cand
    gl = sorted(glob.glob(os.path.join(base, 'best_*.pth')))
    if gl:
        return gl[-1]
    raise FileNotFoundError(f"No best checkpoint under {base}")

def build_model(cfg, device):
    m = cfg['model']
    p = cfg.get('uq', {}).get('dropout_p', 0.0)
    net = UNet3D(
        in_channels=m['in_channels'],
        out_channels=m['out_channels'],
        base_ch=m['base_channels'],
        dropout=p,
    ).to(device)
    return net

def load_weights(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)

def build_loader(cfg, split):
    ds = HITDataset(cfg, split=split, eval_mode=True)
    nw = int(cfg['train'].get('num_workers', 0))
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=nw, pin_memory=False)

def concat_targets(loader):
    ys = []
    for _, y in loader:
        ys.append(y.numpy())  # (B,1,D,H,W)
    return np.concatenate(ys, axis=0)

def save_metrics(figdir, resdir, split, y_true, mu, var):
    os.makedirs(resdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    rmse = float(np.sqrt(((y_true - mu) ** 2).mean()))
    nll  = gaussian_nll(y_true, mu, var)
    covs = gaussian_coverage(y_true, mu, var, levels=(0.8, 0.9, 0.95))
    avg_sigma = float(np.sqrt(var).mean())

    metrics = {'rmse_vs_mu': rmse, 'nll': nll, 'avg_sigma': avg_sigma}
    metrics.update(covs)

    with open(os.path.join(resdir, f'mc_metrics_{split}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # simple calibration plot
    xs = np.array([80, 90, 95])
    ys = np.array([metrics['coverage_80']*100, metrics['coverage_90']*100, metrics['coverage_95']*100])
    plt.figure()
    plt.plot(xs, xs, '--', label='ideal')
    plt.plot(xs, ys, 'o-', label='empirical')
    plt.xlabel('Nominal coverage (%)')
    plt.ylabel('Empirical coverage (%)')
    plt.title(f'Calibration ({split})')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(figdir, f'calibration_{split}.png'))
    plt.close()

    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--T', type=int, default=None)
    ap.add_argument('--split', choices=['val','test'], default='test')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp = cfg['experiment_id']
    resdir = os.path.join(cfg['paths']['results_dir'], exp)
    figdir = os.path.join('figures', exp)

    # loader
    loader = build_loader(cfg, args.split)

    # model + weights
    model = build_model(cfg, device)
    ckpt = find_ckpt(cfg)
    load_weights(model, ckpt, device)

    # T from cfg if not provided
    T = args.T if args.T is not None else int(cfg['eval'].get('mc_samples', 20))

    # MC predict
    out = mc_predict(model, loader, device, T=T)
    mu, var = out['mu'], out['var']

    # Save arrays
    np.save(os.path.join(resdir, f'mc_mean_{args.split}.npy'), mu)
    np.save(os.path.join(resdir, f'mc_var_{args.split}.npy'),  var)

    # Ground truth & metrics
    y_true = concat_targets(loader)
    metrics = save_metrics(figdir, resdir, args.split, y_true, mu, var)

    print(f"[MC] split={args.split} T={T} | "
          f"RMSE_vs_mu={metrics['rmse_vs_mu']:.4f}  "
          f"NLL={metrics['nll']:.4f}  "
          f"Cov80={metrics['coverage_80']:.3f}  "
          f"Cov90={metrics['coverage_90']:.3f}  "
          f"Cov95={metrics['coverage_95']:.3f}  "
          f"avg_sigma={metrics['avg_sigma']:.4f}")

if __name__ == '__main__':
    main()
