import os, glob, json, argparse
import numpy as np
import torch
from src.utils.devices import pick_device
import yaml
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.dataio.channel_dataset import ChannelDataset
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
        m['in_channels'],
        m['out_channels'],
        m['base_channels'],
        p,
    ).to(device)
    return net

def load_weights(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'model' in state:
        model.load_state_dict(state['model'])
    else:
        model.load_state_dict(state)

def build_loader(cfg, split):
    ds = ChannelDataset(cfg, split=split, eval_mode=True)
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
    metrics = {'rmse_vs_mu': float(rmse), 'nll': float(nll), 'avg_sigma': float(np.sqrt(var).mean())}
    metrics['cov80'] = float(gaussian_coverage(y_true, mu, var, 0.8))
    metrics['cov90'] = float(gaussian_coverage(y_true, mu, var, 0.9))
    metrics['cov95'] = float(gaussian_coverage(y_true, mu, var, 0.95))

    with open(os.path.join(resdir, f'mc_metrics_{split}.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # simple calibration plot
    xs = np.array([80, 90, 95])
    ys = np.array([metrics['cov80']*100, metrics['cov90']*100, metrics['cov95']*100])
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
    ap.add_argument('--cuda', action='store_true', help='use CUDA if available')
    ap.add_argument('--conformal', choices=['absolute', 'scaled'], default=None, help='Apply conformal prediction')
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = pick_device(args.cuda)

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
    mu, var, _ = mc_predict(model, loader, device, T=T)

    # Apply conformal prediction if requested
    if args.conformal:
        from src.uq.conformal import apply_conformal, compute_coverage_metrics
        
        # Load conformal calibration
        conformal_path = os.path.join(resdir, f'conformal_mc_{args.conformal}.json')
        if os.path.exists(conformal_path):
            with open(conformal_path, 'r') as f:
                conformal_info = json.load(f)
            
            q_alpha = conformal_info['q_alpha']
            base_sigma = np.sqrt(var)
            
            # Apply conformal intervals
            lo, hi = apply_conformal(mu, base_sigma, q_alpha, mode=args.conformal)
            
            # Save conformal predictions
            np.save(os.path.join(resdir, f'mc_conformal_lo_{args.split}.npy'), lo)
            np.save(os.path.join(resdir, f'mc_conformal_hi_{args.split}.npy'), hi)
            
            print(f"Applied conformal prediction ({args.conformal} mode)")
        else:
            print(f"Warning: Conformal calibration not found: {conformal_path}")

    # Save arrays
    np.save(os.path.join(resdir, f'mc_mean_{args.split}.npy'), mu)
    np.save(os.path.join(resdir, f'mc_var_{args.split}.npy'),  var)

    # Ground truth & metrics
    y_true = concat_targets(loader)
    metrics = save_metrics(figdir, resdir, args.split, y_true, mu, var)
    
    # Add conformal coverage metrics if applied
    if args.conformal and 'lo' in locals():
        from src.uq.conformal import compute_coverage_metrics
        conformal_metrics = compute_coverage_metrics(y_true, lo, hi)
        metrics.update({
            'conformal_coverage': conformal_metrics['coverage'],
            'conformal_width': conformal_metrics['avg_width']
        })
        
        # Re-save metrics with conformal results
        with open(os.path.join(resdir, f'mc_metrics_{args.split}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

    print(f"[MC] split={args.split} T={T} | "
          f"RMSE={metrics['rmse_vs_mu']:.4f}  "
          f"NLL={metrics['nll']:.4f}  "
          f"Cov80={metrics['cov80']:.3f}  "
          f"Cov90={metrics['cov90']:.3f}  "
          f"Cov95={metrics['cov95']:.3f}  "
          f"avg_sigma={metrics['avg_sigma']:.4f}")

if __name__ == '__main__':
    main()
