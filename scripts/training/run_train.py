import argparse, torch, subprocess, json, sys
from pathlib import Path
from src.utils.seeding import seed_all
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.manifest import append_manifest_row
from src.utils.devices import pick_device
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
from src.train.losses import make_loss
from src.train.trainer import train_loop
from torch.utils.data import DataLoader

def get_git_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"

def main(cfg_path, seed, resume, cuda):
    cfg = load_config(cfg_path)
    seed_val = seed or cfg.get('seed', 42)
    seed_all(seed_val)
    log = get_logger()
    exp_id = cfg.get('experiment_id', 'EXPERIMENT')
    out = Path(cfg['paths']['results_dir']) / exp_id
    out.mkdir(parents=True, exist_ok=True)
    tr = ChannelDataset(cfg, 'train')
    va = ChannelDataset(cfg, 'val')
    tl = DataLoader(tr, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=cfg['train']['num_workers'])
    vl = DataLoader(va, batch_size=1, shuffle=False, num_workers=cfg['train']['num_workers'])
    # --- Build model ---
    mcfg = cfg['model']
    uq_method = cfg.get('uq', {}).get('method', 'none')
    dropout_p = cfg.get('uq', {}).get('dropout_p', 0.0)
    
    log.info(f"Building model with UQ method: {uq_method}, dropout_p: {dropout_p}")

    if uq_method == 'variational':
        from src.models.variational_unet3d import VariationalUNet3D
        kl_weight = mcfg.get('kl_weight', 1e-5)
        net = VariationalUNet3D(
            mcfg['in_channels'],
            mcfg['out_channels'],
            mcfg['base_channels'],
            dropout_p,
            kl_weight=kl_weight
        )
        log.info(f"Variational UNet3D created with kl_weight={kl_weight}")
    else:
        from src.models.unet3d import UNet3D
        net = UNet3D(
            mcfg['in_channels'],
            mcfg['out_channels'],
            mcfg['base_channels'],
            dropout_p,
        )
        
        if uq_method == 'mc_dropout':
            net.enable_mc_dropout(p=dropout_p)
            log.info(f"MC Dropout enabled with p={dropout_p}")
        else:
            log.info(f"Using baseline model (method: {uq_method})")
    
    device = pick_device(cuda)
    net = net.to(device)
    crit = make_loss(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['train']['amp']))
    git_hash = get_git_hash()
    print(f"Git hash: {git_hash}")
    print(f"Seed: {seed_val}")
    run_info = {
        "git_hash": git_hash,
        "seed": seed_val,
        "cmd": ' '.join(sys.argv),
        "config": cfg_path,
        "resume": resume
    }
    with open(out / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)
    if uq_method == 'variational':
        from src.train.variational_trainer import variational_train_loop
        best = variational_train_loop(cfg, net, crit, opt, scaler, tl, vl, out, log, resume_path=resume, device=device)
    elif 'swag' in cfg['experiment_id'].lower():
        from src.train.swag_trainer import swag_train_loop
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['train']['epochs'])
        best = swag_train_loop(net, tl, vl, opt, scheduler, out, log, cfg, resume_path=resume, device=device)
    elif cfg['train'].get('loss') == 'physics_informed':
        from src.train.physics_trainer import physics_train_loop
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['train']['epochs'])
        best = physics_train_loop(net, tl, vl, opt, scheduler, out, log, cfg, resume_path=resume, device=device)
    else:
        best = train_loop(cfg, net, crit, opt, scaler, tl, vl, out, log, resume_path=resume, device=device)
    append_manifest_row(cfg_path, seed_val, str(out))
    log.info(f'Best: {best}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--resume', type=str, default=None)
    ap.add_argument('--cuda', action='store_true', help='use CUDA if available')
    a = ap.parse_args()
    main(a.config, a.seed, a.resume, a.cuda)
