import torch
from statistics import mean
from pathlib import Path
def save_ckpt(path, model, epoch, val_loss):
  torch.save({'epoch':epoch,'model':model.state_dict(),'val_loss':float(val_loss)}, path)
def train_loop(cfg, net, criterion, opt, scaler, train_loader, val_loader, results_dir, logger):
    import csv
    from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
    epochs = cfg['train']['epochs']
    patience = cfg['train'].get('early_stop_patience', 10)
    scheduler_type = cfg['train'].get('lr_scheduler', 'cosine')
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(opt, T_max=epochs)
    else:
        scheduler = StepLR(opt, step_size=cfg['train'].get('lr_step', 10), gamma=cfg['train'].get('lr_gamma', 0.5))
    best = float('inf'); best_path = None; bad = 0
    csv_path = Path(results_dir) / 'loss_history.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for ep in range(1, epochs + 1):
            net.train(); tr = []
            for xb, yb in train_loader:
                xb = xb.cuda() if torch.cuda.is_available() else xb
                yb = yb.cuda() if torch.cuda.is_available() else yb
                opt.zero_grad()
                with torch.cuda.amp.autocast(enabled=cfg['train']['amp']):
                    pred = net(xb); loss = criterion(pred, yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tr.append(float(loss.detach().cpu()))
            scheduler.step()
            net.eval(); vl = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.cuda() if torch.cuda.is_available() else xb
                    yb = yb.cuda() if torch.cuda.is_available() else yb
                    pred = net(xb); v = criterion(pred, yb); vl.append(float(v.detach().cpu()))
            trm, vam = mean(tr), mean(vl)
            logger.info(f'Epoch {ep:03d} | train {trm:.4f} | val {vam:.4f}')
            writer.writerow([ep, trm, vam])
            ck = Path(results_dir) / f'epoch_{ep:03d}.pth'; save_ckpt(ck, net, ep, vam)
            if vam < best - 1e-6:
                best = vam; bad = 0
                best_path = Path(results_dir) / f'best_{ep:03d}_{vam:.4f}.pth'; save_ckpt(best_path, net, ep, vam)
                # Save best_model.pth
                best_model_path = Path(results_dir) / 'best_model.pth'
                save_ckpt(best_model_path, net, ep, vam)
            else:
                bad += 1
            if bad >= patience:
                logger.info('Early stopping.'); break
    return best_path
