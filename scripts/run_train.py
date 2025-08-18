import argparse, torch
from pathlib import Path
from src.utils.seeding import seed_all
from src.utils.config import load_config
from src.utils.logging import get_logger
from src.utils.manifest import append_manifest_row
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.train.losses import make_loss
from src.train.trainer import train_loop
from torch.utils.data import DataLoader

def main(cfg_path, seed):
  cfg=load_config(cfg_path); seed_all(seed or cfg.get('seed',42)); log=get_logger()
  exp_id=cfg.get('experiment_id','EXPERIMENT'); out=Path(cfg['paths']['results_dir'])/exp_id; out.mkdir(parents=True,exist_ok=True)
  tr=HITDataset(cfg,'train'); va=HITDataset(cfg,'val')
  tl=DataLoader(tr,batch_size=cfg['train']['batch_size'],shuffle=True,num_workers=cfg['train']['num_workers'])
  vl=DataLoader(va,batch_size=1,shuffle=False,num_workers=cfg['train']['num_workers'])
  mcfg=cfg['model']; net=UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
  if cfg['uq'].get('method','none')=='mc_dropout': net.enable_mc_dropout(p=cfg['uq'].get('dropout_p',0.2))
  net=net.cuda() if torch.cuda.is_available() else net
  crit=make_loss(cfg); opt=torch.optim.AdamW(net.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
  scaler=torch.cuda.amp.GradScaler(enabled=bool(cfg['train']['amp']))
  best=train_loop(cfg, net, crit, opt, scaler, tl, vl, out, log); append_manifest_row(cfg_path, seed or cfg.get('seed',42), str(out)); log.info(f'Best: {best}')

if __name__=='__main__':
  ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--seed',type=int,default=None); a=ap.parse_args(); main(a.config,a.seed)
