import argparse, torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seeding import seed_all
from src.utils.logging import get_logger
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.eval.evaluate import evaluate_model
from src.eval.temp_scaling import TemperatureScaler
from src.eval.conformal import conformal_wrap

def main(cfg_path, seed, mc_samples, temperature_scale, conformal):
  cfg=load_config(cfg_path); seed_all(seed or cfg.get('seed',42)); log=get_logger()
  exp_id=cfg.get('experiment_id','EXPERIMENT'); out=Path(cfg['paths']['results_dir'])/exp_id; out.mkdir(parents=True,exist_ok=True)
  val=HITDataset(cfg,'val',eval_mode=True); test=HITDataset(cfg,'test',eval_mode=True)
  vl=DataLoader(val,batch_size=1,shuffle=False); tl=DataLoader(test,batch_size=1,shuffle=False)
  ckpt=sorted(out.glob('best_*.pth')); assert ckpt, f'No checkpoint in {out}'; ckpt=ckpt[-1]
  mcfg=cfg['model']; net=UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
  state=torch.load(ckpt, map_location='cpu'); net.load_state_dict(state['model'])
  if cfg['uq'].get('method','none')=='mc_dropout': net.enable_mc_dropout(p=cfg['uq'].get('dropout_p',0.2))
  net=net.cuda() if torch.cuda.is_available() else net; log.info(f'Loaded {ckpt.name}')
  e=dict(mc_samples=mc_samples or cfg['eval'].get('mc_samples',32))
  if temperature_scale: from src.eval.temp_scaling import TemperatureScaler; t=TemperatureScaler().fit(vl, net, log); e['temperature']=float(t)
  evaluate_model(cfg, net, vl, out, split='val', **e); evaluate_model(cfg, net, tl, out, split='test', **e)
  if conformal: conformal_wrap(cfg, out, log)

if __name__=='__main__':
  ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--seed',type=int,default=None)
  ap.add_argument('--mc-samples',type=int,default=None); ap.add_argument('--temperature-scale',action='store_true'); ap.add_argument('--conformal',action='store_true')
  a=ap.parse_args(); main(a.config,a.seed,a.mc_samples,a.temperature_scale,a.conformal)
