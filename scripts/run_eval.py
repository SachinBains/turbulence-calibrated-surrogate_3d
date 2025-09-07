import argparse, torch
import json
from src.utils.devices import pick_device
from pathlib import Path
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.utils.seeding import seed_all
from src.utils.logging import get_logger
from src.dataio.channel_dataset import ChannelDataset
from src.models.unet3d import UNet3D
from src.eval.evaluator import evaluate_baseline
from src.eval.temp_scaling import TemperatureScaler
from src.eval.conformal import conformal_wrap

def main(cfg_path, seed, mc_samples, temperature_scale, conformal, cuda):
  cfg=load_config(cfg_path); seed_all(seed or cfg.get('seed',42)); log=get_logger()
  exp_id=cfg.get('experiment_id','EXPERIMENT'); out=Path(cfg['paths']['results_dir'])/exp_id
  # Handle different directory structures
  if 'ensemble' in exp_id.lower() and (out / exp_id / 'members').exists():
      out = out / exp_id / 'members'
  elif (out / exp_id).exists():
      out = out / exp_id
  out.mkdir(parents=True,exist_ok=True)
  val=ChannelDataset(cfg,'val'); test=ChannelDataset(cfg, 'test')
  vl=DataLoader(val,batch_size=1,shuffle=False); tl=DataLoader(test,batch_size=1,shuffle=False)
  # Load best checkpoint (*.pth) by default
  best_ckpts = sorted(out.glob('best_*.pth'))
  assert best_ckpts, f'No checkpoint in {out}'
  ckpt = best_ckpts[-1]
  mcfg = cfg['model']
  net = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
  state = torch.load(ckpt, map_location='cpu', weights_only=False)
  # Handle different checkpoint formats
  if 'model' in state:
      model_state = state['model']
  elif 'swa_model_state_dict' in state:
      model_state = state['swa_model_state_dict']
  elif 'model_state_dict' in state:
      model_state = state['model_state_dict']
  else:
      model_state = state
  
  # Handle DataParallel wrapper (remove 'module.' prefix)
  if any(k.startswith('module.') for k in model_state.keys()):
      model_state = {k.replace('module.', ''): v for k, v in model_state.items()}
  
  net.load_state_dict(model_state)
  if cfg.get('uq', {}).get('method','none')=='mc_dropout': net.enable_mc_dropout(p=cfg.get('uq', {}).get('dropout_p',0.2))
  device = pick_device(cuda)
  net = net.to(device)
  log.info(f'Loaded {ckpt.name}')
  val_metrics = evaluate_baseline(net, vl, device, save_dir=out, cfg=cfg)
  print(f"VAL RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")
  test_metrics = evaluate_baseline(net, tl, device, save_dir=out, cfg=cfg)
  print(f"TEST RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
  # Save metrics to JSON
  val_metrics_json = dict(val_metrics, split='val')
  test_metrics_json = dict(test_metrics, split='test')
  with open(out / 'val_metrics.json', 'w') as f:
      json.dump(val_metrics_json, f, indent=2)
  with open(out / 'test_metrics.json', 'w') as f:
      json.dump(test_metrics_json, f, indent=2)

if __name__=='__main__':
  ap=argparse.ArgumentParser()
  ap.add_argument('--config', required=True)
  ap.add_argument('--seed', type=int, default=None)
  ap.add_argument('--mc-samples', type=int, default=None)
  ap.add_argument('--temperature-scale', action='store_true')
  ap.add_argument('--conformal', action='store_true')
  ap.add_argument('--cuda', action='store_true', help='use CUDA if available')
  a = ap.parse_args()
  main(a.config, a.seed, a.mc_samples, a.temperature_scale, a.conformal, a.cuda)
