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

def load_model_for_config(cfg, device):
    """Load the appropriate model based on configuration."""
    mcfg = cfg['model']
    model_name = mcfg.get('name', 'unet3d')
    uq_method = cfg.get('uq', {}).get('method', 'none')
    
    # For now, always use standard UNet3D (variational will be handled by state dict conversion)
    net = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
    print("Loaded standard UNet3D model")
    else:
        # Standard UNet3D for all other cases
        net = UNet3D(mcfg['in_channels'], mcfg['out_channels'], base_ch=mcfg['base_channels'])
        print("Loaded standard UNet3D model")
    
    return net

def clean_state_dict(state_dict, model_type='standard'):
    """Clean state_dict based on model type."""
    cleaned_state = {}
    
    if model_type == 'swag':
        # Remove SWAG-specific keys
        swag_keys = ['n_averaged', 'swa_n', 'swa_count']
        for key, value in state_dict.items():
            if key not in swag_keys:
                cleaned_state[key] = value
        print(f"Removed SWAG keys: {[k for k in swag_keys if k in state_dict]}")
        
    elif model_type == 'variational':
        # For variational models, we might need to sample from the distributions
        # or use the mean parameters. This depends on your variational implementation.
        # If you want to use mean parameters for evaluation:
        for key, value in state_dict.items():
            if key.endswith('_mu'):
                # Use mean parameters, remove '_mu' suffix
                new_key = key.replace('_mu', '')
                cleaned_state[new_key] = value
            elif not key.endswith('_logvar'):
                # Keep non-variational parameters as is
                cleaned_state[key] = value
        print("Converted variational parameters to deterministic (using means)")
        
    else:
        # Standard model - just handle DataParallel
        for key, value in state_dict.items():
            if key.startswith('module.'):
                cleaned_state[key.replace('module.', '')] = value
            else:
                cleaned_state[key] = value
    
    return cleaned_state

def detect_model_type(state_dict, config):
    """Detect what type of model this checkpoint is from."""
    uq_method = config.get('uq', {}).get('method', 'none')
    
    # Check for specific keys to identify model type
    if 'n_averaged' in state_dict:
        return 'swag'
    elif any(key.endswith('_mu') or key.endswith('_logvar') for key in state_dict.keys()):
        return 'variational'
    elif uq_method == 'mc_dropout':
        return 'mc_dropout'
    elif uq_method == 'ensemble':
        return 'ensemble'
    else:
        return 'standard'

def main(cfg_path, seed, mc_samples, temperature_scale, conformal, cuda):
    cfg = load_config(cfg_path)
    seed_all(seed or cfg.get('seed', 42))
    log = get_logger()
    
    exp_id = cfg.get('experiment_id', 'EXPERIMENT')
    base_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Handle ensemble case - use first member directory
    if 'ensemble' in exp_id.lower():
        members_dir = base_dir / exp_id / 'members'
        if not members_dir.exists():
            members_dir = base_dir / 'members'
        
        if members_dir.exists():
            member_dirs = sorted([d for d in members_dir.iterdir() if d.is_dir() and d.name.startswith('m')])
            if member_dirs:
                out = member_dirs[0]  # Use first member (m00)
                print(f"Using ensemble member: {member_dirs[0].name}")
            else:
                out = base_dir
        else:
            out = base_dir
    else:
        # For non-ensemble cases, use base directory directly
        out = base_dir
    
    out.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    val = ChannelDataset(cfg, 'val')
    test = ChannelDataset(cfg, 'test')
    vl = DataLoader(val, batch_size=1, shuffle=False)
    tl = DataLoader(test, batch_size=1, shuffle=False)
    
    # Find checkpoint with multiple naming patterns
    checkpoint_patterns = ['best_*.pth', 'best_model.pth', 'model_*.pth', '*.pth']
    ckpt = None
    
    for pattern in checkpoint_patterns:
        ckpts = sorted(out.glob(pattern))
        if ckpts:
            # Filter for actual model checkpoints
            if pattern == '*.pth':
                ckpts = [f for f in ckpts if any(word in f.name.lower() 
                                               for word in ['best', 'model', 'checkpoint', 'final'])]
            if ckpts:
                ckpt = ckpts[-1]  # Use the most recent
                break
    
    if ckpt is None:
        available_files = list(out.glob("*"))
        raise FileNotFoundError(f'No checkpoint found in {out}. Available files: {available_files}')
    
    print(f"Loading checkpoint: {ckpt}")
    
    # Load checkpoint
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if 'model' in state:
        model_state = state['model']
    elif 'swa_model_state_dict' in state:
        model_state = state['swa_model_state_dict']
    elif 'model_state_dict' in state:
        model_state = state['model_state_dict']
    elif 'state_dict' in state:
        model_state = state['state_dict']
    else:
        model_state = state
    
    # Detect model type and clean state dict
    model_type = detect_model_type(model_state, cfg)
    print(f"Detected model type: {model_type}")
    
    # Load appropriate model
    net = load_model_for_config(cfg, None)  # Don't move to device yet
    
    # Clean state dict based on model type
    cleaned_state = clean_state_dict(model_state, model_type, net)
    
    # Load state dict with error handling
    try:
        net.load_state_dict(cleaned_state, strict=True)
        print("Successfully loaded state dict (strict mode)")
    except RuntimeError as e:
        print(f"Strict loading failed: {e}")
        print("Trying non-strict loading...")
        try:
            missing_keys, unexpected_keys = net.load_state_dict(cleaned_state, strict=False)
            if missing_keys:
                print(f"Warning - Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"Warning - Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            print("Successfully loaded state dict (non-strict mode)")
        except Exception as e2:
            print(f"Non-strict loading also failed: {e2}")
            print("This might require a different model architecture or custom loading logic.")
            raise
    
    # Enable MC Dropout if needed
    if cfg.get('uq', {}).get('method', 'none') == 'mc_dropout':
        if hasattr(net, 'enable_mc_dropout'):
            net.enable_mc_dropout(p=cfg.get('uq', {}).get('dropout_p', 0.2))
        else:
            print("Warning: MC Dropout requested but model doesn't support enable_mc_dropout")
    
    # Move to device and set eval mode
    device = pick_device(cuda)
    net = net.to(device)
    net.eval()
    
    log.info(f'Loaded {ckpt.name} ({model_type} model)')
    
    # Evaluate
    val_metrics = evaluate_baseline(net, vl, device, save_dir=out, cfg=cfg)
    print(f"VAL RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}")
    
    test_metrics = evaluate_baseline(net, tl, device, save_dir=out, cfg=cfg)
    print(f"TEST RMSE: {test_metrics['rmse']:.4f}, MAE: {test_metrics['mae']:.4f}")
    
    # Save metrics
    val_metrics_json = dict(val_metrics, split='val', model_type=model_type)
    test_metrics_json = dict(test_metrics, split='test', model_type=model_type)
    
    with open(out / 'val_metrics.json', 'w') as f:
        json.dump(val_metrics_json, f, indent=2)
    with open(out / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics_json, f, indent=2)
    
    print(f"Evaluation complete. Results saved to {out}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--mc-samples', type=int, default=None)
    ap.add_argument('--temperature-scale', action='store_true')
    ap.add_argument('--conformal', action='store_true')
    ap.add_argument('--cuda', action='store_true', help='use CUDA if available')
    a = ap.parse_args()
    main(a.config, a.seed, a.mc_samples, a.temperature_scale, a.conformal, a.cuda)