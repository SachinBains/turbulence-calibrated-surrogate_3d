import os
import torch

def pick_device(use_cuda: bool = False) -> torch.device:
    """Return cuda device only if explicitly requested and actually usable."""
    if use_cuda and torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES', '') not in ('', '-1'):
        return torch.device('cuda')
    return torch.device('cpu')
