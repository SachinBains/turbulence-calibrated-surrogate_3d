import random, numpy as np
try: import torch
except Exception: torch=None
def seed_all(seed:int):
  random.seed(seed); np.random.seed(seed)
  if torch is not None:
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try:
      torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    except Exception: pass
