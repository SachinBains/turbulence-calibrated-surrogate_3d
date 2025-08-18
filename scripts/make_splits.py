import argparse, yaml
from src.dataio.splits import make_splits_from_h5
if __name__=='__main__':
  ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); a=ap.parse_args()
  import yaml; cfg=yaml.safe_load(open(a.config,'r')); make_splits_from_h5(cfg)
