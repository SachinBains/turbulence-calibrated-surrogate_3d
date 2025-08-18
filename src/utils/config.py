import yaml, os
def load_config(path):
  cfg=yaml.safe_load(open(path,'r'))
  for k in ['velocity_h5','pressure_h5','splits_dir','results_dir','experiments_manifest']:
    if k in cfg['paths']: cfg['paths'][k]=os.path.normpath(cfg['paths'][k])
  return cfg
