from pathlib import Path
def append_manifest_row(config_path, seed, results_dir):
  man=Path('experiments')/'manifest.csv'
  if not man.exists():
    man.parent.mkdir(parents=True,exist_ok=True); man.write_text('config,seed,results_dir\n',encoding='utf-8')
  with open(man,'a',encoding='utf-8') as f: f.write(f'{config_path},{seed},{results_dir}\n')
