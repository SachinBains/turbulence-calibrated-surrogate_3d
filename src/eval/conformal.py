import numpy as np, h5py, json
from pathlib import Path
def conformal_wrap(cfg, results_dir, logger):
  rd=Path(results_dir)
  with h5py.File(rd/'preds_val.h5','r') as f: mu_v=f['pred_mean'][...]; y_v=f['target'][...]
  with h5py.File(rd/'preds_test.h5','r') as f: mu_t=f['pred_mean'][...]; y_t=f['target'][...]
  res=np.abs(mu_v-y_v).reshape(-1); q90=np.quantile(res,0.90); q95=np.quantile(res,0.95)
  lo90=mu_t-q90; hi90=mu_t+q90; lo95=mu_t-q95; hi95=mu_t+q95
  cov90=float(((y_t>=lo90)&(y_t<=hi90)).mean()); cov95=float(((y_t>=lo95)&(y_t<=hi95)).mean())
  json.dump({'q90':float(q90),'q95':float(q95),'test_cov90':cov90,'test_cov95':cov95}, open(rd/'conformal.json','w'), indent=2)
  logger.info(f'Conformal q90={q90:.4f} (cov={cov90:.3f}), q95={q95:.4f} (cov={cov95:.3f})')
