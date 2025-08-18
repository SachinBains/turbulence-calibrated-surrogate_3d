import logging, sys
def get_logger():
  lg=logging.getLogger('surrogate')
  if not lg.handlers:
    lg.setLevel(logging.INFO); h=logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')); lg.addHandler(h)
  return lg
