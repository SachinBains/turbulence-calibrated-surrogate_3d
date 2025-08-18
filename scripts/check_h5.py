import argparse, yaml, h5py

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
args = ap.parse_args()

cfg = yaml.safe_load(open(args.config, "r"))
VPATH = cfg["paths"]["velocity_h5"]
PPATH = cfg["paths"]["pressure_h5"]
VKEY  = cfg["data"]["velocity_key"]
PKEY  = cfg["data"]["pressure_key"]

print("CONFIG:")
print("  velocity_h5:", VPATH)
print("  pressure_h5:", PPATH)
print("  velocity_key:", VKEY)
print("  pressure_key:", PKEY)

with h5py.File(VPATH, "r") as fv:
    print("\nVELOCITY file keys:", list(fv.keys()))
    print("VELOCITY shape:", fv[VKEY].shape)

with h5py.File(PPATH, "r") as fp:
    print("\nPRESSURE file keys:", list(fp.keys()))
    print("PRESSURE shape:", fp[PKEY].shape)

