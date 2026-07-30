"""Generate and cache a small set of conditioned TD modes for M2 SVD sizing.

Saves complex modes on the common grid plus parameters to an HDF5 file so the
per-mode SVD sizing experiments don't pay the (slow, ~50%-failure) generation cost
each run.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/generate_mode_cache.py \
        --n-valid 700 --out /mnt/d/gw_waveform_data/modes_cache_v1.h5
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import yaml

sys.path.append(os.getcwd())

from src.data.generate_waveforms import sample_parameters
from src.data.mode_waveforms import ModeGrid, generate_td_modes, mode_keys


def load_omega_pool(config: dict) -> np.ndarray:
    data_dir = config["data"].get("data_dir", "")
    for name in ("waveforms_val.h5", "waveforms_train.h5"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            with h5py.File(p, "r") as f:
                if "reference_omega_source_pool" in f:
                    return f["reference_omega_source_pool"][:].astype(np.float64)
    return np.array([], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.long_8d_fs1_v3prod.yaml")
    ap.add_argument("--n-valid", type=int, default=700)
    ap.add_argument("--oversample", type=float, default=2.4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    grid = ModeGrid()
    keys = mode_keys()
    pool = load_omega_pool(config)
    rng = np.random.default_rng(args.seed)

    n_try = int(args.n_valid * args.oversample)
    params = sample_parameters(n_try, config, seed=args.seed)
    if len(pool):
        params[:, 7] = rng.choice(pool, size=len(params))

    n_modes = len(keys)
    out = np.zeros((args.n_valid, n_modes, grid.n), dtype=np.complex64)
    kept = np.zeros((args.n_valid, params.shape[1]), dtype=np.float32)
    nok = 0
    for i in range(len(params)):
        if nok >= args.n_valid:
            break
        mc = generate_td_modes(params[i], config, grid, modes=keys)
        if mc is None:
            continue
        out[nok] = mc.astype(np.complex64)
        kept[nok] = params[i]
        nok += 1
        if nok % 50 == 0:
            print(f"  {nok}/{args.n_valid} valid (tried {i+1})")

    if nok < args.n_valid:
        out = out[:nok]; kept = kept[:nok]
    print(f"Generated {nok} valid (tried {i+1}); writing {args.out}")
    with h5py.File(args.out, "w") as f:
        f.create_dataset("modes_real", data=out.real, compression="gzip", compression_opts=4)
        f.create_dataset("modes_imag", data=out.imag, compression="gzip", compression_opts=4)
        f.create_dataset("parameters", data=kept)
        f.attrs["mode_keys"] = ",".join(f"{l}:{m}" for l, m in keys)
        f.attrs["t_pre"] = grid.t_pre
        f.attrs["t_post"] = grid.t_post
        f.attrs["taper_len"] = grid.taper_len
        f.attrs["sample_rate"] = grid.sample_rate
    print("Done.")


if __name__ == "__main__":
    main()
