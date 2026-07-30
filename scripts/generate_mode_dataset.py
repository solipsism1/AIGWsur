"""Generate a mode-coefficient training dataset (coeffs only — raw modes are too big).

Samples parameters (Sobol + benchmark omega0 pool), generates conditioned TD modes,
encodes them with a fitted PerModeCompressor, and stores only the per-mode SVD
coefficient vectors plus parameters. 600k raw modes would be ~300 GB; the coefficient
store is ~coeff_dim*4 bytes/point.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/generate_mode_dataset.py \
        --compressor /mnt/d/gw_waveform_data/mode_compressor_v1.h5 \
        --n-valid 12000 --seed 1 --out /mnt/d/gw_waveform_data/modes_ds_smoke.h5
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import yaml

sys.path.append(os.getcwd())

from src.data.generate_waveforms import sample_parameters
from src.data.mode_compression import PerModeCompressor
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
    ap.add_argument("--compressor", required=True)
    ap.add_argument("--n-valid", type=int, required=True)
    ap.add_argument("--oversample", type=float, default=2.3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flush-every", type=int, default=2000)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    grid = ModeGrid()
    keys = mode_keys()
    comp = PerModeCompressor.load(args.compressor)
    coeff_dim = comp.coeff_dim
    pool = load_omega_pool(config)
    rng = np.random.default_rng(args.seed)

    n_try = int(args.n_valid * args.oversample)
    params_all = sample_parameters(n_try, config, seed=args.seed)
    if len(pool):
        params_all[:, 7] = rng.choice(pool, size=len(params_all))

    coeffs = np.zeros((args.n_valid, coeff_dim), dtype=np.float32)
    kept = np.zeros((args.n_valid, params_all.shape[1]), dtype=np.float32)
    nok = 0
    t0 = time.time()

    def flush():
        with h5py.File(args.out, "w") as f:
            f.create_dataset("coeffs", data=coeffs[:nok])
            f.create_dataset("parameters", data=kept[:nok])
            f.attrs["mode_keys"] = ",".join(f"{l}:{m}" for l, m in keys)
            f.attrs["coeff_dim"] = coeff_dim
            f.attrs["compressor"] = os.path.basename(args.compressor)
            f.attrs["t_pre"] = grid.t_pre; f.attrs["t_post"] = grid.t_post
            f.attrs["taper_len"] = grid.taper_len; f.attrs["sample_rate"] = grid.sample_rate
            f.attrs["seed"] = args.seed

    for i in range(len(params_all)):
        if nok >= args.n_valid:
            break
        mc = generate_td_modes(params_all[i], config, grid, modes=keys)
        if mc is None:
            continue
        coeffs[nok] = comp.encode(mc[None].astype(np.complex128))[0]
        kept[nok] = params_all[i]
        nok += 1
        if nok % args.flush_every == 0:
            flush()
            rate = nok / (time.time() - t0)
            print(f"  {nok}/{args.n_valid} valid (tried {i+1}, {rate:.1f}/s)")

    flush()
    print(f"Done: {nok} valid (tried {i+1}) in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
