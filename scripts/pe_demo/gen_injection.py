"""Generate the TRUE NRSur7dq4 modes at the injection point (WSL pycbc env).

This is the only step that needs PyCBC. It dumps the true inertial-frame modes
h_lm(t) on the surrogate's common grid so the torch side can project them with the
identical analytic SWSH layer -> a controlled cross-injection: data = true modes,
template = surrogate modes, same projection. Any posterior offset is then purely the
network's intrinsic mode error.

    wsl -e bash -lc 'cd /mnt/c/Users/beka/Desktop/gw_waveform_codex && \
      LAL_DATA_PATH=/home/beka/lal_data \
      /home/beka/gw_waveform_conda/envs/gw-pycbc/bin/python \
      scripts/pe_demo/gen_injection.py'
"""
from __future__ import annotations
import os, sys
import numpy as np
import yaml

sys.path.append(os.getcwd())
from src.data.mode_waveforms import ModeGrid, generate_td_modes, mode_keys

OUT = "scripts/pe_demo/injection_true_modes.npz"

# Injection intrinsic point (q=2, moderate aligned + small in-plane spins, omega0 in range).
INJ = dict(
    params=np.array([2.0, 0.10, 0.05, 0.30, -0.10, 0.05, -0.20, 0.021], dtype=np.float64),
    # extrinsic (recorded here, applied on the torch side):
    iota=0.9, phi=1.2, psi=0.5,
)


def main():
    cfg = yaml.safe_load(open("config.modes_v1.yaml"))
    grid = ModeGrid()
    keys = mode_keys()
    modes = generate_td_modes(INJ["params"], cfg, grid, modes=keys)
    if modes is None:
        raise SystemExit("PyCBC failed to generate the injection point (omega floor?).")
    print(f"true modes: shape {modes.shape} dtype {modes.dtype} "
          f"max|h22|={np.abs(modes[keys.index((2,2))]).max():.3e}")
    np.savez(OUT,
             modes_real=modes.real.astype(np.float64),
             modes_imag=modes.imag.astype(np.float64),
             keys=np.array([f"{l}:{m}" for (l, m) in keys]),
             params=INJ["params"], iota=INJ["iota"], phi=INJ["phi"], psi=INJ["psi"],
             grid_n=grid.n, sample_rate=grid.sample_rate)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
