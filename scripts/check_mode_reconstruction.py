"""M1 master gate: validate the inertial-frame mode -> strain reconstruction.

For each sampled parameter point we pull the raw time-domain ``h_{lm}`` modes from
NRSur7dq4 (coa_phase = 0), project them with the differentiable spin-weighted
spherical harmonics, and compare against the ``(h_+, h_x)`` that PyCBC returns for
the *same* parameters at several orientations ``(iota, phi)``.

Convention (locked empirically, see MODES_V1_SPEC physics invariants)::

    h_+ - i h_x = sum_{lm} h_{lm} * _{-2}Y_{lm}(iota, phi)

and PyCBC's ``coa_phase`` maps to the azimuth as ``phi = pi/2 - coa_phase``. This
single check validates the mode-frame convention, the SWSH convention, and the
projection together. GATE: max relative L2 error <= 1e-6.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/check_mode_reconstruction.py --n 200
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.append(os.getcwd())

from src.data.generate_waveforms import sample_parameters, uses_reference_omega
from src.physics.swsh import default_modes, sYlm


def _complex_mode(re_ts, im_ts, t0: float, dt: float, n_ref: int) -> np.ndarray:
    """Place a (real, imag) mode TimeSeries pair onto the reference time grid."""
    hlm = np.asarray(re_ts.data) + 1j * np.asarray(im_ts.data)
    off = int(round((float(re_ts.start_time) - t0) / dt))
    out = np.zeros(n_ref, dtype=np.complex128)
    src0 = max(0, -off)
    dst0 = max(0, off)
    n = min(len(hlm) - src0, n_ref - dst0)
    if n > 0:
        out[dst0:dst0 + n] = hlm[src0:src0 + n]
    return out


def reconstruct(modes_dict, mode_list, iota: float, phi: float, t0, dt, n_ref) -> np.ndarray:
    acc = np.zeros(n_ref, dtype=np.complex128)
    it = torch.tensor(iota, dtype=torch.float64)
    ip = torch.tensor(phi, dtype=torch.float64)
    for (l, m) in mode_list:
        re_ts, im_ts = modes_dict[(l, m)]
        y = complex(sYlm(l, m, it, ip).item())
        acc += _complex_mode(re_ts, im_ts, t0, dt, n_ref) * y
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.long_8d_fs1_v3prod.yaml")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ell-max", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n-orient", type=int, default=3)
    args = ap.parse_args()

    from pycbc.waveform import get_td_waveform, get_td_waveform_modes

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mode_list = default_modes(args.ell_max, include_m0=True)
    params = sample_parameters(args.n, config, seed=args.seed)
    include_omega = uses_reference_omega(config)
    M = float(config["data"]["reference_total_mass"])
    f_lower = float(config["data"]["f_lower"])
    delta_t = 1.0 / float(config["data"].get("sample_rate", 4096.0))

    from src.data.generate_waveforms import omega0_to_f_ref_hz

    rng = np.random.default_rng(args.seed)
    rel_errors = []
    n_fail_gen = 0

    for i in range(len(params)):
        p = params[i]
        q = float(p[0]); m1 = M * q / (1 + q); m2 = M / (1 + q)
        kw = dict(
            approximant=config["data"]["approximant"], mass1=m1, mass2=m2,
            spin1x=float(p[1]), spin1y=float(p[2]), spin1z=float(p[3]),
            spin2x=float(p[4]), spin2y=float(p[5]), spin2z=float(p[6]),
            delta_t=delta_t, f_lower=f_lower, distance=1.0,
        )
        if include_omega:
            f_ref = omega0_to_f_ref_hz(float(p[7]), M)
            kw["f_ref"] = f_ref
            kw["f_lower"] = min(f_lower, f_ref)
        else:
            kw["f_ref"] = f_lower

        try:
            modes = get_td_waveform_modes(**kw)
        except Exception as exc:
            n_fail_gen += 1
            print(f"  [{i}] mode gen failed: {exc}")
            continue

        for _ in range(args.n_orient):
            iota = float(rng.uniform(0.0, np.pi))
            phi = float(rng.uniform(0.0, 2 * np.pi))
            coa = np.pi / 2 - phi
            hp, hc = get_td_waveform(inclination=iota, coa_phase=coa, **kw)
            t0 = float(hp.start_time); n_ref = len(hp)
            acc = reconstruct(modes, mode_list, iota, phi, t0, delta_t, n_ref)
            ht = np.asarray(hp.data) - 1j * np.asarray(hc.data)
            mask = np.abs(ht) > 0
            rel = np.linalg.norm(acc[mask] - ht[mask]) / np.linalg.norm(ht[mask])
            rel_errors.append(rel)

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(params)}: running max relerr = {max(rel_errors):.2e}")

    rel_errors = np.array(rel_errors)
    print("\n=== M1 reconstruction check ===")
    print(f"  points={len(params)} ({n_fail_gen} gen fails), orientations/pt={args.n_orient}, modes={len(mode_list)}")
    print(f"  mean relerr   = {rel_errors.mean():.3e}")
    print(f"  median relerr = {np.median(rel_errors):.3e}")
    print(f"  max relerr    = {rel_errors.max():.3e}")
    gate = rel_errors.max() <= 1e-6
    print(f"  GATE (max <= 1e-6): {'PASS' if gate else 'FAIL'}")
    sys.exit(0 if gate else 1)


if __name__ == "__main__":
    main()
