"""M2: size a per-mode amplitude/phase SVD for the TD inertial-frame modes.

Two stages:
  1. Conditioning sanity: project conditioned modes to h_+ at random orientations and
     match (PSD-weighted) against PyCBC's true h_+, to confirm the peak-align + taper +
     pad/truncate grid does not degrade the waveform.
  2. Per-mode sizing: fit an amp/phase SVD per (l,m) and find the minimum n_basis whose
     PSD-weighted reconstruction overlap (over |f| in band) reaches the target.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/fit_mode_compressors.py --n 300
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import yaml

sys.path.append(os.getcwd())

from src.data.generate_waveforms import sample_parameters
from src.data.mode_waveforms import (
    ModeGrid, generate_td_modes, mode_keys, modes_to_amp_phase, amp_phase_to_modes,
    project_to_strain, _waveform_kwargs,
)
from src.physics.swsh import SWSHProjector


def load_omega_pool(config: dict) -> np.ndarray:
    data_dir = config["data"].get("data_dir", "")
    for name in ("waveforms_val.h5", "waveforms_train.h5"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            with h5py.File(p, "r") as f:
                if "reference_omega_source_pool" in f:
                    return f["reference_omega_source_pool"][:].astype(np.float64)
                if "omega0" in f:
                    return np.unique(f["omega0"][:].astype(np.float64))
    return np.array([], dtype=np.float64)


def band_psd_weights(n_time: int, dt: float, flo: float, fhi: float):
    """Two-sided 1/PSD weights over |f| in [flo, fhi] for an N-point complex FFT."""
    from pycbc.psd import aLIGOZeroDetHighPower
    f = np.fft.fftfreq(n_time, dt)
    af = np.abs(f)
    band = (af >= flo) & (af <= fhi)
    df = 1.0 / (n_time * dt)
    npos = int(round(fhi / df)) + 1
    psd = np.asarray(aLIGOZeroDetHighPower(npos, df, max(flo, df)))
    w = np.zeros_like(af)
    idx = np.clip((af / df).round().astype(int), 0, npos - 1)
    pv = psd[idx]
    good = band & np.isfinite(pv) & (pv > 0)
    w[good] = 1.0 / pv[good]
    return w


def psd_overlap_td(a_td: np.ndarray, b_td: np.ndarray, w: np.ndarray) -> float:
    A = np.fft.fft(a_td); B = np.fft.fft(b_td)
    num = np.abs(np.sum(A * np.conj(B) * w))
    den = np.sqrt(np.sum((np.abs(A) ** 2) * w) * np.sum((np.abs(B) ** 2) * w))
    return float(num / den) if den > 0 else 0.0


def match_pycbc(h_true_td, h_gen_td, dt, flo, fhi):
    from pycbc.types import TimeSeries
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import match
    a = TimeSeries(h_true_td.astype(np.float64), delta_t=dt)
    b = TimeSeries(h_gen_td.astype(np.float64), delta_t=dt)
    Af = a.to_frequencyseries(); df = Af.delta_f
    psd = aLIGOZeroDetHighPower(len(Af), df, flo)
    m, _ = match(a.to_frequencyseries(), b.to_frequencyseries(), psd=psd,
                 low_frequency_cutoff=flo, high_frequency_cutoff=fhi)
    return float(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.long_8d_fs1_v3prod.yaml")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--fit-frac", type=float, default=0.7)
    ap.add_argument("--target", type=float, default=0.999)
    ap.add_argument("--n-max", type=int, default=120)
    ap.add_argument("--n-sanity", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260606)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    grid = ModeGrid()
    keys = mode_keys()
    dt = 1.0 / grid.sample_rate
    flo = float(config["data"]["f_lower"]); fhi = float(config["data"]["f_final"])

    pool = load_omega_pool(config)
    rng = np.random.default_rng(args.seed)
    params = sample_parameters(args.n, config, seed=args.seed)
    if len(pool):
        params[:, 7] = rng.choice(pool, size=len(params))

    from pycbc.waveform import get_td_waveform
    swsh = SWSHProjector(keys)

    print(f"Generating conditioned TD modes (grid N={grid.n}, peak@{grid.peak_index})...")
    stacks = []
    used_params = []
    sanity = []
    for i in range(len(params)):
        mc = generate_td_modes(params[i], config, grid, modes=keys)
        if mc is None:
            continue
        stacks.append(mc.astype(np.complex64))
        used_params.append(params[i])
        # conditioning sanity on the first few valid points
        if len(sanity) < args.n_sanity:
            kw = _waveform_kwargs(params[i], config)
            for _ in range(2):
                iota = float(rng.uniform(0, np.pi)); phi = float(rng.uniform(0, 2*np.pi))
                coa = np.pi/2 - phi
                hp, _ = get_td_waveform(delta_t=dt, inclination=iota, coa_phase=coa, **kw)
                y = swsh.forward(torch.tensor(iota, dtype=torch.float64),
                                 torch.tensor(phi, dtype=torch.float64)).numpy()
                hplus_grid, _ = project_to_strain(mc, y)
                # align lengths (true h+ vs grid recon) by FFT-domain match (match maximizes time shift)
                n = max(len(hplus_grid), len(hp))
                a = np.zeros(n); b = np.zeros(n)
                a[:len(hp)] = np.asarray(hp.data); b[:len(hplus_grid)] = hplus_grid
                sanity.append(match_pycbc(a, b, dt, flo, fhi))
        if len(stacks) % 50 == 0:
            print(f"  {len(stacks)} valid")
    n_ok = len(stacks)
    print(f"  {n_ok}/{args.n} valid")
    sanity = np.array(sanity)
    print(f"\n[Conditioning sanity] h+ match vs PyCBC over {len(sanity)} orientations: "
          f"mean={sanity.mean():.5f} median={np.median(sanity):.5f} min={sanity.min():.5f}")

    data = np.stack(stacks)  # (n_ok, n_modes, N)
    del stacks
    n_fit = int(args.fit_frac * n_ok)
    w = band_psd_weights(grid.n, dt, flo, fhi)
    candidates = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 100, args.n_max]
    candidates = [c for c in candidates if c <= min(args.n_max, n_fit - 1)]

    print(f"\nPer-mode sizing (target overlap {args.target}, fit={n_fit}, test={n_ok-n_fit}):")
    print(f"{'mode':>8} {'n_basis':>8} {'med_ovlp':>9} {'min_ovlp':>9}")
    total = 0
    res = {}
    for mi, k in enumerate(keys):
        cm = data[:, mi, :].astype(np.complex128)
        fit, test = cm[:n_fit], cm[n_fit:]
        amp_f, ph_f = modes_to_amp_phase(fit)
        am, asd = amp_f.mean(0), amp_f.std(0) + 1e-30
        pm, psd_ = ph_f.mean(0), ph_f.std(0) + 1e-30
        _, _, va = np.linalg.svd((amp_f - am) / asd, full_matrices=False)
        _, _, vp = np.linalg.svd((ph_f - pm) / psd_, full_matrices=False)
        amp_t, ph_t = modes_to_amp_phase(test)
        chosen, stats = args.n_max, None
        for c in candidates:
            ca = ((amp_t - am) / asd) @ va[:c].T
            cp = ((ph_t - pm) / psd_) @ vp[:c].T
            ar = (ca @ va[:c]) * asd + am
            pr = (cp @ vp[:c]) * psd_ + pm
            rec = amp_phase_to_modes(ar, pr)
            ov = np.array([psd_overlap_td(test[j], rec[j], w) for j in range(len(test))])
            if np.median(ov) >= args.target:
                chosen, stats = c, (np.median(ov), ov.min()); break
        if stats is None:
            stats = (np.median(ov), ov.min())
        res[k] = chosen; total += 2 * chosen
        print(f"{str(k):>8} {chosen:>8d} {stats[0]:>9.5f} {stats[1]:>9.5f}")

    print(f"\nTotal coefficients (sum 2*n_basis over {len(keys)} modes): {total}  (legacy h+ used 200)")
    print("per_mode_n_basis =", {str(k): v for k, v in res.items()})


if __name__ == "__main__":
    main()
