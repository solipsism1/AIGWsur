"""M2 final: size per-mode SVD from the cached modes and validate end-to-end h+ match.

Stage 1 — per-mode sizing: for each (l,m) find the minimum n_basis reaching a per-mode
fidelity target (PSD-weighted band overlap for m!=0; relative L2 for the m=0 memory
modes, where amp/phase/in-band overlap is unreliable).

Stage 2 — end-to-end: fit a PerModeCompressor at the chosen allocation, then on held-out
parameters generate the true h+ at random orientations and match against the projection
of the SVD-compressed modes. This is the metric that matters; per-mode targets only feed it.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/size_modes_from_cache.py \
        --cache /mnt/d/gw_waveform_data/modes_cache_v1.h5
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

from src.data.mode_waveforms import (
    ModeGrid, mode_keys, modes_to_amp_phase, amp_phase_to_modes,
    project_to_strain, generate_td_modes, _waveform_kwargs,
)
from src.data.mode_compression import PerModeCompressor, mode_representation
from src.physics.swsh import SWSHProjector


def band_psd_weights(n_time, dt, flo, fhi):
    from pycbc.psd import aLIGOZeroDetHighPower
    f = np.fft.fftfreq(n_time, dt); af = np.abs(f)
    band = (af >= flo) & (af <= fhi)
    df = 1.0 / (n_time * dt); npos = int(round(fhi / df)) + 1
    psd = np.asarray(aLIGOZeroDetHighPower(npos, df, max(flo, df)))
    idx = np.clip((af / df).round().astype(int), 0, npos - 1)
    pv = psd[idx]; w = np.zeros_like(af)
    good = band & np.isfinite(pv) & (pv > 0); w[good] = 1.0 / pv[good]
    return w


def overlap_fft(a, b, w):
    A = np.fft.fft(a); B = np.fft.fft(b)
    num = np.abs(np.sum(A * np.conj(B) * w))
    den = np.sqrt(np.sum(np.abs(A) ** 2 * w) * np.sum(np.abs(B) ** 2 * w))
    return float(num / den) if den > 0 else 0.0


def svd_channels(stack, rep, n_max):
    if rep == "reim":
        chans = [stack.real, stack.imag]
    else:
        a, p = modes_to_amp_phase(stack); chans = [a, p]
    bases = []
    for x in chans:
        mean = x.mean(0); std = x.std(0) + 1e-30
        _, _, vt = np.linalg.svd((x - mean) / std, full_matrices=False)
        bases.append((mean, std, vt[:n_max]))
    return bases


def recon(stack, rep, bases, k):
    if rep == "reim":
        chans = [stack.real, stack.imag]
    else:
        a, p = modes_to_amp_phase(stack); chans = [a, p]
    out = []
    for x, (mean, std, vt) in zip(chans, bases):
        c = ((x - mean) / std) @ vt[:k].T
        out.append((c @ vt[:k]) * std + mean)
    return (out[0] + 1j * out[1]) if rep == "reim" else amp_phase_to_modes(out[0], out[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--config", default="config.long_8d_fs1_v3prod.yaml")
    ap.add_argument("--fit-frac", type=float, default=0.7)
    ap.add_argument("--target", type=float, default=0.999)
    ap.add_argument("--alloc", choices=["flat", "amplitude"], default="amplitude")
    ap.add_argument("--budget", type=float, default=2.0e-3,
                    help="amplitude-alloc mismatch budget per mode (smaller -> more basis)")
    ap.add_argument("--floor", type=float, default=0.95,
                    help="amplitude-alloc minimum per-mode overlap target")
    ap.add_argument("--n-max", type=int, default=160)
    ap.add_argument("--cap", type=int, default=160)
    ap.add_argument("--e2e-points", type=int, default=24)
    ap.add_argument("--out-compressor", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    grid = ModeGrid(); dt = 1.0 / grid.sample_rate
    flo, fhi = float(config["data"]["f_lower"]), float(config["data"]["f_final"])
    keys = mode_keys()

    with h5py.File(args.cache, "r") as f:
        modes = (f["modes_real"][:] + 1j * f["modes_imag"][:]).astype(np.complex128)
        cache_params = f["parameters"][:]
    n = len(modes); n_fit = int(args.fit_frac * n)
    print(f"cache: {n} points, grid N={modes.shape[2]}; fit={n_fit} test={n-n_fit}")
    w = band_psd_weights(grid.n, dt, flo, fhi)
    cands = [2, 4, 8, 12, 16, 24, 32, 48, 64, 80, 100, 120, 140, args.n_max]
    cands = [c for c in cands if c <= min(args.n_max, n_fit - 1)]

    # Per-mode relative amplitude (PSD-weighted band norm), normalized to (2,2).
    def band_norm(stack):
        A = np.fft.fft(stack, axis=1)
        return np.sqrt(np.median(np.sum(np.abs(A) ** 2 * w, axis=1)))
    rel_amp = {k: band_norm(modes[:n_fit, mi]) for mi, k in enumerate(keys)}
    ref = rel_amp[(2, 2)]
    rel_amp = {k: float(v / ref) for k, v in rel_amp.items()}

    # Size by ABSOLUTE contribution to the total mismatch: a mode's reconstruction
    # error matters in proportion to its amplitude, so stop adding basis once
    # rel_amp * mismatch <= budget. Weak modes get few basis even if their own
    # overlap is poor (they barely affect the strain); dominant modes are driven hard.
    def per_mode_fid(test, rep, bases, c):
        rec = recon(test, rep, bases, c)
        if rep == "reim":
            err = np.linalg.norm(rec - test, axis=1) / (np.linalg.norm(test, axis=1) + 1e-30)
            return 1.0 - float(np.median(err))
        return float(np.median([overlap_fft(test[j], rec[j], w) for j in range(len(test))]))

    print(f"\nPer-mode sizing (alloc={args.alloc}, budget={args.budget}, cap {args.cap}):")
    print(f"{'mode':>8} {'rep':>9} {'rel_amp':>8} {'n_basis':>8} {'med_fid':>8} {'contrib':>9}")
    chosen = {}
    for mi, k in enumerate(keys):
        rep = mode_representation(k)
        fit, test = modes[:n_fit, mi], modes[n_fit:, mi]
        bases = svd_channels(fit, rep, args.n_max)
        nb, fid = None, None
        for c in cands:
            if c > args.cap:
                break
            f_med = per_mode_fid(test, rep, bases, c)
            contrib = rel_amp[k] * np.sqrt(max(0.0, 1.0 - f_med ** 2))
            if args.alloc == "flat":
                if f_med >= args.target:
                    nb, fid = c, f_med; break
            elif contrib <= args.budget:
                nb, fid = c, f_med; break
        if nb is None:
            nb = min(args.cap, cands[-1]); fid = per_mode_fid(test, rep, bases, nb)
        chosen[k] = nb
        contrib = rel_amp[k] * np.sqrt(max(0.0, 1.0 - fid ** 2))
        print(f"{str(k):>8} {rep:>9} {rel_amp[k]:>8.3f} {nb:>8d} {fid:>8.5f} {contrib:>9.5f}")

    total = sum(2 * v for v in chosen.values())
    print(f"\nTotal coeff dim: {total}  (legacy h+ = 200)")
    print("per_mode_n_basis =", {f"{l}:{m}": v for (l, m), v in chosen.items()})

    # Stage 2: fit compressor + end-to-end h+ match
    comp = PerModeCompressor(keys, chosen).fit(modes[:n_fit])
    if args.out_compressor:
        comp.save(args.out_compressor); print(f"saved compressor -> {args.out_compressor}")

    from pycbc.waveform import get_td_waveform
    from pycbc.types import TimeSeries
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import match
    swsh = SWSHProjector(keys)
    rng = np.random.default_rng(7)
    matches = []
    test_modes = modes[n_fit:]
    test_params = cache_params[n_fit:]
    coeffs = comp.encode(test_modes)
    recon_modes = comp.decode(coeffs)
    npick = min(args.e2e_points, len(test_modes))
    for j in range(npick):
        kw = _waveform_kwargs(test_params[j], config)
        for _ in range(2):
            iota = float(rng.uniform(0, np.pi)); phi = float(rng.uniform(0, 2 * np.pi))
            hp, _ = get_td_waveform(delta_t=dt, inclination=iota, coa_phase=np.pi / 2 - phi, **kw)
            y = swsh.forward(torch.tensor(iota), torch.tensor(phi)).numpy()
            hpl, _ = project_to_strain(recon_modes[j], y)
            N = max(len(hp), grid.n)
            a = np.zeros(N); a[:len(hp)] = np.asarray(hp.data)
            b = np.zeros(N); b[:grid.n] = hpl
            A = TimeSeries(a, delta_t=dt).to_frequencyseries()
            B = TimeSeries(b, delta_t=dt).to_frequencyseries()
            psd = aLIGOZeroDetHighPower(len(A), A.delta_f, flo)
            matches.append(match(A, B, psd=psd, low_frequency_cutoff=flo, high_frequency_cutoff=fhi)[0])
    matches = np.array(matches)
    print(f"\n[End-to-end] compressed-mode h+ match vs PyCBC over {len(matches)} orientations:")
    print(f"  mean={matches.mean():.5f} median={np.median(matches):.5f} "
          f"min={matches.min():.5f} frac>0.99={np.mean(matches>0.99):.3f}")


if __name__ == "__main__":
    main()
