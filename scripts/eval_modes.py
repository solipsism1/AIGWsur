"""Evaluate the inertial-frame mode surrogate (M4 regression + M6 metrics).

On fresh held-out parameters it reports:
  (a) fixed-orientation (face-on) h+ match — regression vs the old single-h+ model;
  (b) orientation-averaged match over sampled (iota, phi) — the new capability;
  (c) per-mode match;
  (d) match-vs-q table.

Truth: PyCBC get_td_waveform(iota, coa=pi/2-phi) for the strain, get_td_waveform_modes
for per-mode comparison. Runs in the WSL PyCBC env (CPU torch is fine).

    LAL_DATA_PATH=/home/beka/lal_data python scripts/eval_modes.py \
        --run runs/modes_smoke --config config.modes_v1.yaml --n-test 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

sys.path.append(os.getcwd())

from src.data.generate_waveforms import sample_parameters
from src.data.mode_compression import PerModeCompressor
from src.data.mode_waveforms import (
    ModeGrid, generate_td_modes, mode_keys, project_to_strain, _waveform_kwargs,
)
from src.models.flow import build_model
from src.physics.swsh import SWSHProjector


def load_omega_pool(config):
    data_dir = config["data"].get("data_dir", "")
    for name in ("waveforms_val.h5", "waveforms_train.h5"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            with h5py.File(p, "r") as f:
                if "reference_omega_source_pool" in f:
                    return f["reference_omega_source_pool"][:].astype(np.float64)
    return np.array([])


def match_hp(true_td, gen_grid, dt, flo, fhi):
    from pycbc.types import TimeSeries
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import match
    N = max(len(true_td), len(gen_grid))
    a = np.zeros(N); a[:len(true_td)] = true_td
    b = np.zeros(N); b[:len(gen_grid)] = gen_grid
    A = TimeSeries(a, delta_t=dt).to_frequencyseries()
    B = TimeSeries(b, delta_t=dt).to_frequencyseries()
    psd = aLIGOZeroDetHighPower(len(A), A.delta_f, flo)
    return float(match(A, B, psd=psd, low_frequency_cutoff=flo, high_frequency_cutoff=fhi)[0])


def band_overlap(a_td, b_td, w):
    A = np.fft.fft(a_td); B = np.fft.fft(b_td)
    num = np.abs(np.sum(A * np.conj(B) * w))
    den = np.sqrt(np.sum(np.abs(A) ** 2 * w) * np.sum(np.abs(B) ** 2 * w))
    return float(num / den) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="config.modes_v1.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-orient", type=int, default=8)
    ap.add_argument("--seed", type=int, default=999)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    grid = ModeGrid(); dt = 1.0 / grid.sample_rate
    flo, fhi = float(config["data"]["f_lower"]), float(config["data"]["f_final"])
    keys = mode_keys()
    run = Path(args.run)

    stats = np.load(run / "norm_stats.npz")
    cmean = stats["coeffs_mean"]; cstd = stats["coeffs_std"]
    pmin = stats["param_min"]; pmax = stats["param_max"]

    ckpt = args.checkpoint or str(run / "best_model.pt")
    ck = torch.load(ckpt, map_location="cpu")
    model = build_model(config); model.load_state_dict(ck["model_state_dict"]); model.eval()
    comp = PerModeCompressor.load(config["compression"]["compressor_path"])
    swsh = SWSHProjector(keys)
    print(f"loaded {ckpt} (epoch {ck.get('epoch','?')}); coeff_dim={comp.coeff_dim}")

    from pycbc.waveform import get_td_waveform, get_td_waveform_modes
    from pycbc.psd import aLIGOZeroDetHighPower
    f = np.fft.fftfreq(grid.n, dt); af = np.abs(f)
    df = 1.0 / (grid.n * dt); npos = int(round(fhi / df)) + 1
    psd = np.asarray(aLIGOZeroDetHighPower(npos, df, max(flo, df)))
    idx = np.clip((af / df).round().astype(int), 0, npos - 1)
    w = np.where((af >= flo) & (af <= fhi) & np.isfinite(psd[idx]) & (psd[idx] > 0), 1.0 / psd[idx], 0.0)

    pool = load_omega_pool(config)
    rng = np.random.default_rng(args.seed)
    params = sample_parameters(int(args.n_test * 2.4), config, seed=args.seed)
    if len(pool):
        params[:, 7] = rng.choice(pool, size=len(params))

    def predict_modes(p):
        pn = 2.0 * (p - pmin) / (pmax - pmin + 1e-10) - 1.0
        with torch.no_grad():
            std = model(torch.tensor(pn[None], dtype=torch.float32)).numpy()[0]
        raw = std * cstd + cmean
        return comp.decode(raw[None])[0]

    fixed_matches, orient_matches, per_mode_ov, qs = [], [], {k: [] for k in keys}, []
    overlay = []
    amp_ratios = []
    n_done = 0
    for i in range(len(params)):
        if n_done >= args.n_test:
            break
        p = params[i]
        true_modes = generate_td_modes(p, config, grid, modes=keys)
        if true_modes is None:
            continue
        pred_modes = predict_modes(p)
        kw = _waveform_kwargs(p, config)

        # (c) per-mode overlap pred vs true (conditioned modes)
        for mi, k in enumerate(keys):
            per_mode_ov[k].append(band_overlap(true_modes[mi], pred_modes[mi], w))

        # (a) fixed orientation: face-on iota=0, phi=0
        hp0, _ = get_td_waveform(delta_t=dt, inclination=0.0, coa_phase=np.pi / 2, **kw)
        y0 = swsh.forward(torch.tensor(0.0), torch.tensor(0.0)).numpy()
        hpl0, _ = project_to_strain(pred_modes, y0)
        fixed_matches.append(match_hp(np.asarray(hp0.data), hpl0, dt, flo, fhi))
        amp_ratios.append(float(np.max(np.abs(hpl0)) / (np.max(np.abs(np.asarray(hp0.data))) + 1e-300)))

        # (b) orientation-averaged
        for _ in range(args.n_orient):
            iota = float(np.arccos(rng.uniform(-1, 1))); phi = float(rng.uniform(0, 2 * np.pi))
            hp, _ = get_td_waveform(delta_t=dt, inclination=iota, coa_phase=np.pi / 2 - phi, **kw)
            y = swsh.forward(torch.tensor(iota), torch.tensor(phi)).numpy()
            hpl, _ = project_to_strain(pred_modes, y)
            mval = match_hp(np.asarray(hp.data), hpl, dt, flo, fhi)
            orient_matches.append(mval)
            if isinstance(overlay, list) and len(overlay) < 200:
                overlay.append((mval, np.asarray(hp.data).copy(), hpl.copy(), float(iota), float(phi)))
        qs.append(float(p[0]))
        n_done += 1
        if n_done % 25 == 0:
            print(f"  {n_done}/{args.n_test}: fixed~{np.mean(fixed_matches):.4f} "
                  f"orient~{np.mean(orient_matches):.4f}")

    fixed = np.array(fixed_matches); orient = np.array(orient_matches); qs = np.array(qs)
    amp_ratio = np.array(amp_ratios)
    out = {
        "n_test": n_done, "n_orient": args.n_orient,
        "amplitude_ratio": {"median": float(np.median(amp_ratio)), "mean": float(amp_ratio.mean()),
                            "p10": float(np.percentile(amp_ratio, 10)), "p90": float(np.percentile(amp_ratio, 90))},
        "fixed_orientation": {"mean": float(fixed.mean()), "median": float(np.median(fixed)),
                              "min": float(fixed.min()), "frac>0.99": float((fixed > 0.99).mean())},
        "orientation_averaged": {"mean": float(orient.mean()), "median": float(np.median(orient)),
                                 "min": float(orient.min()), "frac>0.99": float((orient > 0.99).mean())},
        "per_mode_overlap_median": {f"{l},{m}": float(np.median(per_mode_ov[(l, m)])) for (l, m) in keys},
    }
    # (d) match vs q
    qbins = [(1, 1.75), (1.75, 2.5), (2.5, 3.25), (3.25, 4.0)]
    fixed_q = []
    # recompute orient matches per test index: orient stored flat; map back
    orient_per_test = orient.reshape(n_done, args.n_orient).mean(axis=1)
    out["match_vs_q"] = []
    for lo, hi in qbins:
        m = (qs >= lo) & (qs < hi)
        if m.sum():
            out["match_vs_q"].append({"q_range": [lo, hi], "n": int(m.sum()),
                                      "fixed_mean": float(fixed[m].mean()),
                                      "fixed_median": float(np.median(fixed[m])),
                                      "orient_mean": float(orient_per_test[m].mean()),
                                      "orient_median": float(np.median(orient_per_test[m]))})

    print("\n=== Mode surrogate evaluation ===")
    print(f"(a) fixed-orientation (face-on) h+ match: mean={out['fixed_orientation']['mean']:.5f} "
          f"median={out['fixed_orientation']['median']:.5f} >0.99={out['fixed_orientation']['frac>0.99']:.3f}")
    print(f"    [old single-h+ model baseline: 0.988]")
    print(f"(b) orientation-averaged match: mean={out['orientation_averaged']['mean']:.5f} "
          f"median={out['orientation_averaged']['median']:.5f} >0.99={out['orientation_averaged']['frac>0.99']:.3f}")
    print(f"    amplitude ratio (surrogate/true, should be ~1): median={out['amplitude_ratio']['median']:.3f} "
          f"[p10={out['amplitude_ratio']['p10']:.3f}, p90={out['amplitude_ratio']['p90']:.3f}]")
    print("(c) per-mode median overlap:")
    for (l, m) in keys:
        print(f"      ({l:>2},{m:>2}): {out['per_mode_overlap_median'][f'{l},{m}']:.4f}")
    print("(d) match vs q:")
    for row in out["match_vs_q"]:
        print(f"      q in {row['q_range']}: n={row['n']} fixed={row['fixed_mean']:.4f} orient={row['orient_mean']:.4f}")

    (run / "eval").mkdir(exist_ok=True)
    with open(run / "eval" / "mode_eval_summary.json", "w") as fh:
        json.dump(out, fh, indent=2)
    # Persist raw per-point arrays so figures/medians can be remade without re-evaluating.
    np.savez(run / "eval" / "eval_arrays.npz", fixed=fixed, orient=orient,
             orient_per_test=orient_per_test, qs=qs, amp_ratio=amp_ratio, seed=args.seed)
    print(f"\nsaved -> {run/'eval'/'mode_eval_summary.json'}, eval_arrays.npz")

    # Figures (M6 analogues of the paper's Fig 1 / Fig 2)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bins = np.logspace(-5, 0, 40)
        # density=True so the two populations (orientation-averaged has n_orient x more
        # samples than fixed) are compared by shape, not by raw count.
        ax.hist(np.clip(1 - orient, 1e-5, 1), bins=bins, density=True, alpha=0.6,
                label=f"orientation-avg (median {np.median(orient):.3f})")
        ax.hist(np.clip(1 - fixed, 1e-5, 1), bins=bins, density=True, alpha=0.6,
                label=f"fixed face-on (median {np.median(fixed):.3f})")
        ax.set_xscale("log"); ax.set_xlabel("mismatch $1-\\mathcal{M}$")
        ax.set_ylabel("probability density")
        ax.set_title(f"Mode surrogate match (fixed $N={n_done}$, "
                     f"orient.\\ $N={len(orient)}$)"); ax.legend()
        fig.tight_layout(); fig.savefig(run / "eval" / "mismatch_hist.png", dpi=150); plt.close(fig)

        if overlay:
            # pick the case whose match is nearest the orientation-averaged median
            target = float(np.median(orient))
            mm, ht, hg, io, ph = min(overlay, key=lambda r: abs(r[0] - target))
            n = max(len(ht), len(hg))
            a = np.zeros(n); a[:len(ht)] = ht; bg = np.zeros(n); bg[:len(hg)] = hg
            Ht = np.fft.rfft(a); Hg = np.fft.rfft(bg)  # |h+(f)| is time-shift invariant
            fr2 = np.fft.rfftfreq(n, dt); b2 = (fr2 >= flo) & (fr2 <= fhi)
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.loglog(fr2[b2], np.abs(Ht[b2]), label="NRSur7dq4 truth", lw=2, alpha=0.8)
            ax.loglog(fr2[b2], np.abs(Hg[b2]), "--", label="mode surrogate", lw=1.5)
            ax.set_xlabel("f [Hz]"); ax.set_ylabel("|h+(f)|")
            ax.set_title(f"h+ overlay (iota={io:.2f}, phi={ph:.2f}, match={mm:.4f})"); ax.legend()
            fig.tight_layout(); fig.savefig(run / "eval" / "hplus_overlay.png", dpi=150); plt.close(fig)
        print(f"figures -> {run/'eval'}/mismatch_hist.png, hplus_overlay.png")
    except Exception as e:
        print(f"plotting skipped: {e}")


if __name__ == "__main__":
    main()
