"""Generate the paper's waveform figures for the mode surrogate.

(1) Whitened, peak-aligned time-domain h+(t) overlay for a clean low-q case and a
    high-q worst case. Whitening by the aLIGOZeroDetHighPower PSD suppresses the
    low-SNR high-frequency content, so the visual agreement matches the PSD-weighted
    match (unlike a log |h+(f)| view, which inflates physically negligible errors).
(2) Per-mode overlay: the well-learned (2,2)/(2,1) modes vs the data-limited
    (4,4)/(2,0) modes -- the figure that visualizes why the orientation-averaged
    match trails the fixed-orientation one.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/paper_figures.py --run runs/modes_m5_600k_ft
"""

from __future__ import annotations

import argparse, os, sys
import numpy as np, torch, yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from src.data.generate_waveforms import sample_parameters
from src.data.mode_compression import PerModeCompressor
from src.data.mode_waveforms import ModeGrid, generate_td_modes, mode_keys, project_to_strain, _waveform_kwargs
from src.models.flow import build_model
from src.physics.swsh import SWSHProjector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/modes_m5_600k_ft")
    ap.add_argument("--config", default="config.modes_v1.yaml")
    ap.add_argument("--n-scan", type=int, default=60)
    ap.add_argument("--seed", type=int, default=314)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    grid = ModeGrid(); dt = 1.0/grid.sample_rate; N = grid.n
    flo, fhi = float(config["data"]["f_lower"]), float(config["data"]["f_final"])
    keys = mode_keys()
    stats = np.load(os.path.join(args.run, "norm_stats.npz"))
    cmean, cstd, pmin, pmax = stats["coeffs_mean"], stats["coeffs_std"], stats["param_min"], stats["param_max"]
    net = build_model(config); net.load_state_dict(torch.load(os.path.join(args.run,"best_model.pt"), map_location="cpu")["model_state_dict"]); net.eval()
    comp = PerModeCompressor.load(config["compression"]["compressor_path"])
    swsh = SWSHProjector(keys)

    from pycbc.waveform import get_td_waveform
    from pycbc.types import TimeSeries
    from pycbc.psd import aLIGOZeroDetHighPower
    from pycbc.filter import match

    # whitening filter on the common-grid rFFT bins
    freqs = np.fft.rfftfreq(N, dt); df = freqs[1]-freqs[0]
    npos = int(round(fhi/df))+2
    psd = np.asarray(aLIGOZeroDetHighPower(npos, df, max(flo, df)))
    idx = np.clip((freqs/df).round().astype(int), 0, npos-1)
    pv = psd[idx]
    white = np.where((freqs>=flo)&(freqs<=fhi)&np.isfinite(pv)&(pv>0), 1.0/np.sqrt(pv), 0.0)

    def predict_modes(p):
        pn = 2.0*(p-pmin)/(pmax-pmin+1e-10)-1.0
        with torch.no_grad():
            s = net(torch.tensor(pn[None],dtype=torch.float32)).numpy()[0]
        return comp.decode((s*cstd+cmean)[None])[0]

    def whiten_align(h_true_td, h_surr_grid):
        """Pad/whiten both to length N, peak-align the surrogate to the truth by xcorr."""
        a = np.zeros(N); a[:min(len(h_true_td),N)] = h_true_td[:N]
        b = np.zeros(N); b[:min(len(h_surr_grid),N)] = h_surr_grid[:N]
        aw = np.fft.irfft(np.fft.rfft(a)*white, n=N)
        bw = np.fft.irfft(np.fft.rfft(b)*white, n=N)
        # align by circular cross-correlation peak
        xcorr = np.fft.irfft(np.fft.rfft(aw)*np.conj(np.fft.rfft(bw)), n=N)
        shift = int(np.argmax(xcorr)); shift = shift if shift < N//2 else shift-N
        bw = np.roll(bw, shift)
        return aw, bw

    def pycbc_match(h_true_td, h_surr_grid):
        n = max(len(h_true_td), N); a=np.zeros(n); a[:len(h_true_td)]=h_true_td; b=np.zeros(n); b[:N]=h_surr_grid
        A=TimeSeries(a,delta_t=dt).to_frequencyseries(); B=TimeSeries(b,delta_t=dt).to_frequencyseries()
        p=aLIGOZeroDetHighPower(len(A),A.delta_f,flo)
        return float(match(A,B,psd=p,low_frequency_cutoff=flo,high_frequency_cutoff=fhi)[0])

    # scan test points; record a representative orientation match
    import h5py
    with h5py.File("/mnt/d/gw_waveform_data/nrsur7dq4_omega_8d_v3/waveforms_val.h5","r") as f:
        pool = f["reference_omega_source_pool"][:]
    rng = np.random.default_rng(args.seed)
    params = sample_parameters(int(args.n_scan*2.4), config, seed=args.seed); params[:,7]=rng.choice(pool,size=len(params))
    cases = []
    for p in params:
        if len([c for c in cases]) >= args.n_scan: break
        tm = generate_td_modes(p, config, grid, modes=keys)
        if tm is None: continue
        pm = predict_modes(p)
        iota = float(np.arccos(rng.uniform(-1,1))); phi = float(rng.uniform(0,2*np.pi))
        kw = _waveform_kwargs(p, config)
        hp,_ = get_td_waveform(delta_t=dt, inclination=iota, coa_phase=np.pi/2-phi, **kw)
        y = swsh.forward(torch.tensor(iota), torch.tensor(phi)).numpy()
        hpl,_ = project_to_strain(pm, y)
        m = pycbc_match(np.asarray(hp.data), hpl)
        cases.append(dict(q=float(p[0]), iota=iota, phi=phi, match=m,
                          htrue=np.asarray(hp.data).copy(), hsurr=hpl.copy(), tmodes=tm, pmodes=pm))

    lowq = [c for c in cases if c["q"] < 2.0]
    highq = [c for c in cases if c["q"] > 3.3]
    best = max(lowq or cases, key=lambda c: c["match"])
    worst = min(highq or cases, key=lambda c: c["match"])

    # --- Figure 1: whitened TD overlay, best + worst, merger zoom ---
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for ax, c, tag in [(axes[0], best, "best, low $q$"), (axes[1], worst, "worst, high $q$")]:
        aw, bw = whiten_align(c["htrue"], c["hsurr"])
        pk = int(np.argmax(np.abs(aw)))
        lo, hi = max(0, pk-1400), min(len(aw), pk+250)
        t = (np.arange(lo, hi)-pk)*dt*1e3
        ax.plot(t, aw[lo:hi], color="C0", lw=1.8, alpha=0.85, label="NRSur7dq4")
        ax.plot(t, bw[lo:hi], "--", color="C1", lw=1.3, label="surrogate")
        ax.set_ylabel("whitened $h_+$")
        ax.text(0.02, 0.92, f"{tag}: $q={c['q']:.2f}$, $\\iota={c['iota']:.2f}$, $\\mathcal{{M}}={c['match']:.3f}$",
                transform=ax.transAxes, fontsize=9, va="top")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("$t - t_{\\rm peak}$ [ms]")
    fig.tight_layout(); fig.savefig(os.path.join(args.run,"eval","hplus_whitened.png"), dpi=150); plt.close(fig)

    # --- Figure 2: per-mode overlay (well-learned vs data-limited) ---
    c = best
    panel_modes = [(2,2), (2,1), (4,4), (2,0)]
    labels = {(2,2):"$(2,2)$ dominant", (2,1):"$(2,1)$", (4,4):"$(4,4)$ subdominant", (2,0):"$(2,0)$ memory"}
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
    for ax, k in zip(axes.flat, panel_modes):
        mi = keys.index(k)
        tt = c["tmodes"][mi]; pp = c["pmodes"][mi]
        pk2 = int(np.argmax(np.abs(c["tmodes"][keys.index((2,2))])))
        lo, hi = max(0, pk2-1400), min(len(tt), pk2+250)
        t = (np.arange(lo, hi)-pk2)*dt*1e3
        ax.plot(t, np.real(tt[lo:hi]), color="C0", lw=1.6, alpha=0.85, label="truth")
        ax.plot(t, np.real(pp[lo:hi]), "--", color="C1", lw=1.2, label="surrogate")
        ax.set_title(labels[k], fontsize=10)
        ax.set_xlabel("$t-t_{\\rm peak}$ [ms]"); ax.set_ylabel("$\\mathrm{Re}\\,h_{\\ell m}$")
    axes[0,0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"Per-mode reconstruction ($q={c['q']:.2f}$)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(args.run,"eval","per_mode_overlay.png"), dpi=150); plt.close(fig)

    print(f"best: q={best['q']:.2f} iota={best['iota']:.2f} match={best['match']:.4f}")
    print(f"worst: q={worst['q']:.2f} iota={worst['iota']:.2f} match={worst['match']:.4f}")
    print(f"saved -> {args.run}/eval/hplus_whitened.png, per_mode_overlay.png")


if __name__ == "__main__":
    main()
