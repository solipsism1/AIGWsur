"""Product-level orientation figure: the SAME binary, face-on vs inclined.

Replaces the per-mode overlay (an internal quantity) with a final-product view of
the orientation limitation: viewed face-on the surrogate h+(t) is faithful; viewed
inclined -- where the subdominant modes contribute -- it degrades. This is the
observable-strain version of the same point Table I makes with the
fixed-orientation vs orientation-averaged columns.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/pe_demo/fig_orientation.py \
        --run runs/modes_m5_600k_ft
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

IOTA_FACE = 0.0       # face-on: fixed-orientation quantity (only m=+-2 contribute)
IOTA_INCL = 1.3       # inclined ~75 deg: subdominant modes contribute most
PHI = 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/modes_m5_600k_ft")
    ap.add_argument("--config", default="config.modes_v1.yaml")
    ap.add_argument("--n-scan", type=int, default=90)
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--out", default="paper/figures/hplus_orientation.png")
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
        a = np.zeros(N); a[:min(len(h_true_td),N)] = h_true_td[:N]
        b = np.zeros(N); b[:min(len(h_surr_grid),N)] = h_surr_grid[:N]
        aw = np.fft.irfft(np.fft.rfft(a)*white, n=N)
        bw = np.fft.irfft(np.fft.rfft(b)*white, n=N)
        xcorr = np.fft.irfft(np.fft.rfft(aw)*np.conj(np.fft.rfft(bw)), n=N)
        shift = int(np.argmax(xcorr)); shift = shift if shift < N//2 else shift-N
        return aw, np.roll(bw, shift)

    def m_at(p, pm, iota):
        kw = _waveform_kwargs(p, config)
        hp,_ = get_td_waveform(delta_t=dt, inclination=iota, coa_phase=np.pi/2-PHI, **kw)
        y = swsh.forward(torch.tensor(iota), torch.tensor(PHI)).numpy()
        hpl,_ = project_to_strain(pm, y)
        ht = np.asarray(hp.data)
        n = max(len(ht), N); a=np.zeros(n); a[:len(ht)]=ht; b=np.zeros(n); b[:N]=hpl
        A=TimeSeries(a,delta_t=dt).to_frequencyseries(); B=TimeSeries(b,delta_t=dt).to_frequencyseries()
        pp=aLIGOZeroDetHighPower(len(A),A.delta_f,flo)
        mval = float(match(A,B,psd=pp,low_frequency_cutoff=flo,high_frequency_cutoff=fhi)[0])
        return mval, ht, hpl

    import h5py
    with h5py.File("/mnt/d/gw_waveform_data/nrsur7dq4_omega_8d_v3/waveforms_val.h5","r") as f:
        pool = f["reference_omega_source_pool"][:]
    rng = np.random.default_rng(args.seed)
    params = sample_parameters(int(args.n_scan*2.4), config, seed=args.seed); params[:,7]=rng.choice(pool,size=len(params))

    # pick the case with the largest face-on -> inclined match drop, among moderate-q
    # points whose face-on match is already excellent (so the drop is purely orientation).
    best = None; n_done = 0
    for p in params:
        if n_done >= args.n_scan: break
        tm = generate_td_modes(p, config, grid, modes=keys)
        if tm is None: continue
        n_done += 1
        if not (1.8 <= p[0] <= 3.2): continue
        pm = predict_modes(p)
        mf, htf, hsf = m_at(p, pm, IOTA_FACE)
        if mf < 0.975: continue
        mi, hti, hsi = m_at(p, pm, IOTA_INCL)
        drop = mf - mi
        cand = dict(q=float(p[0]), mf=mf, mi=mi, drop=drop,
                    htf=htf, hsf=hsf, hti=hti, hsi=hsi)
        if best is None or drop > best["drop"]:
            best = cand
    if best is None:
        raise SystemExit("no suitable case found; widen the scan")
    c = best
    print(f"chosen: q={c['q']:.2f}  face-on M={c['mf']:.3f}  inclined M={c['mi']:.3f}  drop={c['drop']:.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    panels = [(axes[0], c["htf"], c["hsf"], IOTA_FACE, c["mf"], "face-on"),
              (axes[1], c["hti"], c["hsi"], IOTA_INCL, c["mi"], "inclined")]
    for ax, ht, hs, iota, m, tag in panels:
        aw, bw = whiten_align(ht, hs)
        pk = int(np.argmax(np.abs(aw)))
        lo, hi = max(0, pk-1400), min(len(aw), pk+250)
        t = (np.arange(lo, hi)-pk)*dt*1e3
        ax.plot(t, aw[lo:hi], color="C0", lw=1.8, alpha=0.85, label="NRSur7dq4")
        ax.plot(t, bw[lo:hi], "--", color="C1", lw=1.3, label="surrogate")
        ax.set_ylabel("whitened $h_+$")
        ax.text(0.02, 0.92,
                f"{tag}: $\\iota={iota:.2f}$, $\\mathcal{{M}}={m:.3f}$",
                transform=ax.transAxes, fontsize=10, va="top")
        ax.legend(loc="upper right", fontsize=8)
    axes[0].text(0.98, 0.06, f"$q={c['q']:.2f}$ (same binary)", transform=axes[0].transAxes,
                 fontsize=9, ha="right", va="bottom", color="0.4")
    axes[1].set_xlabel("$t - t_{\\rm peak}$ [ms]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150); plt.close(fig)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
