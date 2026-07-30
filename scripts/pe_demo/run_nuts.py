"""Cross-injection NUTS recovery: inject the TRUE NRSur waveform, recover with the
differentiable surrogate likelihood. Zero-noise (Asimov) so any posterior offset
from the injected values is *pure surrogate systematic*, not noise scatter.

Data and template share the identical analytic SWSH projection; only the intrinsic
mode content differs (true NRSur modes vs network prediction). So this isolates the
network's mode error as the source of any bias -- exactly the question we want.

Runs in the Windows torch env (CUDA). PSD weights are precomputed (no pycbc needed).

    python scripts/pe_demo/run_nuts.py --check     # sanity only (SNR, match, logL)
    python scripts/pe_demo/run_nuts.py --warmup 300 --samples 500
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np
import torch

sys.path.append(os.getcwd())
import yaml
from src.physics.projection import load_torch_strain_model, project_strain_torch

RUN = "runs/modes_m5_600k_ft"
CFG = "config.modes_v1.yaml"
INJ_NPZ = "scripts/pe_demo/injection_true_modes.npz"
PSD_NPZ = "data/psd_weights_band.npz"
OUTDIR = "scripts/pe_demo/out"

# Sampled params and their uniform prior bounds; everything else fixed at injection.
SAMPLED = ["q", "chi1z", "chi2z", "iota", "lndL", "tc", "phi"]
BOUNDS = {
    "q": (1.0, 4.0), "chi1z": (-0.8, 0.8), "chi2z": (-0.8, 0.8),
    "iota": (0.0, np.pi), "lndL": (-1.5, 1.5), "tc": (-0.008, 0.008),
    "phi": (0.0, 2 * np.pi),
}


def build(device, dtype):
    cfg = yaml.safe_load(open(CFG))
    tsm, stats = load_torch_strain_model(RUN, cfg, device="cpu")
    tsm = tsm.to(device=device, dtype=dtype)
    pmin = torch.tensor(stats["param_min"], device=device, dtype=dtype)
    pmax = torch.tensor(stats["param_max"], device=device, dtype=dtype)
    freqs = tsm.freqs.to(dtype)
    df = float(freqs[1] - freqs[0])

    inj = np.load(INJ_NPZ)
    inj_keys = [k for k in inj["keys"]]
    key_to_row = {k: i for i, k in enumerate(inj_keys)}
    order = [key_to_row[f"{l}:{m}"] for (l, m) in tsm.keys]   # reorder to tsm/swsh order
    modes = (inj["modes_real"] + 1j * inj["modes_imag"])[order]
    true_modes = torch.tensor(modes, device=device, dtype=torch.complex128).to(
        torch.complex64 if dtype == torch.float32 else torch.complex128).unsqueeze(0)
    inj_p = {"params": inj["params"], "iota": float(inj["iota"]),
             "phi": float(inj["phi"]), "psi": float(inj["psi"])}

    w = np.load(PSD_NPZ)
    assert np.allclose(w["freqs"], freqs.cpu().numpy(), atol=1e-6), "PSD freq grid mismatch"
    weight = torch.tensor(w["weights"], device=device, dtype=dtype)   # normalized 1/Sn
    twopif = 2.0 * np.pi * freqs

    def project(modes_c, iota, phi, psi):
        y = tsm.swsh.forward(iota, phi).to(modes_c.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(0).expand(modes_c.shape[0], -1)
        hp, hc = project_strain_torch(modes_c, y)
        strain = torch.cos(2 * psi).unsqueeze(-1) * hp + torch.sin(2 * psi).unsqueeze(-1) * hc
        return torch.fft.rfft(strain, dim=-1)[:, tsm.band_mask]

    def inner(a, b):  # <a|b> = 4 df Re sum a b* w
        return 4.0 * df * torch.sum((a * b.conj()).real * weight, dim=-1)

    # --- injection data d0 (true modes), scaled to target SNR set later ---
    t = lambda x: torch.tensor(x, device=device, dtype=dtype)
    d0 = project(true_modes, t(inj_p["iota"]), t(inj_p["phi"]), t(inj_p["psi"]))[0]

    def template(full):  # full: dict name->tensor (sampled) ; returns band strain (complex)
        p = inj_p["params"]
        intr = torch.stack([
            full["q"], t(p[1]), t(p[2]), full["chi1z"],
            t(p[4]), t(p[5]), full["chi2z"], t(p[7])])
        pn = (2.0 * (intr - pmin) / (pmax - pmin + 1e-10) - 1.0).unsqueeze(0)
        H = tsm.strain_fd(pn, full["iota"].unsqueeze(0), full["phi"].unsqueeze(0),
                          t(inj_p["psi"]).unsqueeze(0))[0]
        H = H * torch.exp(-full["lndL"]) * torch.exp(1j * twopif * full["tc"])
        return H

    return dict(tsm=tsm, device=device, dtype=dtype, d0=d0, inner=inner,
                template=template, inj_p=inj_p, t=t, df=df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--snr", type=float, default=25.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--max-tree-depth", type=int, default=8)
    ap.add_argument("--float64", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = torch.float64 if args.float64 else torch.float32
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    B = build(device, dtype)
    d0, inner, template, inj_p, t = B["d0"], B["inner"], B["template"], B["inj_p"], B["t"]

    # scale data to target SNR (in the normalized-PSD inner product; shape == physical PSD)
    snr0 = torch.sqrt(inner(d0, d0)).item()
    A = args.snr / snr0
    d = (A * d0).detach()

    truth = {"q": inj_p["params"][0], "chi1z": inj_p["params"][3], "chi2z": inj_p["params"][6],
             "iota": inj_p["iota"], "lndL": 0.0, "tc": 0.0, "phi": inj_p["phi"]}

    def neg_match_and_ll(vals):
        full = {k: t(vals[k]) for k in SAMPLED}
        h = A * template(full)
        # rescale-and-time-maximised match is what eval reports; here use the plain
        # overlap at fixed phase/time (NUTS samples tc/phi explicitly).
        hh = inner(h, h); dd = inner(d, d)
        m = (inner(d, h) / torch.sqrt(hh * dd)).item()
        ll = (-0.5 * inner(d - h, d - h)).item()
        return m, ll, torch.sqrt(hh).item()

    m_t, ll_t, snr_h = neg_match_and_ll(truth)
    print(f"[truth] SNR(data)={args.snr:.1f}  SNR(template)={snr_h:.2f}  "
          f"match(d, h_inj)={m_t:.4f}  logL={ll_t:.2f}")
    print(f"  amplitude scale A={A:.3e}, natural data SNR units snr0={snr0:.3e}")
    if args.check:
        # also show logL sensitivity: perturb q to confirm a peak near truth
        for dq in (-0.1, -0.05, 0.0, 0.05, 0.1):
            v = dict(truth); v["q"] = truth["q"] + dq
            _, ll, _ = neg_match_and_ll(v)
            print(f"   q={v['q']:.3f}: logL={ll:.2f}")
        return

    # ---------------- NUTS ----------------
    import pyro, pyro.distributions as dist
    from pyro.infer import MCMC, NUTS
    from pyro.infer.autoguide.initialization import init_to_value
    os.makedirs(OUTDIR, exist_ok=True)

    def model():
        vals = {}
        for k in SAMPLED:
            lo, hi = BOUNDS[k]
            vals[k] = pyro.sample(k, dist.Uniform(t(lo), t(hi)))
        full = {k: vals[k] for k in SAMPLED}
        h = A * template(full)
        ll = -0.5 * inner(d - h, d - h)
        pyro.factor("loglik", ll)

    init = {k: t(float(truth[k])) for k in SAMPLED}
    kernel = NUTS(model, max_tree_depth=args.max_tree_depth, target_accept_prob=0.8,
                  init_strategy=init_to_value(values=init))
    mcmc = MCMC(kernel, num_samples=args.samples, warmup_steps=args.warmup,
                num_chains=1, disable_progbar=False)
    t0 = time.time()
    mcmc.run()
    dt = time.time() - t0
    samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}
    print(f"\nNUTS done in {dt:.1f}s ({args.samples} samples)")
    div = int(mcmc.diagnostics().get("divergences", {}).get("chain 0", []).__len__()) \
        if "divergences" in mcmc.diagnostics() else -1

    print(f"\n{'param':>7} {'truth':>9} {'post.mean':>10} {'post.std':>10} {'bias/sig':>9}")
    rows = {}
    for k in SAMPLED:
        s = samples[k]; mu, sd = float(s.mean()), float(s.std())
        bias_sig = (mu - truth[k]) / sd if sd > 0 else 0.0
        rows[k] = (truth[k], mu, sd, bias_sig)
        print(f"{k:>7} {truth[k]:>9.4f} {mu:>10.4f} {sd:>10.4f} {bias_sig:>9.2f}")

    np.savez(os.path.join(OUTDIR, "nuts_samples.npz"),
             **{f"s_{k}": samples[k] for k in SAMPLED},
             **{f"truth_{k}": truth[k] for k in SAMPLED},
             snr=args.snr, match_truth=m_t, walltime=dt)
    print(f"\nsaved -> {os.path.join(OUTDIR, 'nuts_samples.npz')}")


if __name__ == "__main__":
    main()
