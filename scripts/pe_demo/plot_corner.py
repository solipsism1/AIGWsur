"""Corner plot + bias table for the cross-injection NUTS recovery.

Reads scripts/pe_demo/out/nuts_samples.npz and overlays the injected truth.
Bias is reported in units of the marginal posterior sigma (the figure of merit:
|posterior mean - injected| / posterior std). <~1 sigma => no significant bias.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner

NPZ = "scripts/pe_demo/out/nuts_samples.npz"
SAMPLED = ["q", "chi1z", "chi2z", "iota", "lndL", "tc", "phi"]
LABELS = {"q": r"$q$", "chi1z": r"$\chi_{1z}$", "chi2z": r"$\chi_{2z}$",
          "iota": r"$\iota$", "lndL": r"$\ln d_L$", "tc": r"$t_c$ [s]", "phi": r"$\varphi$"}


def main():
    d = np.load(NPZ)
    samples = np.stack([d[f"s_{k}"] for k in SAMPLED], axis=1)
    truths = [float(d[f"truth_{k}"]) for k in SAMPLED]
    labels = [LABELS[k] for k in SAMPLED]

    print(f"SNR={float(d['snr']):.1f}  match(d,h_inj)={float(d['match_truth']):.4f}  "
          f"walltime={float(d['walltime']):.1f}s  N={samples.shape[0]}")
    print(f"\n{'param':>7} {'truth':>10} {'mean':>10} {'std':>10} {'bias/sig':>9}")
    worst = 0.0
    for i, k in enumerate(SAMPLED):
        mu, sd = samples[:, i].mean(), samples[:, i].std()
        b = (mu - truths[i]) / sd if sd > 0 else 0.0
        worst = max(worst, abs(b))
        print(f"{k:>7} {truths[i]:>10.4f} {mu:>10.4f} {sd:>10.4f} {b:>9.2f}")
    print(f"\nworst |bias|/sigma = {worst:.2f}  "
          f"({'NO significant bias' if worst < 1.0 else 'BIAS PRESENT'})")

    # Full diagnostic corner (all 7 params), undersampled -> smoothed.
    fig = corner.corner(samples, labels=labels, truths=truths,
                        truth_color="C3", show_titles=True, smooth=1.0, bins=22,
                        title_fmt=".3f", quantiles=[0.16, 0.5, 0.84],
                        title_kwargs={"fontsize": 9}, label_kwargs={"fontsize": 11})
    fig.suptitle(f"Cross-injection NUTS recovery (true NRSur into surrogate, "
                 f"zero noise, SNR={float(d['snr']):.0f})", fontsize=11)
    out = "scripts/pe_demo/out/corner.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved -> {out}")

    # Focused paper figure: physically meaningful params only (drop tc, phi nuisances).
    FOCUS = ["q", "chi1z", "chi2z", "iota", "lndL"]
    fi = [SAMPLED.index(k) for k in FOCUS]
    fig2 = corner.corner(samples[:, fi], labels=[LABELS[k] for k in FOCUS],
                         truths=[truths[i] for i in fi], truth_color="C3",
                         show_titles=True, smooth=1.0, bins=22, title_fmt=".3f",
                         quantiles=[0.16, 0.5, 0.84], color="C0",
                         title_kwargs={"fontsize": 10}, label_kwargs={"fontsize": 13})
    out2 = "paper/figures/pe_corner.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"saved -> {out2}")


if __name__ == "__main__":
    main()
