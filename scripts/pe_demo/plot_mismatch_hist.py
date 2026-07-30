"""Publication version of the mismatch histogram (paper Fig. 1).

Reads the saved per-point match arrays (no re-evaluation) and plots the two
distributions density-normalized -- so the fixed-orientation (N=150) and
orientation-averaged (N=150 x n_orient) populations are compared by shape, not
raw count -- with the empty low-mismatch range trimmed and no in-plot title.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "runs/modes_m5_600k_ft/eval/eval_arrays.npz"
OUT = "paper/figures/mismatch_hist.png"


def main():
    d = np.load(ARR)
    fixed, orient = d["fixed"], d["orient"]
    mm_fixed = np.clip(1 - fixed, 1e-4, 1)
    mm_orient = np.clip(1 - orient, 1e-4, 1)
    bins = np.logspace(-3, 0, 31)

    fig, ax = plt.subplots(figsize=(7, 4.3))
    # step outlines (with a faint fill) so the two overlapping populations stay
    # single-colored -- filled translucent bars blend into a misleading third color.
    ax.hist(mm_orient, bins=bins, density=True, histtype="stepfilled", lw=2,
            edgecolor="C0", facecolor="C0", alpha=0.15)
    ax.hist(mm_orient, bins=bins, density=True, histtype="step", lw=2, color="C0",
            label=f"orientation-averaged  (median {np.median(orient):.3f})")
    ax.hist(mm_fixed, bins=bins, density=True, histtype="stepfilled", lw=2,
            edgecolor="C1", facecolor="C1", alpha=0.15)
    ax.hist(mm_fixed, bins=bins, density=True, histtype="step", lw=2, color="C1",
            label=f"fixed, face-on  (median {np.median(fixed):.3f})")
    ax.axvline(1 - np.median(orient), color="C0", ls="--", lw=1)
    ax.axvline(1 - np.median(fixed), color="C1", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1)
    ax.set_xlabel(r"mismatch $1-\mathcal{M}$")
    ax.set_ylabel("probability density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved -> {OUT}  (fixed N={len(fixed)}, orient N={len(orient)})")


if __name__ == "__main__":
    main()
