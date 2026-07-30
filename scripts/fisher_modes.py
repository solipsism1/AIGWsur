"""M6: Fisher matrix over the full intrinsic + extrinsic parameter set.

The mode surrogate makes the strain differentiable in every parameter:
  * intrinsic (q, chi1, chi2, omega0)  -> autodiff through the network,
  * extrinsic (iota, phi, psi, ln dL, tc) -> autodiff through the analytic projection.

Fisher_ij = 4 Re sum_f (d_i h)(d_j h)* / S_n(f) * df, with h the (single-detector)
strain F+ h+ + Fx h_x scaled by 1/dL and time-shifted by tc. We verify the autodiff
derivatives against finite differences before assembling the Fisher matrix.

Run in the WSL PyCBC env:
    LAL_DATA_PATH=/home/beka/lal_data python scripts/fisher_modes.py \
        --run runs/modes_smoke --config config.modes_v1.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
import yaml

sys.path.append(os.getcwd())

from src.physics.projection import load_torch_strain_model

PARAM_NAMES = ["q", "chi1x", "chi1y", "chi1z", "chi2x", "chi2y", "chi2z", "omega0",
               "iota", "phi", "psi", "ln_dL", "tc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="config.modes_v1.yaml")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    torch.set_default_dtype(torch.float64)
    tsm, stats = load_torch_strain_model(args.run, config, device="cpu")
    tsm = tsm.double()
    pmin = torch.tensor(stats["param_min"], dtype=torch.float64)
    pmax = torch.tensor(stats["param_max"], dtype=torch.float64)
    freqs = tsm.freqs.double()
    flo = float(config["data"]["f_lower"]); fhi = float(config["data"]["f_final"])

    from pycbc.psd import aLIGOZeroDetHighPower
    df = float(freqs[1] - freqs[0])
    npos = int(round(fhi / df)) + 1
    psd_full = np.asarray(aLIGOZeroDetHighPower(npos, df, max(flo, df)))
    idx = np.clip((freqs.numpy() / df).round().astype(int), 0, npos - 1)
    Sn_np = psd_full[idx]
    inv_Sn = np.where(np.isfinite(Sn_np) & (Sn_np > 0), 1.0 / Sn_np, 0.0)
    Sn = torch.tensor(Sn_np, dtype=torch.float64)
    inv_Sn_t = torch.tensor(inv_Sn, dtype=torch.float64)

    # A representative intrinsic point + face-on-ish orientation.
    rng = np.random.default_rng(args.seed)
    theta0 = torch.tensor([2.0, 0.1, 0.05, 0.3, -0.1, 0.05, -0.2, 0.021,
                           0.9, 1.2, 0.5, 0.0, 0.0], dtype=torch.float64)
    lndL0 = 0.0
    twopif = 2.0 * np.pi * freqs

    def strain(full):
        intr = full[:8]
        iota, phi, psi, lndL, tc = full[8], full[9], full[10], full[11], full[12]
        pn = 2.0 * (intr - pmin) / (pmax - pmin + 1e-10) - 1.0
        Hf = tsm.strain_fd(pn.unsqueeze(0), iota.unsqueeze(0), phi.unsqueeze(0),
                           psi.unsqueeze(0))[0]
        Hf = Hf * torch.exp(-(lndL - lndL0)) * torch.exp(1j * twopif * tc)
        return torch.cat([Hf.real, Hf.imag])

    h0 = strain(theta0)
    n_band = h0.shape[0] // 2
    print(f"strain dim: {n_band} band bins; SNR check (4 df sum |h|^2/Sn):", end=" ")
    hc = h0[:n_band] + 1j * h0[n_band:]
    snr2 = 4 * df * torch.sum(torch.abs(hc) ** 2 * inv_Sn_t)
    print(f"{torch.sqrt(snr2).item():.2f}")

    # Autodiff Jacobian (2 n_band, 13)
    J = torch.autograd.functional.jacobian(strain, theta0)
    dH = (J[:n_band] + 1j * J[n_band:])  # (n_band, 13)

    # Finite-difference verification per parameter (overlap of derivative vectors)
    print("\nAutodiff vs finite-difference derivative (normalized overlap):")
    eps_scale = {0: 1e-3, 7: 1e-5, 11: 1e-3, 12: 1e-4}
    worst = 1.0
    for i in range(13):
        eps = eps_scale.get(i, 1e-4)
        tp = theta0.clone(); tp[i] += eps
        tm = theta0.clone(); tm[i] -= eps
        fd = (strain(tp) - strain(tm)) / (2 * eps)
        fdc = fd[:n_band] + 1j * fd[n_band:]
        ad = dH[:, i]
        num = torch.abs(torch.vdot(ad, fdc))
        den = torch.sqrt(torch.vdot(ad, ad).real * torch.vdot(fdc, fdc).real) + 1e-300
        ov = (num / den).item()
        worst = min(worst, ov)
        print(f"  {PARAM_NAMES[i]:>7}: overlap={ov:.5f}")
    print(f"worst derivative overlap: {worst:.5f}  ({'PASS' if worst > 0.999 else 'CHECK'})")

    # Assemble Fisher: 4 Re sum dH_i conj(dH_j)/Sn df
    weight = (4.0 * df * inv_Sn_t).to(torch.complex128)
    F = torch.real(torch.einsum("fi,fj->ij", dH.conj() * weight.unsqueeze(1), dH))
    F = F.numpy()
    print("\nFisher matrix assembled (13x13). diag:")
    for i in range(13):
        print(f"  {PARAM_NAMES[i]:>7}: F_ii={F[i,i]:.3e}")
    # Regularize for inversion display
    try:
        cond = np.linalg.cond(F)
        cov = np.linalg.inv(F)
        sig = np.sqrt(np.clip(np.diag(cov), 0, None))
        print(f"\ncondition number: {cond:.3e}")
        print("1-sigma marginalized uncertainties:")
        for i in range(13):
            print(f"  {PARAM_NAMES[i]:>7}: {sig[i]:.3e}")
    except np.linalg.LinAlgError as e:
        print("Fisher inversion failed:", e)

    os.makedirs(os.path.join(args.run, "eval"), exist_ok=True)
    np.savez(os.path.join(args.run, "eval", "fisher.npz"), fisher=F, params=theta0.numpy(),
             param_names=np.array(PARAM_NAMES))
    print(f"\nsaved -> {os.path.join(args.run, 'eval', 'fisher.npz')}")


if __name__ == "__main__":
    main()
