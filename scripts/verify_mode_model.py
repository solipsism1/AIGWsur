"""M3 verification: per-mode head model forward shape + finite Jacobian.

Confirms the ModeResNetRegressor produces correctly shaped output and that autograd
returns a finite Jacobian w.r.t. the 8 physical inputs (needed for the PE / Fisher
claims). Runs in the Windows torch env.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.append(os.getcwd())

from src.data.mode_compression import PerModeCompressor
from src.data.mode_waveforms import mode_keys
from src.models.flow import build_model


def main():
    comp_path = sys.argv[1] if len(sys.argv) > 1 else r"D:/gw_waveform_data/mode_compressor_v1.h5"
    comp = PerModeCompressor.load(comp_path)
    keys = mode_keys()
    per_mode = {f"{l}:{m}": comp.n_basis[(l, m)] for (l, m) in keys}
    coeff_dim = sum(2 * comp.n_basis[k] for k in keys)
    print(f"compressor: {len(keys)} modes, coeff_dim={coeff_dim}")

    config = {
        "data": {"param_dim": 8},
        "compression": {"ell_max": 4, "include_m0": True, "per_mode_n_basis": per_mode},
        "model": {
            "type": "mode_resnet_regressor", "param_dim": 8, "hidden_dims": [512],
            "n_blocks": 6, "dropout": 0.0, "loss": "huber", "huber_delta": 0.5,
            "loss_weighting": "none", "fourier_dim": 128, "fourier_scale": 1.0,
        },
    }
    model = build_model(config)
    model.eval()

    x = torch.randn(4, 8)
    y = model(x)
    assert y.shape == (4, coeff_dim), f"bad output shape {y.shape}"
    print(f"forward OK: input {tuple(x.shape)} -> output {tuple(y.shape)}; finite={torch.isfinite(y).all().item()}")

    # Jacobian of the (mean) output w.r.t. one 8-vector input via autograd.
    xi = torch.randn(1, 8, requires_grad=True)
    yi = model(xi)
    jac = torch.autograd.functional.jacobian(lambda z: model(z).sum(0), xi)  # (coeff_dim?, ...)
    # full Jacobian via jacrev for shape (coeff_dim, 8)
    from torch.func import jacrev, functional_call
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}

    def f(single):
        return functional_call(model, (params, buffers), (single.unsqueeze(0),)).squeeze(0)

    J = jacrev(f)(torch.randn(8))
    print(f"Jacobian shape {tuple(J.shape)} (expect ({coeff_dim}, 8)); "
          f"finite={torch.isfinite(J).all().item()}; "
          f"||J||={J.norm().item():.3e}; nonzero_frac={(J.abs() > 0).float().mean().item():.3f}")
    ok = J.shape == (coeff_dim, 8) and torch.isfinite(J).all().item() and J.norm().item() > 0
    print("M3 GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
