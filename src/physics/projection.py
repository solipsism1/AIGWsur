"""Differentiable coefficient -> strain pipeline in PyTorch.

Wraps a fitted PerModeCompressor and the SWSH projector so the whole map

    standardized coeffs -> raw coeffs -> per-mode amp/phase (or re/im)
                        -> complex modes h_lm(t) -> h_+(t), h_x(t) -> h+(f)

is autograd-differentiable. This powers (i) M5 match-based fine-tuning (backprop a
PSD-weighted mismatch over sampled orientations) and (ii) the M6 Fisher matrix
(intrinsic derivatives through the net, extrinsic through this analytic projection).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch

from src.data.mode_compression import PerModeCompressor, mode_representation
from src.physics.swsh import SWSHProjector


class TorchModeDecoder(torch.nn.Module):
    """Decode a standardized coefficient vector to complex modes on the time grid."""

    def __init__(self, compressor: PerModeCompressor,
                 coeffs_mean: np.ndarray, coeffs_std: np.ndarray, dtype=torch.float32):
        super().__init__()
        self.keys: List[Tuple[int, int]] = list(compressor.keys)
        self.slices = compressor.mode_slices()
        self.reps = {k: mode_representation(k) for k in self.keys}
        self.register_buffer("coeffs_mean", torch.as_tensor(coeffs_mean, dtype=dtype))
        self.register_buffer("coeffs_std", torch.as_tensor(coeffs_std, dtype=dtype))
        for k in self.keys:
            s0, s1 = compressor.svd[k]
            kk = f"{k[0]}_{k[1]}"
            for ci, sv in enumerate((s0, s1)):
                self.register_buffer(f"{kk}_c{ci}_mean", torch.as_tensor(sv.mean, dtype=dtype))
                self.register_buffer(f"{kk}_c{ci}_std", torch.as_tensor(sv.std, dtype=dtype))
                self.register_buffer(f"{kk}_c{ci}_basis", torch.as_tensor(sv.basis, dtype=dtype))

    def forward(self, std_coeffs: torch.Tensor) -> torch.Tensor:
        """``std_coeffs``: ``(B, coeff_dim)`` -> complex modes ``(B, n_modes, N)``."""
        raw = std_coeffs * self.coeffs_std + self.coeffs_mean
        cols = []
        for k in self.keys:
            kk = f"{k[0]}_{k[1]}"
            sl0, sl1 = self.slices[k]
            ch = []
            for ci, sl in ((0, sl0), (1, sl1)):
                mean = getattr(self, f"{kk}_c{ci}_mean")
                std = getattr(self, f"{kk}_c{ci}_std")
                basis = getattr(self, f"{kk}_c{ci}_basis")
                ch.append((raw[:, sl] @ basis) * std + mean)
            if self.reps[k] == "reim":
                mode = torch.complex(ch[0], ch[1])
            else:
                mode = ch[0] * torch.exp(1j * ch[1].to(torch.complex64 if ch[1].dtype == torch.float32 else torch.complex128))
            cols.append(mode)
        return torch.stack(cols, dim=1)


class FastModeDecoder(torch.nn.Module):
    """Fused decoder: all 21 modes in batched einsums instead of a Python loop.

    Mathematically identical to TorchModeDecoder, but pads every mode's per-channel
    SVD basis to a common width and decodes them in two batched matmuls, collapsing
    ~100 tiny kernel launches into a handful. This removes the fixed-overhead that
    dominated single-waveform latency. The amp/phase nonlinearity (and the re/im
    branch for m=0 memory modes) is applied vectorized with a mask.
    """

    def __init__(self, compressor: PerModeCompressor,
                 coeffs_mean: np.ndarray, coeffs_std: np.ndarray, dtype=torch.float32):
        super().__init__()
        keys = list(compressor.keys)
        slices = compressor.mode_slices()
        M = len(keys)
        max_nb = max(compressor.n_basis[k] for k in keys)
        N = compressor.svd[keys[0]][0].mean.shape[0]
        coeff_dim = compressor.coeff_dim

        self.register_buffer("coeffs_mean", torch.as_tensor(coeffs_mean, dtype=dtype))
        self.register_buffer("coeffs_std", torch.as_tensor(coeffs_std, dtype=dtype))
        # padded bases (M, max_nb, N) and per-(M,N) mean/std for both channels
        basis0 = np.zeros((M, max_nb, N), np.float64); basis1 = np.zeros((M, max_nb, N), np.float64)
        mean0 = np.zeros((M, N)); std0 = np.ones((M, N)); mean1 = np.zeros((M, N)); std1 = np.ones((M, N))
        # gather indices into [raw | 0] (coeff_dim is the zero-pad slot)
        idx0 = np.full((M, max_nb), coeff_dim, np.int64); idx1 = np.full((M, max_nb), coeff_dim, np.int64)
        reim = np.zeros(M, bool)
        for mi, k in enumerate(keys):
            nb = compressor.n_basis[k]
            s0, s1 = compressor.svd[k]
            basis0[mi, :nb] = s0.basis; basis1[mi, :nb] = s1.basis
            mean0[mi] = s0.mean; std0[mi] = s0.std; mean1[mi] = s1.mean; std1[mi] = s1.std
            sl0, sl1 = slices[k]
            idx0[mi, :nb] = np.arange(sl0.start, sl0.stop)
            idx1[mi, :nb] = np.arange(sl1.start, sl1.stop)
            reim[mi] = (mode_representation(k) == "reim")
        self.register_buffer("basis0", torch.as_tensor(basis0, dtype=dtype))
        self.register_buffer("basis1", torch.as_tensor(basis1, dtype=dtype))
        self.register_buffer("mean0", torch.as_tensor(mean0, dtype=dtype))
        self.register_buffer("std0", torch.as_tensor(std0, dtype=dtype))
        self.register_buffer("mean1", torch.as_tensor(mean1, dtype=dtype))
        self.register_buffer("std1", torch.as_tensor(std1, dtype=dtype))
        self.register_buffer("idx0", torch.as_tensor(idx0))
        self.register_buffer("idx1", torch.as_tensor(idx1))
        self.register_buffer("reim", torch.as_tensor(reim))

    def forward(self, std_coeffs: torch.Tensor) -> torch.Tensor:
        raw = std_coeffs * self.coeffs_std + self.coeffs_mean
        zeros = torch.zeros(raw.shape[0], 1, dtype=raw.dtype, device=raw.device)
        rawp = torch.cat([raw, zeros], dim=1)               # (B, coeff_dim+1)
        c0 = rawp[:, self.idx0]                              # (B, M, max_nb)
        c1 = rawp[:, self.idx1]
        ch0 = torch.einsum("bmk,mkn->bmn", c0, self.basis0) * self.std0 + self.mean0  # (B, M, N)
        ch1 = torch.einsum("bmk,mkn->bmn", c1, self.basis1) * self.std1 + self.mean1
        cdtype = torch.complex64 if ch0.dtype == torch.float32 else torch.complex128
        ap = ch0.to(cdtype) * torch.exp(1j * ch1.to(cdtype))   # amp/phase modes
        ri = torch.complex(ch0, ch1)                           # re/im modes
        mask = self.reim.view(1, -1, 1)
        return torch.where(mask, ri, ap)


def project_strain_torch(modes_complex: torch.Tensor, swsh_vec: torch.Tensor):
    """``modes_complex`` ``(B, n_modes, N)``, ``swsh_vec`` ``(B, n_modes)`` complex.

    Returns ``(h_plus, h_cross)`` real tensors ``(B, N)``; ``z=sum_lm h_lm Y_lm``,
    ``h_+ = Re z``, ``h_x = -Im z``.
    """
    z = torch.einsum("bm,bmn->bn", swsh_vec, modes_complex)
    return z.real, -z.imag


class TorchStrainModel(torch.nn.Module):
    """params -> h+(f) (band-limited), fully differentiable, for Fisher / match loss."""

    def __init__(self, net, decoder: TorchModeDecoder, keys, sample_rate: float,
                 f_lower: float, f_final: float, grid_n: int):
        super().__init__()
        self.net = net
        self.decoder = decoder
        self.swsh = SWSHProjector(keys)
        self.keys = list(keys)
        self.sample_rate = float(sample_rate)
        self.grid_n = int(grid_n)
        freqs = np.fft.rfftfreq(self.grid_n, 1.0 / self.sample_rate)
        band = (freqs >= f_lower) & (freqs <= f_final)
        self.register_buffer("band_mask", torch.as_tensor(band))
        self.register_buffer("freqs", torch.as_tensor(freqs[band], dtype=torch.float32))

    def strain_from_std_coeffs(self, std_coeffs: torch.Tensor, iota: torch.Tensor,
                               phi: torch.Tensor, psi: torch.Tensor = None) -> torch.Tensor:
        """Band-limited strain directly from standardized coefficients (bypasses the net).

        Used for the match fine-tuning target: decode the true dataset coeffs and project
        at the same orientation as the prediction.
        """
        modes = self.decoder(std_coeffs)
        y = self.swsh.forward(iota, phi).to(modes.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(0).expand(modes.shape[0], -1)
        hp, hc = project_strain_torch(modes, y)
        strain = hp if psi is None else (torch.cos(2 * psi).unsqueeze(-1) * hp
                                         + torch.sin(2 * psi).unsqueeze(-1) * hc)
        return torch.fft.rfft(strain, dim=-1)[:, self.band_mask]

    def strain_fd(self, params_norm: torch.Tensor, iota: torch.Tensor, phi: torch.Tensor,
                  psi: torch.Tensor = None) -> torch.Tensor:
        """Return band-limited h+(f) (or detector strain if psi given), complex ``(B, n_band)``."""
        std = self.net(params_norm)
        modes = self.decoder(std)
        y = self.swsh.forward(iota, phi).to(modes.dtype)
        if y.ndim == 1:
            y = y.unsqueeze(0).expand(modes.shape[0], -1)
        hp, hc = project_strain_torch(modes, y)
        if psi is None:
            strain = hp
        else:
            c = torch.cos(2 * psi).unsqueeze(-1); s = torch.sin(2 * psi).unsqueeze(-1)
            strain = c * hp + s * hc
        Hf = torch.fft.rfft(strain, dim=-1)
        return Hf[:, self.band_mask]


def load_torch_strain_model(run_dir, config, device="cpu", fast=True):
    """Convenience loader: build net + decoder from a run dir and config.

    ``fast=True`` uses the fused FastModeDecoder (same math, ~10x lower single-waveform
    latency); set False for the reference loop decoder.
    """
    import yaml as _yaml
    from pathlib import Path
    from src.models.flow import build_model
    run = Path(run_dir)
    stats = np.load(run / "norm_stats.npz")
    ck = torch.load(run / "best_model.pt", map_location=device)
    net = build_model(config); net.load_state_dict(ck["model_state_dict"]); net.eval()
    comp = PerModeCompressor.load(config["compression"]["compressor_path"])
    Dec = FastModeDecoder if fast else TorchModeDecoder
    dec = Dec(comp, stats["coeffs_mean"], stats["coeffs_std"])
    keys = comp.keys
    grid_n = comp.svd[keys[0]][0].mean.shape[0]
    tsm = TorchStrainModel(net, dec, keys, config["data"]["sample_rate"],
                           config["data"]["f_lower"], config["data"]["f_final"], grid_n)
    return tsm.to(device), stats
