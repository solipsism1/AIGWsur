"""Dataset of per-mode SVD coefficients for training the mode regressor.

Unlike WaveformDataset (which stream-encodes raw waveforms), the mode pipeline stores
coefficients directly (raw modes are too large), so this dataset just loads the
coefficient matrix, standardizes per channel, and normalizes parameters to [-1, 1].
Held-out splits must reuse the training-set statistics via apply_normalization_stats.
"""

from __future__ import annotations

from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ModeCoeffDataset(Dataset):
    def __init__(self, h5_path: str, n_samples: Optional[int] = None,
                 index_range: Optional[Tuple[int, int]] = None):
        with h5py.File(h5_path, "r") as f:
            coeffs = f["coeffs"]
            params = f["parameters"]
            if index_range is not None:
                lo, hi = index_range
                self.coeffs = coeffs[lo:hi].astype(np.float32)
                self.params = params[lo:hi].astype(np.float32)
            elif n_samples is not None:
                self.coeffs = coeffs[:n_samples].astype(np.float32)
                self.params = params[:n_samples].astype(np.float32)
            else:
                self.coeffs = coeffs[:].astype(np.float32)
                self.params = params[:].astype(np.float32)
            self.mode_keys = f.attrs.get("mode_keys", "")
            self.coeff_dim = int(f.attrs.get("coeff_dim", self.coeffs.shape[1]))

        self.coeffs_mean = self.coeffs.mean(axis=0)
        self.coeffs_std = self.coeffs.std(axis=0) + 1e-10
        self.param_min = self.params.min(axis=0)
        self.param_max = self.params.max(axis=0)
        self._build_tensors()

    def _build_tensors(self):
        std = (self.coeffs - self.coeffs_mean) / self.coeffs_std
        norm = 2.0 * (self.params - self.param_min) / (self.param_max - self.param_min + 1e-10) - 1.0
        self.coeffs_tensor = torch.from_numpy(std.astype(np.float32))
        self.params_tensor = torch.from_numpy(norm.astype(np.float32))

    def apply_normalization_stats(self, coeffs_mean, coeffs_std, param_min, param_max):
        self.coeffs_mean = np.asarray(coeffs_mean, dtype=np.float32)
        self.coeffs_std = np.asarray(coeffs_std, dtype=np.float32)
        self.param_min = np.asarray(param_min, dtype=np.float32)
        self.param_max = np.asarray(param_max, dtype=np.float32)
        self._build_tensors()

    def unstandardize_coeffs(self, coeffs_std: np.ndarray) -> np.ndarray:
        return coeffs_std * self.coeffs_std + self.coeffs_mean

    def unnormalize_params(self, params_norm: np.ndarray) -> np.ndarray:
        return (params_norm + 1.0) / 2.0 * (self.param_max - self.param_min) + self.param_min

    def __len__(self) -> int:
        return len(self.params)

    def __getitem__(self, idx):
        return self.params_tensor[idx], self.coeffs_tensor[idx]
