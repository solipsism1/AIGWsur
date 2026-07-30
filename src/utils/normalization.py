"""Helpers for exact dataset normalization and path inference."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def infer_train_data_path(data_path: str | Path) -> Path:
    """Infer the training-set HDF5 path from a validation or test path."""
    data_path = Path(data_path)
    name = data_path.name
    if "_val.h5" in name:
        return data_path.with_name(name.replace("_val.h5", "_train.h5"))
    if "_test.h5" in name:
        return data_path.with_name(name.replace("_test.h5", "_train.h5"))
    return data_path


def compute_normalization_stats(
    waveform_h5_path: str | Path,
    compressor,
    n_samples: Optional[int] = None,
    chunk_size: int = 512,
) -> dict:
    """Compute exact coefficient and parameter normalization stats in chunks."""
    waveform_h5_path = Path(waveform_h5_path)
    with h5py.File(waveform_h5_path, "r") as f:
        params_ds = f["parameters"]
        real_ds = f["hp_real"]
        imag_ds = f["hp_imag"]

        total = len(params_ds) if n_samples is None else min(len(params_ds), n_samples)
        if total == 0:
            raise ValueError(f"No samples available in {waveform_h5_path}")

        param_dim = params_ds.shape[1]
        coeff_dim = 2 * compressor.n_basis

        param_min = np.full(param_dim, np.inf, dtype=np.float64)
        param_max = np.full(param_dim, -np.inf, dtype=np.float64)
        coeff_sum = np.zeros(coeff_dim, dtype=np.float64)
        coeff_sumsq = np.zeros(coeff_dim, dtype=np.float64)

        count = 0
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)

            params = params_ds[start:stop].astype(np.float64)
            real = real_ds[start:stop]
            imag = imag_ds[start:stop]

            coeffs_real, coeffs_imag = compressor.encode(real, imag)
            coeffs = np.concatenate([coeffs_real, coeffs_imag], axis=1).astype(np.float64)

            param_min = np.minimum(param_min, params.min(axis=0))
            param_max = np.maximum(param_max, params.max(axis=0))
            coeff_sum += coeffs.sum(axis=0)
            coeff_sumsq += np.square(coeffs).sum(axis=0)
            count += len(params)

        coeffs_mean = coeff_sum / count
        coeffs_var = coeff_sumsq / count - np.square(coeffs_mean)
        coeffs_std = np.sqrt(np.maximum(coeffs_var, 0.0)) + 1e-10

    return {
        "coeffs_mean": coeffs_mean.astype(np.float32),
        "coeffs_std": coeffs_std.astype(np.float32),
        "param_min": param_min.astype(np.float32),
        "param_max": param_max.astype(np.float32),
        "count": count,
        "source_path": str(waveform_h5_path),
    }
