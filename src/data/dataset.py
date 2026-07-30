"""
PyTorch Dataset for compressed gravitational waveform coefficients.

Loads an HDF5 file of NRSur7dq4 waveforms, compresses each waveform to SVD
coefficients, and returns (parameters, coefficients) pairs for flow training.

Normalization conventions:
  - Parameters are mapped to [-1, 1] using per-dataset min/max.
  - SVD coefficients are standardized using per-dataset statistics.
  - Held-out splits must reuse training-set statistics for evaluation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.compression import WaveformCompressor


class WaveformDataset(Dataset):
    """Dataset of normalized parameter vectors and waveform coefficients."""

    def __init__(
        self,
        waveform_h5_path: str,
        compressor: Optional[WaveformCompressor] = None,
        compressor_path: Optional[str] = None,
        fit_compressor: bool = False,
        n_basis: int = 100,
        n_samples: Optional[int] = None,
        compressor_fit_n_samples: Optional[int] = None,
        compressor_fit_strategy: str = "all",
        compressor_fit_seed: int = 42,
        compressor_svd_solver: str = "auto",
        compressor_representation: str = "amp_phase",
        compressor_greedy_extra_basis: int = 0,
    ):
        # Load only params and freqs upfront — waveforms are stream-encoded below.
        with h5py.File(waveform_h5_path, "r") as f:
            if n_samples is not None:
                self.params = f["parameters"][:n_samples]
            else:
                self.params = f["parameters"][:]
            self.freqs = f["frequencies"][:]

        self._waveform_h5_path = waveform_h5_path
        self._n_samples = n_samples  # None means all
        n_total = len(self.params)

        self.compressor_fit_indices = None
        self.compressor_fit_info = None

        if compressor is not None:
            self.compressor = compressor
        elif compressor_path is not None:
            print(f"  Loading compressor from {compressor_path}...")
            self.compressor = WaveformCompressor.load(compressor_path)
        elif fit_compressor:
            self.compressor = WaveformCompressor(
                n_basis=n_basis,
                svd_solver=compressor_svd_solver,
                representation=compressor_representation,
                greedy_extra_basis=compressor_greedy_extra_basis,
            )

            total_samples = n_total
            requested_fit_samples = (
                total_samples if compressor_fit_n_samples is None else int(compressor_fit_n_samples)
            )
            n_fit = min(total_samples, requested_fit_samples)
            fit_strategy = compressor_fit_strategy.lower()

            if fit_strategy == "all" or n_fit >= total_samples:
                fit_strategy = "all"
                fit_indices = np.arange(total_samples, dtype=np.int64)
            elif fit_strategy == "first_n":
                fit_indices = np.arange(n_fit, dtype=np.int64)
            elif fit_strategy == "random_subset":
                rng = np.random.default_rng(compressor_fit_seed)
                fit_indices = np.sort(rng.choice(total_samples, size=n_fit, replace=False))
            elif fit_strategy == "hard_corner_mixed":
                rng = np.random.default_rng(compressor_fit_seed)
                chi1_mag = np.linalg.norm(self.params[:, 1:4], axis=1)
                chi2_mag = np.linalg.norm(self.params[:, 4:7], axis=1)
                hard_mask = (
                    (self.params[:, 0] >= 3.3)
                    & (chi1_mag >= 0.6)
                    & (chi2_mag >= 0.6)
                )
                hard_indices = np.flatnonzero(hard_mask)
                if len(hard_indices) >= n_fit:
                    fit_indices = rng.choice(hard_indices, size=n_fit, replace=False)
                else:
                    all_indices = np.arange(total_samples, dtype=np.int64)
                    easy_indices = np.setdiff1d(all_indices, hard_indices, assume_unique=True)
                    n_easy = min(n_fit - len(hard_indices), len(easy_indices))
                    extra_indices = rng.choice(easy_indices, size=n_easy, replace=False)
                    fit_indices = np.concatenate([hard_indices, extra_indices])
                fit_indices = np.sort(fit_indices.astype(np.int64))
            else:
                raise ValueError(
                    "compressor_fit_strategy must be one of: "
                    "'all', 'first_n', 'random_subset', 'hard_corner_mixed'"
                )

            self.compressor_fit_indices = fit_indices
            print(
                f"  Fitting compressor on {len(fit_indices)} / {total_samples} samples "
                f"({fit_strategy}, solver={compressor_svd_solver})..."
            )
            # Load only the fit subset — avoids loading all waveforms into RAM.
            with h5py.File(waveform_h5_path, "r") as f:
                hp_real_fit = f["hp_real"][fit_indices]
                hp_imag_fit = f["hp_imag"][fit_indices]
            self.compressor_fit_info = self.compressor.fit(
                hp_real_fit,
                hp_imag_fit,
                freqs=self.freqs,
            )
            del hp_real_fit, hp_imag_fit
            self.compressor_fit_info.update(
                {
                    "fit_strategy": fit_strategy,
                    "requested_fit_samples": requested_fit_samples,
                    "fit_seed": compressor_fit_seed,
                }
            )
        else:
            raise ValueError(
                "Provide one of: compressor object, compressor_path, or fit_compressor=True"
            )

        print("  Encoding waveforms to SVD coefficients (streaming)...")
        self.coeffs_real, self.coeffs_imag = self._encode_from_h5(
            waveform_h5_path, n_total
        )
        self.coeffs = np.concatenate([self.coeffs_real, self.coeffs_imag], axis=1)

        self.coeffs_mean = self.coeffs.mean(axis=0)
        self.coeffs_std = self.coeffs.std(axis=0) + 1e-10
        self.coeffs_standardized = (self.coeffs - self.coeffs_mean) / self.coeffs_std

        self.param_min = self.params.min(axis=0)
        self.param_max = self.params.max(axis=0)
        self.params_normalized = (
            2.0 * (self.params - self.param_min)
            / (self.param_max - self.param_min + 1e-10)
            - 1.0
        )

        self.params_tensor = torch.from_numpy(self.params_normalized.astype(np.float32))
        self.coeffs_tensor = torch.from_numpy(self.coeffs_standardized.astype(np.float32))

        print(
            f"Dataset ready: {len(self)} samples, "
            f"{self.coeffs.shape[1]} coeff dims, "
            f"{self.params.shape[1]} parameter dims"
        )

    def __len__(self) -> int:
        return len(self.params)

    def _encode_from_h5(
        self,
        path: str,
        n_total: int,
        batch_size: int = 2000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Stream-encode all waveforms from HDF5 without loading all into RAM.

        Opens each chunk file directly when path is a VDS or a list of paths,
        to avoid HDF5 caching entire source files (which can exhaust RAM).
        Reads batch_size rows at a time.
        """
        # If path has an adjacent .chunks metadata file listing source files,
        # open each source file directly to avoid VDS RAM caching.
        import os
        chunks_meta = os.path.join(os.path.dirname(path), "chunks", ".chunk_list.txt")
        source_files = self._detect_vds_sources(path)

        if source_files:
            return self._encode_from_multiple_h5(source_files, batch_size)

        # Fallback: read from single h5 with minimal HDF5 cache (64 MB).
        coeffs_real_list = []
        coeffs_imag_list = []
        fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_cache(0, 521, 64 * 1024 * 1024, 0.75)  # 64 MB chunk cache
        fid = h5py.h5f.open(path.encode(), h5py.h5f.ACC_RDONLY, fapl=fapl)
        with h5py.File(fid) as f:
            for start in range(0, n_total, batch_size):
                stop = min(n_total, start + batch_size)
                hp_r = f["hp_real"][start:stop]
                hp_i = f["hp_imag"][start:stop]
                cr, ci = self.compressor.encode(hp_r, hp_i)
                if not np.isfinite(cr).all() or not np.isfinite(ci).all():
                    raise ValueError(
                        f"Non-finite compressed coefficients in samples [{start}, {stop})"
                    )
                coeffs_real_list.append(cr.astype(np.float32, copy=False))
                coeffs_imag_list.append(ci.astype(np.float32, copy=False))
        return np.concatenate(coeffs_real_list, axis=0), np.concatenate(coeffs_imag_list, axis=0)

    def _detect_vds_sources(self, path: str):
        """Return list of (filepath, n_samples) if path is an HDF5 VDS, else empty list."""
        try:
            sources = []
            with h5py.File(path, "r") as f:
                ds = f["hp_real"]
                if ds.is_virtual:
                    vds_sources = ds.virtual_sources()
                    for src in vds_sources:
                        # src has .file_name and .dset_name and shape info
                        import os
                        src_path = src.file_name
                        if not os.path.isabs(src_path):
                            src_path = os.path.join(os.path.dirname(path), src_path)
                        with h5py.File(src_path, "r") as sf:
                            n = sf["hp_real"].shape[0]
                        sources.append((src_path, n))
            return sources
        except Exception:
            return []

    def _encode_from_multiple_h5(
        self,
        source_files,
        batch_size: int = 2000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode waveforms from multiple h5 files, one at a time, small batches."""
        coeffs_real_list = []
        coeffs_imag_list = []
        for src_path, n_src in source_files:
            print(f"  Encoding {src_path} ({n_src} samples)...")
            fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
            fapl.set_cache(0, 521, 32 * 1024 * 1024, 0.75)  # 32 MB cache per file
            fid = h5py.h5f.open(src_path.encode(), h5py.h5f.ACC_RDONLY, fapl=fapl)
            with h5py.File(fid) as f:
                for start in range(0, n_src, batch_size):
                    stop = min(n_src, start + batch_size)
                    hp_r = f["hp_real"][start:stop]
                    hp_i = f["hp_imag"][start:stop]
                    cr, ci = self.compressor.encode(hp_r, hp_i)
                    if not np.isfinite(cr).all() or not np.isfinite(ci).all():
                        raise ValueError(
                            f"Non-finite compressed coefficients in {src_path} [{start},{stop})"
                        )
                    coeffs_real_list.append(cr.astype(np.float32, copy=False))
                    coeffs_imag_list.append(ci.astype(np.float32, copy=False))
        return np.concatenate(coeffs_real_list, axis=0), np.concatenate(coeffs_imag_list, axis=0)

    def _encode_waveforms_in_chunks(
        self,
        real: np.ndarray,
        imag: np.ndarray,
        chunk_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Encode pre-loaded waveforms in bounded chunks (kept for callers that pass arrays)."""
        coeffs_real = []
        coeffs_imag = []
        total = len(real)
        for start in range(0, total, chunk_size):
            stop = min(total, start + chunk_size)
            chunk_real, chunk_imag = self.compressor.encode(real[start:stop], imag[start:stop])
            if not np.isfinite(chunk_real).all() or not np.isfinite(chunk_imag).all():
                raise ValueError(
                    "Non-finite compressed coefficients while encoding "
                    f"samples [{start}, {stop})"
                )
            coeffs_real.append(chunk_real.astype(np.float32, copy=False))
            coeffs_imag.append(chunk_imag.astype(np.float32, copy=False))

        return np.concatenate(coeffs_real, axis=0), np.concatenate(coeffs_imag, axis=0)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.params_tensor[idx], self.coeffs_tensor[idx]

    def apply_normalization_stats(
        self,
        coeffs_mean: np.ndarray,
        coeffs_std: np.ndarray,
        param_min: np.ndarray,
        param_max: np.ndarray,
    ):
        """Apply externally computed normalization statistics to this dataset."""
        self.coeffs_mean = np.asarray(coeffs_mean, dtype=np.float32)
        self.coeffs_std = np.asarray(coeffs_std, dtype=np.float32)
        self.param_min = np.asarray(param_min, dtype=np.float32)
        self.param_max = np.asarray(param_max, dtype=np.float32)

        self.coeffs_standardized = (self.coeffs - self.coeffs_mean) / self.coeffs_std
        self.params_normalized = (
            2.0 * (self.params - self.param_min)
            / (self.param_max - self.param_min + 1e-10)
            - 1.0
        )

        self.params_tensor = torch.from_numpy(self.params_normalized.astype(np.float32))
        self.coeffs_tensor = torch.from_numpy(self.coeffs_standardized.astype(np.float32))

    def unstandardize_coeffs(self, coeffs_std: np.ndarray) -> np.ndarray:
        """Convert standardized coefficients back to raw SVD coefficients."""
        return coeffs_std * self.coeffs_std + self.coeffs_mean

    def unnormalize_params(self, params_norm: np.ndarray) -> np.ndarray:
        """Convert normalized parameters back to physical units."""
        return (params_norm + 1.0) / 2.0 * (self.param_max - self.param_min) + self.param_min

    def decode_raw_coeffs_to_waveform(self, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode raw, unstandardized coefficients to waveform real/imag arrays."""
        n_basis = self.compressor.n_basis
        coeffs_r = coeffs[:, :n_basis]
        coeffs_i = coeffs[:, n_basis:]
        return self.compressor.decode(coeffs_r, coeffs_i, self.freqs)

    def decode_to_waveform(self, coeffs_std: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decode standardized coefficients to waveform real/imag arrays."""
        coeffs = self.unstandardize_coeffs(coeffs_std)
        return self.decode_raw_coeffs_to_waveform(coeffs)
