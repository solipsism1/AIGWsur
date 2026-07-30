"""
Compress frequency-domain waveforms using an SVD basis.

Reduces thousands of frequency bins to a compact set of coefficients while
retaining nearly all of the waveform variance in the real and imaginary parts
of h+(f).
"""

from __future__ import annotations

from typing import Optional, Tuple

import h5py
import numpy as np
from scipy.sparse.linalg import svds


def _decode_attr(value, default: str) -> str:
    """Decode an HDF5 attribute that may be stored as bytes."""
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _aligo_zero_det_high_power(freqs: np.ndarray, f_lower: float) -> np.ndarray:
    """Return a lightweight fallback aLIGO design PSD approximation.

    When PyCBC/LALSimulation is installed, use that implementation instead.
    This fallback is kept only for environments that cannot import PyCBC.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = np.full_like(freqs, np.inf, dtype=np.float64)
    valid = freqs >= f_lower
    f = freqs[valid]
    x = f / 215.0
    s0 = 1.0e-49
    psd[valid] = s0 * (
        (284.9 * x**-4.8)
        + (17.11 * x**-1.7)
        + x**3 * ((2.36 * x**2 - 3.84 * x + 1.25) / (x**2 + x + 1))
    )
    return psd


class WaveformCompressor:
    """SVD-based compression for gravitational waveforms."""

    def __init__(
        self,
        n_basis: int = 100,
        svd_solver: str = "auto",
        representation: str = "amp_phase",
        greedy_extra_basis: int = 0,
    ):
        self.n_basis = n_basis
        self.svd_solver = svd_solver
        self.representation = representation.lower()
        self.greedy_extra_basis = max(0, int(greedy_extra_basis))
        self.basis_real = None
        self.basis_imag = None
        self.mean_real = None
        self.mean_imag = None
        self.std_real = None
        self.std_imag = None
        self.singular_values_real = None
        self.singular_values_imag = None
        self.frequency_weights = None
        self.joint_scale = 1.0
        self.psd_source = "none"
        self.is_fitted = False
        self.fit_info = None

    def _append_greedy_basis(
        self,
        centered_scaled: np.ndarray,
        basis: np.ndarray,
        n_extra: int,
        chunk_size: int = 512,
    ) -> tuple[np.ndarray, list[int]]:
        """Append orthonormal residual directions for high-error fit examples."""
        if n_extra <= 0:
            return basis, []

        basis = np.asarray(basis, dtype=np.float32)
        selected: list[int] = []
        n_samples = centered_scaled.shape[0]

        for _ in range(n_extra):
            max_err = -np.inf
            max_idx = 0
            for start in range(0, n_samples, chunk_size):
                stop = min(start + chunk_size, n_samples)
                chunk = centered_scaled[start:stop]
                coeffs = chunk @ basis.T
                residual = chunk - coeffs @ basis
                err = np.einsum("ij,ij->i", residual, residual)
                local_idx = int(np.argmax(err))
                if float(err[local_idx]) > max_err:
                    max_err = float(err[local_idx])
                    max_idx = start + local_idx

            vec = centered_scaled[max_idx].astype(np.float32, copy=True)
            for _ in range(2):
                vec -= (vec @ basis.T) @ basis
            norm = float(np.linalg.norm(vec))
            if norm < 1e-8:
                break
            basis = np.vstack([basis, (vec / norm)[None].astype(np.float32)])
            selected.append(max_idx)

        if selected:
            print(f"  Added {len(selected)} greedy residual basis vectors")
        return basis, selected

    def _compute_component_svd(
        self,
        centered: np.ndarray,
        n_basis: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray, float, str]:
        """Compute a truncated or full SVD for one waveform component."""
        n_basis = self.n_basis if n_basis is None else int(n_basis)
        n_samples, n_freq = centered.shape
        max_rank = min(n_samples, n_freq)
        if n_basis > max_rank:
            raise ValueError(
                f"n_basis={n_basis} exceeds the available matrix rank {max_rank}"
            )

        solver = self.svd_solver.lower()
        if solver not in {"auto", "full", "truncated", "randomized"}:
            raise ValueError(
                "svd_solver must be one of: 'auto', 'full', 'truncated', or 'randomized'"
            )

        use_truncated = False
        use_randomized = False
        if solver == "randomized":
            use_randomized = n_basis < max_rank
        elif solver == "truncated":
            use_truncated = n_basis < max_rank
        elif solver == "auto":
            use_truncated = n_basis < max_rank and max_rank > 4 * n_basis
            use_randomized = use_truncated and n_basis >= 1000

        total_variance = float(np.square(centered, dtype=np.float64).sum())

        if use_randomized:
            vt, singular_values = self._compute_randomized_svd(centered, n_basis)
            explained_variance = float((singular_values ** 2).sum() / max(total_variance, 1e-30))
            return vt, singular_values, explained_variance, "randomized"

        if use_truncated:
            try:
                _, singular_values, vt = svds(
                    centered,
                    k=n_basis,
                    return_singular_vectors="vh",
                )
            except Exception as exc:
                print(f"  ARPACK SVD failed ({exc}); falling back to randomized SVD")
                vt, singular_values = self._compute_randomized_svd(centered, n_basis)
                explained_variance = float((singular_values ** 2).sum() / max(total_variance, 1e-30))
                return vt, singular_values, explained_variance, "randomized"
            order = np.argsort(singular_values)[::-1]
            singular_values = singular_values[order]
            vt = vt[order]
            basis = vt[:n_basis]
            explained_variance = float((singular_values ** 2).sum() / max(total_variance, 1e-30))
            solver_used = "truncated"
        else:
            _, singular_values_full, vt_full = np.linalg.svd(centered, full_matrices=False)
            singular_values = singular_values_full[:n_basis]
            basis = vt_full[:n_basis]
            explained_variance = float(
                (singular_values ** 2).sum()
                / max(float((singular_values_full ** 2).sum()), 1e-30)
            )
            solver_used = "full"

        return basis, singular_values, explained_variance, solver_used

    def _compute_randomized_svd(
        self,
        centered: np.ndarray,
        n_basis: int,
        n_oversamples: int = 32,
        n_iter: int = 2,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Approximate leading right singular vectors without ARPACK."""
        data = np.asarray(centered, dtype=np.float32, order="C")
        n_samples, n_features = data.shape
        rank = min(n_samples, n_features)
        sample_dim = min(rank, n_basis + n_oversamples)
        rng = np.random.default_rng(12345)

        omega = rng.standard_normal((n_features, sample_dim)).astype(np.float32)
        q, _ = np.linalg.qr(data @ omega, mode="reduced")
        q = q.astype(np.float32, copy=False)

        for _ in range(max(0, int(n_iter))):
            z = data.T @ q
            q, _ = np.linalg.qr(data @ z, mode="reduced")
            q = q.astype(np.float32, copy=False)

        small = q.T @ data
        _, singular_values, vt = np.linalg.svd(small, full_matrices=False)
        return vt[:n_basis].astype(np.float32, copy=False), singular_values[:n_basis].astype(np.float32)

    def _frequency_weights(self, freqs: np.ndarray) -> np.ndarray:
        """Build PSD-whitening weights for the stored frequency grid."""
        freqs = np.asarray(freqs, dtype=np.float64)
        f_lower = float(freqs[freqs > 0][0])
        psd_source = "fallback_analytic"
        try:
            from pycbc.psd import aLIGOZeroDetHighPower

            delta_f = float(freqs[1] - freqs[0])
            n_pad = int(round(float(freqs[0]) / delta_f)) if freqs[0] > 0 else 0
            psd_series = aLIGOZeroDetHighPower(len(freqs) + n_pad, delta_f, f_lower)
            psd = np.asarray(psd_series, dtype=np.float64)[n_pad : n_pad + len(freqs)]
            psd_source = "pycbc_aLIGOZeroDetHighPower"
        except Exception:
            psd = _aligo_zero_det_high_power(freqs, f_lower=f_lower)
        weights = np.zeros_like(freqs, dtype=np.float64)
        valid = np.isfinite(psd) & (psd > 0)
        weights[valid] = 1.0 / np.sqrt(psd[valid])
        max_weight = float(weights.max())
        if max_weight > 0:
            weights /= max_weight
        self.psd_source = psd_source
        return weights.astype(np.float32)

    def _fit_joint_complex(
        self,
        real: np.ndarray,
        imag: np.ndarray,
        freqs: np.ndarray,
        whiten: bool,
    ) -> dict:
        """Fit a joint real/imag SVD basis."""
        n_samples, n_freq = real.shape
        total_basis = 2 * self.n_basis
        svd_basis = max(1, total_basis - self.greedy_extra_basis)
        if whiten:
            weights = self._frequency_weights(freqs)
            label = "Joint whitened complex"
        else:
            weights = np.ones(n_freq, dtype=np.float32)
            label = "Joint complex"
        joint = np.concatenate([real * weights, imag * weights], axis=1)

        self.mean_real = joint.mean(axis=0)
        self.std_real = np.ones_like(self.mean_real, dtype=np.float32)
        centered = joint - self.mean_real
        self.joint_scale = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))) + 1e-30)
        centered_scaled = centered / self.joint_scale

        self.basis_real, singular_values, explained, solver_used = self._compute_component_svd(
            centered_scaled,
            n_basis=svd_basis,
        )
        self.basis_real, greedy_indices = self._append_greedy_basis(
            centered_scaled,
            self.basis_real,
            total_basis - len(self.basis_real),
        )
        self.basis_imag = np.empty((0, n_freq), dtype=np.float32)
        self.mean_imag = np.empty((0,), dtype=np.float32)
        self.std_imag = np.empty((0,), dtype=np.float32)
        self.singular_values_real = singular_values
        self.singular_values_imag = np.empty((0,), dtype=np.float32)
        self.frequency_weights = weights
        self.is_fitted = True

        info = {
            "representation": self.representation,
            "explained_variance_joint": float(explained),
            "n_basis": self.n_basis,
            "n_coeffs": total_basis,
            "n_svd_basis": svd_basis,
            "n_greedy_basis": len(greedy_indices),
            "n_freq": n_freq,
            "n_samples": n_samples,
            "joint_scale": self.joint_scale,
            "psd_source": self.psd_source,
            "svd_solver": self.svd_solver,
            "svd_solver_joint": solver_used,
        }
        self.fit_info = info

        print(
            f"{label} SVD compression: "
            f"{n_freq} freq bins -> {total_basis} coefficients"
        )
        print(f"  Joint explained variance: {explained:.6f}")
        print(f"  SVD solver: {solver_used}")
        return info

    def _fit_normalized_joint_complex(
        self,
        real: np.ndarray,
        imag: np.ndarray,
        freqs: np.ndarray,
        whiten: bool,
    ) -> dict:
        """Fit SVD to normalized waveform shapes plus one log-norm coefficient."""
        n_samples, n_freq = real.shape
        shape_basis = 2 * self.n_basis - 1
        svd_shape_basis = max(1, shape_basis - self.greedy_extra_basis)
        if shape_basis < 1:
            raise ValueError("n_basis must be at least 1 for normalized joint compression")

        if whiten:
            weights = self._frequency_weights(freqs)
            label = "Normalized joint whitened complex"
        else:
            weights = np.ones(n_freq, dtype=np.float32)
            label = "Normalized joint complex"

        joint = np.concatenate([real * weights, imag * weights], axis=1)
        norms = np.sqrt(np.square(joint, dtype=np.float64).sum(axis=1)).astype(np.float32)
        norms = np.maximum(norms, 1e-30)
        shapes = joint / norms[:, None]

        self.mean_real = shapes.mean(axis=0)
        self.std_real = np.ones_like(self.mean_real, dtype=np.float32)
        centered = shapes - self.mean_real
        self.joint_scale = float(np.sqrt(np.mean(np.square(centered, dtype=np.float64))) + 1e-30)
        centered_scaled = centered / self.joint_scale

        self.basis_real, singular_values, explained, solver_used = self._compute_component_svd(
            centered_scaled,
            n_basis=svd_shape_basis,
        )
        self.basis_real, greedy_indices = self._append_greedy_basis(
            centered_scaled,
            self.basis_real,
            shape_basis - len(self.basis_real),
        )
        self.basis_imag = np.empty((0, n_freq), dtype=np.float32)
        self.mean_imag = np.empty((0,), dtype=np.float32)
        self.std_imag = np.empty((0,), dtype=np.float32)
        self.singular_values_real = singular_values
        self.singular_values_imag = np.empty((0,), dtype=np.float32)
        self.frequency_weights = weights
        self.is_fitted = True

        info = {
            "representation": self.representation,
            "explained_variance_joint_shape": float(explained),
            "n_basis": self.n_basis,
            "n_coeffs": 2 * self.n_basis,
            "n_shape_basis": shape_basis,
            "n_svd_basis": svd_shape_basis,
            "n_greedy_basis": len(greedy_indices),
            "n_freq": n_freq,
            "n_samples": n_samples,
            "joint_scale": self.joint_scale,
            "psd_source": self.psd_source,
            "svd_solver": self.svd_solver,
            "svd_solver_joint": solver_used,
        }
        self.fit_info = info

        print(
            f"{label} SVD compression: "
            f"{n_freq} freq bins -> 1 log-norm + {shape_basis} shape coefficients"
        )
        print(f"  Joint shape explained variance: {explained:.6f}")
        print(f"  SVD solver: {solver_used}")
        return info

    def fit(self, real: np.ndarray, imag: np.ndarray, freqs: Optional[np.ndarray] = None) -> dict:
        """Fit SVD bases from arrays of training waveforms."""
        n_samples, n_freq = real.shape

        if self.representation in {
            "normalized_joint_whitened_complex",
            "normalized_joint_complex",
        }:
            if freqs is None:
                raise ValueError(f"freqs must be provided for {self.representation} compression")
            return self._fit_normalized_joint_complex(
                real,
                imag,
                freqs,
                whiten=self.representation == "normalized_joint_whitened_complex",
            )

        if self.representation in {"joint_whitened_complex", "joint_complex"}:
            if freqs is None:
                raise ValueError(f"freqs must be provided for {self.representation} compression")
            return self._fit_joint_complex(
                real,
                imag,
                freqs,
                whiten=self.representation == "joint_whitened_complex",
            )

        if self.representation not in {"amp_phase", "polar"}:
            raise ValueError(
                "representation must be one of: 'amp_phase', 'polar', "
                "'joint_whitened_complex', 'joint_complex', "
                "'normalized_joint_whitened_complex', or 'normalized_joint_complex'"
            )

        amp = np.sqrt(real**2 + imag**2)
        phase = np.unwrap(np.arctan2(imag, real), axis=1)

        self.mean_real = amp.mean(axis=0)
        self.mean_imag = phase.mean(axis=0)
        self.std_real = amp.std(axis=0) + 1e-10
        self.std_imag = phase.std(axis=0) + 1e-10

        centered_amp = (amp - self.mean_real) / self.std_real
        centered_phase = (phase - self.mean_imag) / self.std_imag

        self.basis_real, singular_amp, var_amp, solver_amp = self._compute_component_svd(centered_amp)
        self.basis_imag, singular_phase, var_phase, solver_phase = self._compute_component_svd(centered_phase)

        self.singular_values_real = singular_amp
        self.singular_values_imag = singular_phase

        self.is_fitted = True

        info = {
            "explained_variance_amp": float(var_amp),
            "explained_variance_phase": float(var_phase),
            "explained_variance_real": float(var_amp),
            "explained_variance_imag": float(var_phase),
            "n_basis": self.n_basis,
            "n_freq": n_freq,
            "n_samples": n_samples,
            "svd_solver": self.svd_solver,
            "svd_solver_amp": solver_amp,
            "svd_solver_phase": solver_phase,
            "svd_solver_real": solver_amp,
            "svd_solver_imag": solver_phase,
        }
        self.fit_info = info

        print(f"Polar SVD compression: {n_freq} freq bins -> {self.n_basis} coefficients")
        print(f"  Amp explained variance: {var_amp:.6f}")
        print(f"  Phase explained variance: {var_phase:.6f}")
        print(f"  SVD solver (amp / phase): {solver_amp} / {solver_phase}")

        return info

    def encode(self, real: np.ndarray, imag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project waveforms onto the fitted SVD basis."""
        assert self.is_fitted, "Must call fit() before encode()"

        if self.representation in {
            "normalized_joint_whitened_complex",
            "normalized_joint_complex",
        }:
            weights = self.frequency_weights.astype(np.float64, copy=False)
            joint = np.concatenate(
                [
                    real.astype(np.float64, copy=False) * weights,
                    imag.astype(np.float64, copy=False) * weights,
                ],
                axis=1,
            )
            norms = np.sqrt(np.square(joint, dtype=np.float64).sum(axis=1))
            norms = np.maximum(norms, 1e-30)
            shapes = joint / norms[:, None]
            mean = self.mean_real.astype(np.float64, copy=False)
            basis = self.basis_real.astype(np.float64, copy=False)
            scale = max(float(self.joint_scale), 1e-300)
            centered_shapes = shapes
            centered_shapes -= mean
            centered_shapes *= 1.0 / scale
            shape_coeffs = centered_shapes @ basis.T
            coeffs = np.concatenate([np.log(norms)[:, None], shape_coeffs], axis=1)
            return (
                coeffs[:, : self.n_basis].astype(np.float32),
                coeffs[:, self.n_basis :].astype(np.float32),
            )

        if self.representation in {"joint_whitened_complex", "joint_complex"}:
            weights = self.frequency_weights.astype(np.float64, copy=False)
            joint = np.concatenate(
                [
                    real.astype(np.float64, copy=False) * weights,
                    imag.astype(np.float64, copy=False) * weights,
                ],
                axis=1,
            )
            mean = self.mean_real.astype(np.float64, copy=False)
            basis = self.basis_real.astype(np.float64, copy=False)
            scale = max(float(self.joint_scale), 1e-300)
            joint -= mean
            joint *= 1.0 / scale
            coeffs = joint @ basis.T
            return (
                coeffs[:, : self.n_basis].astype(np.float32),
                coeffs[:, self.n_basis :].astype(np.float32),
            )
        
        amp = np.sqrt(real**2 + imag**2)
        phase = np.unwrap(np.arctan2(imag, real), axis=1)

        centered_amp = (amp - self.mean_real) / self.std_real
        centered_phase = (phase - self.mean_imag) / self.std_imag

        coeffs_amp = centered_amp @ self.basis_real.T
        coeffs_phase = centered_phase @ self.basis_imag.T

        return coeffs_amp.astype(np.float32), coeffs_phase.astype(np.float32)

    def decode(
        self,
        coeffs_amp: np.ndarray,
        coeffs_phase: np.ndarray,
        freqs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct waveforms from compressed coefficients."""
        assert self.is_fitted, "Compressor must be fitted before decode()"

        if self.representation in {
            "normalized_joint_whitened_complex",
            "normalized_joint_complex",
        }:
            coeffs = np.concatenate([coeffs_amp, coeffs_phase], axis=1).astype(np.float64, copy=False)
            norms = np.exp(coeffs[:, :1])
            shape_coeffs = coeffs[:, 1:]
            basis = self.basis_real.astype(np.float64, copy=False)
            mean = self.mean_real.astype(np.float64, copy=False)
            scale = max(float(self.joint_scale), 1e-300)
            joint_shape = (shape_coeffs @ basis) * scale + mean
            joint = joint_shape * norms
            n_freq = len(self.frequency_weights)
            weights = self.frequency_weights.astype(np.float64, copy=False)
            inverse_weights = np.zeros_like(weights, dtype=np.float64)
            valid = weights > 0
            inverse_weights[valid] = 1.0 / weights[valid]
            real = joint[:, :n_freq] * inverse_weights
            imag = joint[:, n_freq:] * inverse_weights
            return real.astype(np.float32), imag.astype(np.float32)

        if self.representation in {"joint_whitened_complex", "joint_complex"}:
            coeffs = np.concatenate([coeffs_amp, coeffs_phase], axis=1).astype(np.float64, copy=False)
            basis = self.basis_real.astype(np.float64, copy=False)
            mean = self.mean_real.astype(np.float64, copy=False)
            scale = max(float(self.joint_scale), 1e-300)
            joint = (coeffs @ basis) * scale + mean
            n_freq = len(self.frequency_weights)
            weights = self.frequency_weights.astype(np.float64, copy=False)
            inverse_weights = np.zeros_like(weights, dtype=np.float64)
            valid = weights > 0
            inverse_weights[valid] = 1.0 / weights[valid]
            real = joint[:, :n_freq] * inverse_weights
            imag = joint[:, n_freq:] * inverse_weights
            return real.astype(np.float32), imag.astype(np.float32)

        centered_amp = coeffs_amp @ self.basis_real
        centered_phase = coeffs_phase @ self.basis_imag

        amp = centered_amp * self.std_real + self.mean_real
        phase = centered_phase * self.std_imag + self.mean_imag
        
        real = amp * np.cos(phase)
        imag = amp * np.sin(phase)

        return real, imag

    def save(self, path: str):
        """Persist compressor state to an HDF5 file."""
        with h5py.File(path, "w") as f:
            f.create_dataset("basis_real", data=self.basis_real)
            f.create_dataset("basis_imag", data=self.basis_imag)
            f.create_dataset("mean_real", data=self.mean_real)
            f.create_dataset("mean_imag", data=self.mean_imag)
            f.create_dataset("std_real", data=self.std_real)
            f.create_dataset("std_imag", data=self.std_imag)
            f.create_dataset("singular_values_real", data=self.singular_values_real)
            f.create_dataset("singular_values_imag", data=self.singular_values_imag)
            if self.frequency_weights is not None:
                f.create_dataset("frequency_weights", data=self.frequency_weights)
            f.attrs["n_basis"] = self.n_basis
            f.attrs["svd_solver"] = self.svd_solver
            f.attrs["representation"] = self.representation
            f.attrs["joint_scale"] = self.joint_scale
            f.attrs["psd_source"] = self.psd_source
            f.attrs["greedy_extra_basis"] = self.greedy_extra_basis

    @classmethod
    def load(cls, path: str) -> "WaveformCompressor":
        """Load a previously saved compressor from HDF5."""
        with h5py.File(path, "r") as f:
            comp = cls(
                n_basis=int(f.attrs["n_basis"]),
                svd_solver=_decode_attr(f.attrs.get("svd_solver"), "auto"),
                representation=_decode_attr(f.attrs.get("representation"), "amp_phase"),
                greedy_extra_basis=int(f.attrs.get("greedy_extra_basis", 0)),
            )
            comp.basis_real = f["basis_real"][:]
            comp.basis_imag = f["basis_imag"][:]
            comp.mean_real = f["mean_real"][:]
            comp.mean_imag = f["mean_imag"][:]
            comp.std_real = f["std_real"][:]
            comp.std_imag = f["std_imag"][:]
            comp.singular_values_real = f["singular_values_real"][:]
            comp.singular_values_imag = f["singular_values_imag"][:]
            if "frequency_weights" in f:
                comp.frequency_weights = f["frequency_weights"][:]
            comp.joint_scale = float(f.attrs.get("joint_scale", 1.0))
            comp.psd_source = _decode_attr(f.attrs.get("psd_source"), "unknown")
            comp.is_fitted = True
        return comp


def compute_reconstruction_matches(
    compressor: WaveformCompressor,
    real_arr: np.ndarray,
    imag_arr: np.ndarray,
    freqs: np.ndarray,
    n_test: int = 1000,
) -> np.ndarray:
    """Compute PyCBC matches between original and SVD-reconstructed waveforms."""
    try:
        from pycbc.filter import match
        from pycbc.psd import aLIGOZeroDetHighPower
        from pycbc.types import FrequencySeries
    except ImportError as exc:
        raise RuntimeError(
            "PyCBC is required for reconstruction-match checks."
        ) from exc

    coeffs_real, coeffs_imag = compressor.encode(real_arr[:n_test], imag_arr[:n_test])
    recon_real, recon_imag = compressor.decode(coeffs_real, coeffs_imag, freqs)

    delta_f = float(freqs[1] - freqs[0])
    f_lower = float(freqs[0])
    n_pad = int(round(f_lower / delta_f)) if f_lower > 0 else 0
    psd = aLIGOZeroDetHighPower(len(freqs) + n_pad, delta_f, f_lower)

    matches = []
    for idx in range(n_test):
        h_orig = (real_arr[idx] + 1j * imag_arr[idx]).astype(np.complex128)
        h_recon = (recon_real[idx] + 1j * recon_imag[idx]).astype(np.complex128)
        if n_pad:
            h_orig = np.pad(h_orig, (n_pad, 0), mode="constant")
            h_recon = np.pad(h_recon, (n_pad, 0), mode="constant")

        h1 = FrequencySeries(h_orig, delta_f=delta_f)
        h2 = FrequencySeries(h_recon, delta_f=delta_f)

        match_value, _ = match(h1, h2, psd=psd, low_frequency_cutoff=f_lower)
        matches.append(match_value)

    matches = np.array(matches)
    print(f"SVD reconstruction matches ({n_test} samples):")
    print(f"  Mean:   {matches.mean():.6f}")
    print(f"  Median: {np.median(matches):.6f}")
    print(f"  Min:    {matches.min():.6f}")
    print(f"  >0.999: {(matches > 0.999).mean() * 100:.1f}%")
    return matches
