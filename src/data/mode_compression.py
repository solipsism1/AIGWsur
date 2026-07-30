"""Per-mode, per-channel SVD compression for the conditioned TD modes.

Each ``(l, m)`` mode gets its own SVD basis with its own ``n_basis``. Oscillatory
modes (``m != 0``) are compressed in amplitude/phase (smooth envelope + monotonic
phase); the nonoscillatory memory modes (``m == 0``) are compressed in real/imag,
where amplitude/phase is ill-defined.

Per the v1 invariants there is no global output squashing: every channel is
standardized (per-frequency mean/std) and modes are concatenated into one coefficient
vector. The model head is split per mode so each mode's dynamic range is independent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import os
import re

import h5py
import numpy as np

from src.data.mode_waveforms import modes_to_amp_phase, amp_phase_to_modes


def resolve_data_path(path: str) -> str:
    """Translate Windows drive paths to /mnt/<drive> when running under Linux/WSL."""
    if os.name != "nt":
        m = re.match(r"^([A-Za-z]):[\\/](.*)$", str(path))
        if m:
            return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    return str(path)


def mode_representation(key: Tuple[int, int]) -> str:
    """amp/phase for oscillatory modes, real/imag for the m=0 memory modes."""
    return "reim" if key[1] == 0 else "ampphase"


def _channels(mode_complex: np.ndarray, rep: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the two real channels (c0, c1) for a stack of one mode, ``(n, N)``."""
    if rep == "reim":
        return mode_complex.real.copy(), mode_complex.imag.copy()
    amp, phase = modes_to_amp_phase(mode_complex)
    return amp, phase


def _from_channels(c0: np.ndarray, c1: np.ndarray, rep: str) -> np.ndarray:
    if rep == "reim":
        return c0 + 1j * c1
    return amp_phase_to_modes(c0, c1)


class _ChannelSVD:
    """Standardized SVD for one real channel."""

    def __init__(self, n_basis: int):
        self.n_basis = int(n_basis)
        self.mean = None
        self.std = None
        self.basis = None

    def fit(self, x: np.ndarray):
        self.mean = x.mean(0)
        self.std = x.std(0) + 1e-30
        _, _, vt = np.linalg.svd((x - self.mean) / self.std, full_matrices=False)
        self.basis = vt[: self.n_basis].astype(np.float32)
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (((x - self.mean) / self.std) @ self.basis.T).astype(np.float32)

    def decode(self, c: np.ndarray) -> np.ndarray:
        return (c @ self.basis) * self.std + self.mean


class PerModeCompressor:
    """Holds one ``_ChannelSVD`` pair per mode and (de)codes concatenated coeffs."""

    def __init__(self, keys: List[Tuple[int, int]], n_basis: Dict[Tuple[int, int], int]):
        self.keys = [(int(l), int(m)) for l, m in keys]
        self.n_basis = {tuple(k): int(v) for k, v in n_basis.items()}
        self.svd: Dict[Tuple[int, int], Tuple[_ChannelSVD, _ChannelSVD]] = {}
        self.is_fitted = False

    @property
    def coeff_dim(self) -> int:
        return sum(2 * self.n_basis[k] for k in self.keys)

    def mode_slices(self) -> Dict[Tuple[int, int], Tuple[slice, slice]]:
        """Coefficient-vector slices ``{key: (c0_slice, c1_slice)}``."""
        out = {}
        off = 0
        for k in self.keys:
            nb = self.n_basis[k]
            out[k] = (slice(off, off + nb), slice(off + nb, off + 2 * nb))
            off += 2 * nb
        return out

    def fit(self, modes: np.ndarray):
        """``modes``: ``(n_samples, n_modes, N)`` complex, in ``self.keys`` order."""
        for mi, k in enumerate(self.keys):
            rep = mode_representation(k)
            c0, c1 = _channels(modes[:, mi, :], rep)
            self.svd[k] = (
                _ChannelSVD(self.n_basis[k]).fit(c0),
                _ChannelSVD(self.n_basis[k]).fit(c1),
            )
        self.is_fitted = True
        return self

    def encode(self, modes: np.ndarray) -> np.ndarray:
        assert self.is_fitted
        cols = []
        for mi, k in enumerate(self.keys):
            rep = mode_representation(k)
            c0, c1 = _channels(modes[:, mi, :], rep)
            s0, s1 = self.svd[k]
            cols.append(s0.encode(c0))
            cols.append(s1.encode(c1))
        return np.concatenate(cols, axis=1)

    def decode(self, coeffs: np.ndarray) -> np.ndarray:
        assert self.is_fitted
        n = coeffs.shape[0]
        n_freq = self.svd[self.keys[0]][0].mean.shape[0]
        out = np.zeros((n, len(self.keys), n_freq), dtype=np.complex128)
        slices = self.mode_slices()
        for mi, k in enumerate(self.keys):
            rep = mode_representation(k)
            s0, s1 = self.svd[k]
            sl0, sl1 = slices[k]
            c0 = s0.decode(coeffs[:, sl0])
            c1 = s1.decode(coeffs[:, sl1])
            out[:, mi, :] = _from_channels(c0, c1, rep)
        return out

    def save(self, path: str):
        with h5py.File(path, "w") as f:
            f.attrs["mode_keys"] = ",".join(f"{l}:{m}" for l, m in self.keys)
            for k in self.keys:
                g = f.create_group(f"{k[0]}_{k[1]}")
                g.attrs["n_basis"] = self.n_basis[k]
                g.attrs["representation"] = mode_representation(k)
                for ci, sv in enumerate(self.svd[k]):
                    sg = g.create_group(f"ch{ci}")
                    sg.create_dataset("mean", data=sv.mean)
                    sg.create_dataset("std", data=sv.std)
                    sg.create_dataset("basis", data=sv.basis)

    @classmethod
    def load(cls, path: str) -> "PerModeCompressor":
        path = resolve_data_path(path)
        with h5py.File(path, "r") as f:
            keys = [tuple(int(x) for x in p.split(":")) for p in f.attrs["mode_keys"].split(",")]
            nb = {k: int(f[f"{k[0]}_{k[1]}"].attrs["n_basis"]) for k in keys}
            comp = cls(keys, nb)
            for k in keys:
                g = f[f"{k[0]}_{k[1]}"]
                pair = []
                for ci in range(2):
                    sg = g[f"ch{ci}"]
                    sv = _ChannelSVD(nb[k])
                    sv.mean = sg["mean"][:]; sv.std = sg["std"][:]; sv.basis = sg["basis"][:]
                    pair.append(sv)
                comp.svd[k] = tuple(pair)
            comp.is_fitted = True
        return comp
