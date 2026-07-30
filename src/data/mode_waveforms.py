"""Generate inertial-frame NRSur7dq4 spherical-harmonic modes in the TIME domain.

NRSur7dq4 has no native FD-mode generator (and ``get_fd_waveform`` rejects it), so we
pull the time-domain modes ``h_{lm}(t)`` from ``get_td_waveform_modes`` and condition
them onto a common, peak-aligned time grid. Each mode is then represented by a smooth
amplitude envelope and a monotonic phase (the standard surrogate representation),
which compresses far better than the oscillatory real/imag parts and — unlike an FD
representation — has no single-sideband ambiguity (m>0 modes live at negative
frequency; see project notes).

Conditioning
------------
* All 21 modes of one waveform share a single time grid (PyCBC guarantees this).
* We peak-align on the ``(2, 2)`` amplitude maximum: the peak sits at index ``T_pre``
  on a common grid of length ``T_pre + T_post``.
* Inspiral length varies ~2x across parameters; shorter waveforms are zero-padded at
  the (low-frequency) start, longer ones truncated there. A Planck taper over the
  first ``taper_len`` live samples smooths the turn-on so the pad/truncation edge does
  not ring when the reconstructed strain is later FFT'd for matching.

Reconstruction (eval): ``z(t) = sum_lm A_lm e^{i phi_lm} _{-2}Y_lm(iota, phi)``,
then ``h_+ = Re z``, ``h_x = -Im z`` (azimuth maps to PyCBC as ``phi = pi/2 - coa``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.generate_waveforms import omega0_to_f_ref_hz
from src.physics.swsh import default_modes

PEAK_MODE = (2, 2)


def mode_keys(ell_max: int = 4, include_m0: bool = True) -> List[Tuple[int, int]]:
    """Ordered ``(l, m)`` keys for the surrogate, matching the SWSH ordering."""
    return default_modes(ell_max, include_m0=include_m0)


@dataclass(frozen=True)
class ModeGrid:
    """Common peak-aligned time grid for the conditioned modes."""
    t_pre: int = 4352      # samples before the (2,2) peak (covers observed max ~4207)
    t_post: int = 192      # samples after the peak (covers ringdown ~145)
    taper_len: int = 192   # Planck taper length applied at each waveform's live start
    sample_rate: float = 4096.0

    @property
    def n(self) -> int:
        return self.t_pre + self.t_post

    @property
    def peak_index(self) -> int:
        return self.t_pre


def _planck_window(n: int) -> np.ndarray:
    """Planck taper ramping 0->1 over ``n`` samples (smooth turn-on)."""
    if n <= 1:
        return np.ones(max(n, 0), dtype=np.float64)
    t = np.linspace(0.0, 1.0, n + 2)[1:-1]
    z = (1.0 / t) - (1.0 / (1.0 - t))
    return 1.0 / (1.0 + np.exp(z))


def _waveform_kwargs(params: np.ndarray, config: dict) -> dict:
    """PyCBC kwargs for one parameter vector (reuses legacy f_ref logic)."""
    data = config["data"]
    M = float(data["reference_total_mass"])
    q = float(params[0])
    kw = dict(
        approximant=data["approximant"], mass1=M * q / (1.0 + q), mass2=M / (1.0 + q),
        spin1x=float(params[1]), spin1y=float(params[2]), spin1z=float(params[3]),
        spin2x=float(params[4]), spin2y=float(params[5]), spin2z=float(params[6]),
        distance=1.0,
    )
    f_lower = float(data["f_lower"])
    if params.shape[0] > 7:
        f_ref = omega0_to_f_ref_hz(float(params[7]), M)
        kw["f_ref"] = f_ref
        kw["f_lower"] = min(f_lower, f_ref)
    else:
        kw["f_ref"] = f_lower
        kw["f_lower"] = f_lower
    return kw


def generate_td_modes(
    params: np.ndarray,
    config: dict,
    grid: ModeGrid,
    modes: Optional[List[Tuple[int, int]]] = None,
) -> Optional[np.ndarray]:
    """Return conditioned complex modes on the common grid, shape ``(n_modes, N)``.

    Returns ``None`` if PyCBC fails for this parameter point (e.g. omega_ref below the
    NRSur floor).
    """
    from pycbc.waveform import get_td_waveform_modes

    if modes is None:
        modes = mode_keys()
    kw = _waveform_kwargs(np.asarray(params), config)
    try:
        td = get_td_waveform_modes(delta_t=1.0 / grid.sample_rate, **kw)
    except Exception:
        return None

    re22, im22 = td[PEAK_MODE]
    a22 = np.sqrt(np.asarray(re22.data) ** 2 + np.asarray(im22.data) ** 2)
    ip = int(np.argmax(a22))
    src_len = len(a22)

    # Map source index k -> grid index j = k - ip + T_pre. Live grid span:
    j0 = grid.peak_index - ip            # grid index of source sample 0
    g_start = max(0, j0)                 # first live grid index
    g_stop = min(grid.n, j0 + src_len)   # one past last live grid index
    s_start = g_start - j0               # corresponding source start
    n_live = g_stop - g_start
    if n_live <= grid.taper_len:
        return None  # degenerate / too short to taper

    taper = _planck_window(grid.taper_len)
    out = np.zeros((len(modes), grid.n), dtype=np.complex128)
    for mi, (l, m) in enumerate(modes):
        re, im = td[(l, m)]
        h = (np.asarray(re.data) + 1j * np.asarray(im.data))[s_start:s_start + n_live]
        h = h.copy()
        h[:grid.taper_len] *= taper
        out[mi, g_start:g_stop] = h
    return out


def modes_to_amp_phase(modes_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Complex modes ``(n_modes, N)`` -> ``(amp, phase)`` with constant-extrapolated phase.

    Amplitude is exactly ``|h|`` (zero off-support). Phase is unwrapped on the live
    support and held constant in the zero regions so the SVD never sees the garbage
    phase of zeros (reconstruction is unaffected since amp=0 there).
    """
    amp = np.abs(modes_complex)
    phase = np.unwrap(np.angle(modes_complex), axis=1)
    for i in range(modes_complex.shape[0]):
        live = np.flatnonzero(amp[i] > 0)
        if live.size == 0:
            phase[i, :] = 0.0
            continue
        lo, hi = live[0], live[-1]
        phase[i, :lo] = phase[i, lo]
        phase[i, hi + 1:] = phase[i, hi]
    return amp, phase


def amp_phase_to_modes(amp: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Inverse of :func:`modes_to_amp_phase`."""
    return amp * np.exp(1j * phase)


def project_to_strain(
    modes_complex: np.ndarray,
    swsh_vec: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project modes to ``(h_plus, h_cross)`` on the time grid.

    ``swsh_vec`` is ``_{-2}Y_lm(iota, phi)`` per mode (complex, shape ``(n_modes,)``),
    in the same mode ordering. ``z = sum_lm h_lm Y_lm``; ``h_+ = Re z, h_x = -Im z``.
    """
    z = np.tensordot(swsh_vec, modes_complex, axes=(0, 0))
    return np.real(z), -np.imag(z)
