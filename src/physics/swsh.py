"""Differentiable spin-weighted spherical harmonics ``_{-2}Y_{lm}(iota, phi)``.

Implemented in PyTorch via the Wigner small-``d`` closed form so the projection
from inertial-frame modes to ``(h_+, h_x)`` is differentiable in the orientation
angles ``(iota, phi)``.

Convention
----------
We use the Goldberg et al. (1967) convention, which is the one LAL's
``XLALSpinWeightedSphericalHarmonic`` follows::

    _{s}Y_{lm}(theta, phi) = (-1)^m * sqrt((2l+1)/(4pi))
                             * d^l_{m,-s}(theta) * exp(i m phi)

For spin weight ``s = -2`` this needs ``d^l_{m, 2}(theta)``. The Wigner small-``d``
function is evaluated from its explicit finite sum (exact for the integer
``l <= 4`` we need); only powers of ``cos(theta/2)`` and ``sin(theta/2)`` appear,
so the result is smooth and autograd-friendly in ``theta = iota``.

Rather than reasoning about sign conventions, this module is validated to
``<= 1e-10`` against ``lal.SpinWeightedSphericalHarmonic`` in
``tests/test_swsh.py`` / the M1 check; if that passes the convention is correct.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple

import torch

SPIN_WEIGHT = -2


@lru_cache(maxsize=None)
def _wigner_d_terms(ell: int, m: int, mp: int) -> Tuple[Tuple[float, int, int], ...]:
    """Return ``(coeff, pow_cos, pow_sin)`` terms of the Wigner small-d sum.

    ``d^ell_{m, mp}(theta) = sum_k coeff * cos(theta/2)^pow_cos * sin(theta/2)^pow_sin``
    with the Edmonds/Wikipedia closed form::

        coeff_k = (-1)^(k - m + mp)
                  * sqrt((l+m)!(l-m)!(l+mp)!(l-mp)!)
                  / ((l+m-k)! k! (l-k-mp)! (k-m+mp)!)
        pow_cos = 2l - 2k + m - mp
        pow_sin = 2k - m + mp
    """
    terms: List[Tuple[float, int, int]] = []
    k_min = max(0, m - mp)
    k_max = min(ell + m, ell - mp)
    pref = math.sqrt(
        math.factorial(ell + m)
        * math.factorial(ell - m)
        * math.factorial(ell + mp)
        * math.factorial(ell - mp)
    )
    for k in range(k_min, k_max + 1):
        denom = (
            math.factorial(ell + m - k)
            * math.factorial(k)
            * math.factorial(ell - k - mp)
            * math.factorial(k - m + mp)
        )
        coeff = ((-1.0) ** (k - m + mp)) * pref / denom
        pow_cos = 2 * ell - 2 * k + m - mp
        pow_sin = 2 * k - m + mp
        terms.append((coeff, pow_cos, pow_sin))
    return tuple(terms)


def wigner_d(ell: int, m: int, mp: int, theta: torch.Tensor) -> torch.Tensor:
    """Differentiable real Wigner small-d ``d^ell_{m, mp}(theta)``."""
    half = 0.5 * theta
    c = torch.cos(half)
    s = torch.sin(half)
    out = torch.zeros_like(theta)
    for coeff, pc, ps in _wigner_d_terms(ell, m, mp):
        out = out + coeff * (c ** pc) * (s ** ps)
    return out


def sYlm(
    ell: int,
    m: int,
    theta: torch.Tensor,
    phi: torch.Tensor,
    spin: int = SPIN_WEIGHT,
) -> torch.Tensor:
    """Single complex spin-weighted spherical harmonic ``_{spin}Y_{lm}(theta, phi)``.

    ``theta`` (= inclination ``iota``) and ``phi`` are broadcastable real tensors.
    Returns a complex tensor of the broadcasted shape.
    """
    prefactor = ((-1.0) ** m) * math.sqrt((2 * ell + 1) / (4.0 * math.pi))
    d = wigner_d(ell, m, -spin, theta)
    real = prefactor * d * torch.cos(m * phi)
    imag = prefactor * d * torch.sin(m * phi)
    return torch.complex(real, imag)


class SWSHProjector:
    """Caches the ``(l, m)`` ordering and evaluates the SWSH vector in one shot.

    ``modes`` is the ordered list of ``(l, m)`` keys matching the surrogate's mode
    output ordering; ``forward(iota, phi)`` returns a complex tensor of shape
    ``(..., n_modes)`` so it contracts directly against a stacked mode array.
    """

    def __init__(self, modes: Sequence[Tuple[int, int]], spin: int = SPIN_WEIGHT):
        self.modes: List[Tuple[int, int]] = [(int(l), int(m)) for l, m in modes]
        self.spin = int(spin)

    def __len__(self) -> int:
        return len(self.modes)

    def forward(self, iota: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        cols = [sYlm(l, m, iota, phi, spin=self.spin) for (l, m) in self.modes]
        return torch.stack(cols, dim=-1)

    __call__ = forward


def default_modes(ell_max: int = 4, include_m0: bool = True) -> List[Tuple[int, int]]:
    """All ``(l, m)`` with ``2 <= l <= ell_max``; drop ``m = 0`` if requested."""
    modes: List[Tuple[int, int]] = []
    for ell in range(2, ell_max + 1):
        for m in range(-ell, ell + 1):
            if m == 0 and not include_m0:
                continue
            modes.append((ell, m))
    return modes
