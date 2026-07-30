"""
Conditional Neural Spline Flow for gravitational waveform generation.

The model learns the conditional distribution p(coefficients | parameters),
where:
  - parameters are 7D BBH physical parameters [q, χ₁ₓ, χ₁ᵧ, χ₁_z, χ₂ₓ, χ₂ᵧ, χ₂_z]
  - coefficients are 200D standardised SVD basis coefficients of h+(f)

Architecture:
  1. A conditioning MLP encodes physical parameters into a context vector.
  2. A sequence of masked autoregressive rational-quadratic spline transforms,
     interleaved with random permutations and batch normalisation, maps a
     standard Gaussian base distribution to the coefficient distribution
     conditioned on the context vector.

Reference: Durkan et al. "Neural Spline Flows", NeurIPS 2019.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    RandomPermutation,
    BatchNorm,
)
from typing import Mapping, Optional


class FourierFeatures(nn.Module):
    """Random Fourier Features to resolve spectral bias in highly oscillatory mappings."""
    def __init__(self, input_dim: int, fourier_dim: int = 64, scale: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(input_dim, fourier_dim) * scale)
        self.out_dim = input_dim + 2 * fourier_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * torch.pi * (x @ self.B)
        return torch.cat([x, torch.sin(proj), torch.cos(proj)], dim=-1)
class ConditioningNetwork(nn.Module):
    """MLP that encodes physical parameters into a fixed-size context embedding.

    Input : 7D [q, χ₁ₓ, χ₁ᵧ, χ₁_z, χ₂ₓ, χ₂ᵧ, χ₂_z] (pre-normalised to [-1, 1])
    Output: context_dim embedding vector passed to each coupling layer of the flow.

    LayerNorm after each linear layer stabilises training without requiring
    careful learning-rate tuning.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dims: list = [256, 256, 256],
        context_dim: int = 64,
        dropout: float = 0.0,
        fourier_dim: int = 64,
        fourier_scale: float = 10.0,
    ):
        super().__init__()

        self.fourier = FourierFeatures(input_dim, fourier_dim, fourier_scale) if fourier_dim > 0 else None

        layers = []
        in_dim = self.fourier.out_dim if self.fourier is not None else input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, context_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            x = self.fourier(x)
        return self.net(x)


class WaveformFlow(nn.Module):
    """Conditional normalizing flow: BBH parameters → waveform SVD coefficients.

    Uses rational-quadratic neural spline coupling layers (NSF) conditioned on
    physical parameters via a learned context vector. Training maximises the
    log-likelihood of the true SVD coefficients under the learned distribution.
    """

    def __init__(
        self,
        data_dim: int = 200,           # 2 * n_basis (Re + Im SVD coefficients)
        context_dim: int = 64,         # size of the context embedding
        n_coupling_layers: int = 10,   # number of autoregressive transforms
        hidden_features: int = 256,    # width of MADE networks inside each transform
        n_spline_bins: int = 8,        # number of spline knots per dimension
        tail_bound: float = 5.0,       # linear tails beyond ±tail_bound σ
        conditioning_hidden_dims: list = [256, 256, 256],
        param_dim: int = 7,
        dropout: float = 0.0,
        fourier_dim: int = 64,
        fourier_scale: float = 10.0,
    ):
        super().__init__()

        self.data_dim = data_dim
        self.context_dim = context_dim
        self.param_dim = param_dim
        self.metric_name = "NLL"

        # Context encoder: parameters → embedding for the flow
        self.context_net = ConditioningNetwork(
            input_dim=param_dim,
            hidden_dims=conditioning_hidden_dims,
            context_dim=context_dim,
            dropout=dropout,
            fourier_dim=fourier_dim,
            fourier_scale=fourier_scale,
        )

        # Build the sequence of flow transforms
        transforms = []
        for i in range(n_coupling_layers):
            # Masked autoregressive rational-quadratic spline
            transforms.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=data_dim,
                    hidden_features=hidden_features,
                    context_features=context_dim,
                    num_bins=n_spline_bins,
                    tails="linear",
                    tail_bound=tail_bound,
                    num_blocks=2,
                    use_residual_blocks=True,
                    activation=torch.nn.functional.gelu,
                    dropout_probability=dropout,
                )
            )
            # Randomly permute dimensions to mix information across coupling layers
            transforms.append(RandomPermutation(features=data_dim))
            # Batch normalisation between all but the final coupling layer
            if i < n_coupling_layers - 1:
                transforms.append(BatchNorm(features=data_dim))

        base_dist = StandardNormal([data_dim])

        self.flow = Flow(
            transform=CompositeTransform(transforms),
            distribution=base_dist,
        )

    def forward(self, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Compute the log-probability of coefficients given parameters.

        Used as the training objective: loss = -mean(log_prob).

        Args:
            params : (batch, 7)        normalised physical parameters
            coeffs : (batch, data_dim) standardised SVD coefficients

        Returns:
            log_prob: (batch,) log p(coeffs | params)
        """
        context = self.context_net(params)
        return self.flow.log_prob(coeffs, context=context)

    def training_loss(self, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Return the scalar training objective for the flow."""
        return -self(params, coeffs).mean()

    def latent_from_coeffs(self, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Map waveform coefficients into the conditional base latent space.

        The returned z coordinates are the morphology latent variables for the
        current physical parameters. Nearby points in this space are the natural
        place to probe generative variations, instead of averaging waveform
        coefficients directly.
        """
        context = self.context_net(params)
        latent, _ = self.flow._transform(coeffs, context=context)
        return latent

    def coeffs_from_latent(self, params: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """Decode conditional base-latent coordinates into waveform coefficients."""
        if latent.shape[-1] != self.data_dim:
            raise ValueError(
                f"Expected latent dimension {self.data_dim}, got {latent.shape[-1]}"
            )
        context = self.context_net(params)
        coeffs, _ = self.flow._transform.inverse(latent, context=context)
        return coeffs

    def sample(self, params: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Draw waveform coefficient samples conditioned on parameters.

        Args:
            params   : (batch, 7) normalised physical parameters
            n_samples: number of independent samples per parameter point

        Returns:
            coeffs: (batch * n_samples, data_dim) sampled coefficients
        """
        context = self.context_net(params)
        if n_samples > 1:
            context = context.repeat_interleave(n_samples, dim=0)
        return self.flow.sample(1, context=context).squeeze(1)

    @torch.no_grad()
    def _generate_waveform_legacy(self, params: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Generate SVD coefficients for a batch of parameter points.

        Args:
            params       : (batch, 7) normalised physical parameters
            deterministic: If True, use z=0 (MAP / mean of base distribution).
                           If False, draw z ~ N(0, 0.5²) for stochastic generation.
                           Stochastic mode better reflects the learned uncertainty.

        Returns:
            coeffs: (batch, data_dim) generated SVD coefficients
        """
        context = self.context_net(params)
        if deterministic:
            z = torch.zeros(params.shape[0], self.data_dim, device=params.device)
        else:
            # Sample with reduced temperature (σ=0.5) to avoid extreme tail samples
            z = torch.randn(params.shape[0], self.data_dim, device=params.device) * 0.5

        coeffs, _ = self.flow._transform.inverse(z, context=context)
        return coeffs

    @torch.no_grad()
    def generate_waveform(
        self,
        params: torch.Tensor,
        deterministic: bool = False,
        latent: Optional[torch.Tensor] = None,
        temperature: float = 0.5,
    ) -> torch.Tensor:
        """Generate SVD coefficients with optional explicit morphology latent z."""
        if latent is not None:
            z = latent.to(device=params.device, dtype=params.dtype)
            if z.ndim == 1:
                z = z.unsqueeze(0)
            if z.shape[0] == 1 and params.shape[0] > 1:
                z = z.expand(params.shape[0], -1)
            if z.shape != (params.shape[0], self.data_dim):
                raise ValueError(
                    "latent must have shape "
                    f"({params.shape[0]}, {self.data_dim}) or ({self.data_dim},)"
                )
        elif deterministic:
            z = torch.zeros(params.shape[0], self.data_dim, device=params.device)
        else:
            z = torch.randn(params.shape[0], self.data_dim, device=params.device) * float(temperature)

        return self.coeffs_from_latent(params, z)


class ResNetBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.fc1(x)
        h = F.gelu(h)
        h = self.norm1(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return x + F.gelu(h)


def _init_regression_loss_state(
    module: nn.Module,
    output_dim: int,
    loss_type: str,
    huber_delta: float,
    loss_weighting: str,
):
    """Initialize coefficient and match-aware loss state for regressors."""
    module.loss_type = loss_type.lower()
    module.huber_delta = float(huber_delta)
    module.loss_weighting = loss_weighting.lower()
    if "match" in module.loss_type:
        module.metric_name = "MatchLoss"
    elif module.loss_type in {"physical_mse", "relative_mse"}:
        module.metric_name = "PhysicalMSE"
    else:
        module.metric_name = "MSE" if module.loss_type == "mse" else "Huber"

    module.register_buffer("coeffs_std", torch.ones(output_dim))
    module.register_buffer("coeffs_mean", torch.zeros(output_dim), persistent=False)
    module.register_buffer("waveform_mean_proj", torch.zeros(output_dim), persistent=False)
    module.register_buffer("waveform_mean_norm_sq", torch.ones((), dtype=torch.float32), persistent=False)
    module.register_buffer("waveform_joint_scale", torch.ones((), dtype=torch.float32), persistent=False)
    module.waveform_loss_ready = False
    module.coeff_loss_weight = 0.0
    module.match_loss_weight = 1.0
    module.norm_loss_weight = 0.0
    module.match_phase_max = True
    module.match_eps = 1e-12


def _configure_regression_loss_state(
    module: nn.Module,
    compressor,
    coeffs_mean,
    coeffs_std,
    model_cfg: Mapping,
):
    """Attach training-set statistics and compressed-space overlap geometry."""
    device = module.coeffs_std.device
    module.coeffs_std.copy_(torch.as_tensor(coeffs_std, dtype=torch.float32, device=device))
    module.coeffs_mean.copy_(torch.as_tensor(coeffs_mean, dtype=torch.float32, device=device))

    module.coeff_loss_weight = float(model_cfg.get("coeff_loss_weight", 0.02 if "match" in module.loss_type else 1.0))
    module.match_loss_weight = float(model_cfg.get("match_loss_weight", 1.0))
    module.norm_loss_weight = float(model_cfg.get("norm_loss_weight", 0.02 if "match" in module.loss_type else 0.0))
    module.match_phase_max = bool(model_cfg.get("match_phase_max", True))
    module.match_eps = float(model_cfg.get("match_eps", 1e-12))

    needs_waveform_geometry = (
        "match" in module.loss_type
        or module.loss_type in {"physical_mse", "relative_mse"}
    )
    if not needs_waveform_geometry:
        return

    representation = getattr(compressor, "representation", "").lower()
    if representation != "joint_whitened_complex":
        raise ValueError(
            "match-aware coefficient loss currently requires "
            "compression.representation: joint_whitened_complex"
        )

    basis = np.asarray(compressor.basis_real, dtype=np.float32)
    mean = np.asarray(compressor.mean_real, dtype=np.float32)
    joint_scale = max(float(compressor.joint_scale), 1e-30)
    # Work in the compressor's dimensionless centered-SVD units. The physical
    # strain scale is ~1e-22, so raw strain-space norms underflow float32 and
    # make overlap losses numerically meaningless.
    mean_proj = (basis @ mean) / joint_scale
    mean_norm_sq = np.float32(np.dot(mean, mean) / (joint_scale * joint_scale))

    module.waveform_mean_proj.copy_(torch.as_tensor(mean_proj, dtype=torch.float32, device=device))
    module.waveform_mean_norm_sq.copy_(torch.as_tensor(mean_norm_sq, dtype=torch.float32, device=device))
    module.waveform_joint_scale.copy_(torch.ones((), dtype=torch.float32, device=device))
    module.waveform_loss_ready = True


def _coefficient_regression_loss(module: nn.Module, pred: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    if module.loss_weighting in {"none", "uniform"}:
        weight = torch.ones_like(module.coeffs_std)
    elif module.loss_weighting in {"coeff_std", "physical"}:
        weight = module.coeffs_std / module.coeffs_std.max()
    else:
        raise ValueError("loss_weighting must be one of: 'coeff_std', 'physical', 'none'")

    pred_weighted = pred * weight.unsqueeze(0)
    coeffs_weighted = coeffs * weight.unsqueeze(0)
    if module.loss_type in {"mse", "match", "match_mse", "overlap", "overlap_mse"}:
        return F.mse_loss(pred_weighted, coeffs_weighted)
    if module.loss_type == "huber":
        return F.smooth_l1_loss(pred_weighted, coeffs_weighted, beta=module.huber_delta)
    raise ValueError(
        "Regressor loss_type must be one of: 'mse', 'huber', 'match', "
        "'match_mse', 'overlap', or 'overlap_mse'"
    )


def _raw_coeffs(module: nn.Module, coeffs_std: torch.Tensor) -> torch.Tensor:
    return coeffs_std * module.coeffs_std.unsqueeze(0) + module.coeffs_mean.unsqueeze(0)


def _whitened_norm_sq(module: nn.Module, raw_coeffs: torch.Tensor) -> torch.Tensor:
    scale = module.waveform_joint_scale
    mean_dot = raw_coeffs @ module.waveform_mean_proj
    coeff_norm = torch.sum(raw_coeffs * raw_coeffs, dim=1)
    return (
        module.waveform_mean_norm_sq
        + 2.0 * scale * mean_dot
        + scale * scale * coeff_norm
    ).clamp_min(module.match_eps)


def _whitened_dot(module: nn.Module, raw_a: torch.Tensor, raw_b: torch.Tensor) -> torch.Tensor:
    scale = module.waveform_joint_scale
    mean_dot = (raw_a + raw_b) @ module.waveform_mean_proj
    coeff_dot = torch.sum(raw_a * raw_b, dim=1)
    return module.waveform_mean_norm_sq + scale * mean_dot + scale * scale * coeff_dot


def _match_aware_regression_loss(
    module: nn.Module,
    pred: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    if not module.waveform_loss_ready:
        raise RuntimeError("match-aware loss was requested before configure_waveform_loss()")

    pred_raw = _raw_coeffs(module, pred)
    target_raw = _raw_coeffs(module, coeffs)
    pred_norm_sq = _whitened_norm_sq(module, pred_raw)
    target_norm_sq = _whitened_norm_sq(module, target_raw)
    dot = _whitened_dot(module, pred_raw, target_raw)
    denom = torch.sqrt((pred_norm_sq * target_norm_sq).clamp_min(module.match_eps))
    match = dot / denom
    match = torch.nan_to_num(match, nan=0.0, posinf=0.0, neginf=0.0)
    if module.match_phase_max:
        match = torch.abs(match)
    match = match.clamp(0.0, 1.0)

    loss = module.match_loss_weight * torch.mean(1.0 - match)
    if module.norm_loss_weight > 0:
        log_pred_norm = 0.5 * torch.log(pred_norm_sq)
        log_target_norm = 0.5 * torch.log(target_norm_sq)
        loss = loss + module.norm_loss_weight * F.mse_loss(log_pred_norm, log_target_norm)
    if module.coeff_loss_weight > 0:
        loss = loss + module.coeff_loss_weight * _coefficient_regression_loss(module, pred, coeffs)
    return loss


def _physical_mse_regression_loss(
    module: nn.Module,
    pred: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    if not module.waveform_loss_ready:
        raise RuntimeError("physical_mse loss was requested before configure_waveform_loss()")

    pred_raw = torch.nan_to_num(_raw_coeffs(module, pred), nan=0.0, posinf=0.0, neginf=0.0)
    target_raw = torch.nan_to_num(_raw_coeffs(module, coeffs), nan=0.0, posinf=0.0, neginf=0.0)
    diff = pred_raw - target_raw
    target_norm_sq = _whitened_norm_sq(module, target_raw)
    rel_sq_error = torch.sum(diff * diff, dim=1) / target_norm_sq
    rel_sq_error = torch.nan_to_num(rel_sq_error, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.mean(rel_sq_error)


def _regression_training_loss(module: nn.Module, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
    pred = module(params)
    if "match" in module.loss_type or "overlap" in module.loss_type:
        return _match_aware_regression_loss(module, pred, coeffs)
    if module.loss_type in {"physical_mse", "relative_mse"}:
        return _physical_mse_regression_loss(module, pred, coeffs)
    return _coefficient_regression_loss(module, pred, coeffs)


def _apply_output_bound(module: nn.Module, output: torch.Tensor) -> torch.Tensor:
    scale = float(getattr(module, "output_tanh_scale", 0.0))
    if scale <= 0:
        return output
    return scale * torch.tanh(output / scale)


class ResNetRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 200,
        hidden_dim: int = 512,
        n_blocks: int = 4,
        dropout: float = 0.0,
        loss_type: str = "huber",
        huber_delta: float = 0.05,
        loss_weighting: str = "coeff_std",
        fourier_dim: int = 64,
        fourier_scale: float = 10.0,
        output_tanh_scale: float = 0.0,
    ):
        super().__init__()
        self.output_tanh_scale = float(output_tanh_scale)
        self.fourier = FourierFeatures(input_dim, fourier_dim, fourier_scale) if fourier_dim > 0 else None
        in_dim = self.fourier.out_dim if self.fourier is not None else input_dim

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        _init_regression_loss_state(self, output_dim, loss_type, huber_delta, loss_weighting)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            params = self.fourier(params)
        h = F.gelu(self.in_proj(params))
        for block in self.blocks:
            h = block(h)
        return _apply_output_bound(self, self.out_proj(h))

    def training_loss(self, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        return _regression_training_loss(self, params, coeffs)

    def configure_waveform_loss(self, compressor, coeffs_mean, coeffs_std, model_cfg: Mapping):
        _configure_regression_loss_state(self, compressor, coeffs_mean, coeffs_std, model_cfg)

    @torch.no_grad()
    def generate_waveform(self, params: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        return self(params)


class ModeResNetRegressor(nn.Module):
    """Shared RFF->ResNet trunk with one linear head per inertial-frame mode.

    The output is the concatenation of per-mode coefficient vectors (each mode has
    ``2 * n_basis[mode]`` channels: amp+phase, or real+imag for the m=0 memory modes),
    in the order of ``mode_keys``. There is no global output squashing — per-channel
    scaling is handled by coefficient standardization in the dataset, so the heads emit
    raw standardized coefficients. This keeps each mode's dynamic range independent (a
    single saturating tanh on a heterogeneous 3500-vector would clip the dominant modes).

    The split heads also let later stages weight the per-mode loss or fine-tune
    individual modes without touching the trunk.
    """

    def __init__(
        self,
        mode_keys,
        mode_n_basis,
        input_dim: int = 8,
        hidden_dim: int = 512,
        n_blocks: int = 6,
        dropout: float = 0.0,
        loss_type: str = "huber",
        huber_delta: float = 0.5,
        loss_weighting: str = "none",
        fourier_dim: int = 128,
        fourier_scale: float = 1.0,
    ):
        super().__init__()
        self.mode_keys = [(int(l), int(m)) for l, m in mode_keys]
        self.mode_n_basis = {(int(l), int(m)): int(nb)
                             for (l, m), nb in zip(self.mode_keys, mode_n_basis)} \
            if not isinstance(mode_n_basis, dict) else \
            {(int(l), int(m)): int(v) for (l, m), v in mode_n_basis.items()}
        output_dim = sum(2 * self.mode_n_basis[k] for k in self.mode_keys)
        self.output_dim = output_dim
        self.output_tanh_scale = 0.0  # invariant: no global squashing across modes

        self.fourier = FourierFeatures(input_dim, fourier_dim, fourier_scale) if fourier_dim > 0 else None
        in_dim = self.fourier.out_dim if self.fourier is not None else input_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(n_blocks)])
        self.heads = nn.ModuleDict({
            f"{l}_{m}": nn.Linear(hidden_dim, 2 * self.mode_n_basis[(l, m)])
            for (l, m) in self.mode_keys
        })
        _init_regression_loss_state(self, output_dim, loss_type, huber_delta, loss_weighting)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            params = self.fourier(params)
        h = F.gelu(self.in_proj(params))
        for block in self.blocks:
            h = block(h)
        return torch.cat([self.heads[f"{l}_{m}"](h) for (l, m) in self.mode_keys], dim=-1)

    def training_loss(self, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        return _regression_training_loss(self, params, coeffs)

    def configure_waveform_loss(self, compressor, coeffs_mean, coeffs_std, model_cfg: Mapping):
        # match-aware geometry isn't defined for the per-mode compressor; coefficient
        # losses (huber/mse) only need coeffs_std/mean, set below.
        device = self.coeffs_std.device
        self.coeffs_std.copy_(torch.as_tensor(coeffs_std, dtype=torch.float32, device=device))
        self.coeffs_mean.copy_(torch.as_tensor(coeffs_mean, dtype=torch.float32, device=device))

    @torch.no_grad()
    def generate_waveform(self, params: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        return self(params)


class SplitResNetRegressor(nn.Module):
    """ResNet with shared backbone that splits into separate amp and phase branches.

    Motivation: amp and phase SVD coefficients have different statistics and
    difficulty (phase ~1.5x harder). Dedicated branch capacity may help.
    """

    def __init__(
        self,
        input_dim: int = 8,
        n_basis: int = 100,
        hidden_dim: int = 512,
        n_shared_blocks: int = 3,
        n_branch_blocks: int = 3,
        dropout: float = 0.0,
        loss_type: str = "huber",
        huber_delta: float = 0.5,
        loss_weighting: str = "none",
        fourier_dim: int = 128,
        fourier_scale: float = 8.0,
        output_tanh_scale: float = 10.0,
    ):
        super().__init__()
        output_dim = 2 * n_basis
        self.n_basis = n_basis
        self.output_tanh_scale = float(output_tanh_scale)
        self.fourier = FourierFeatures(input_dim, fourier_dim, fourier_scale) if fourier_dim > 0 else None
        in_dim = self.fourier.out_dim if self.fourier is not None else input_dim

        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.shared_blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(n_shared_blocks)])
        self.amp_blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(n_branch_blocks)])
        self.phase_blocks = nn.ModuleList([ResNetBlock(hidden_dim, dropout) for _ in range(n_branch_blocks)])
        self.amp_out = nn.Linear(hidden_dim, n_basis)
        self.phase_out = nn.Linear(hidden_dim, n_basis)
        _init_regression_loss_state(self, output_dim, loss_type, huber_delta, loss_weighting)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            params = self.fourier(params)
        h = F.gelu(self.in_proj(params))
        for block in self.shared_blocks:
            h = block(h)
        h_amp = h
        for block in self.amp_blocks:
            h_amp = block(h_amp)
        h_phase = h
        for block in self.phase_blocks:
            h_phase = block(h_phase)
        out = torch.cat([self.amp_out(h_amp), self.phase_out(h_phase)], dim=-1)
        return _apply_output_bound(self, out)

    def training_loss(self, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        return _regression_training_loss(self, params, coeffs)

    def configure_waveform_loss(self, compressor, coeffs_mean, coeffs_std, model_cfg: Mapping):
        _configure_regression_loss_state(self, compressor, coeffs_mean, coeffs_std, model_cfg)

    @torch.no_grad()
    def generate_waveform(self, params: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        del deterministic
        return self(params)


class WaveformRegressor(nn.Module):
    """Deterministic baseline that maps BBH parameters directly to SVD coefficients."""

    def __init__(
        self,
        input_dim: int = 7,
        output_dim: int = 200,
        hidden_dims: list = [256, 256, 256],
        dropout: float = 0.0,
        loss_type: str = "huber",
        huber_delta: float = 0.05,
        loss_weighting: str = "coeff_std",
        fourier_dim: int = 64,
        fourier_scale: float = 10.0,
        output_tanh_scale: float = 0.0,
    ):
        super().__init__()
        self.output_tanh_scale = float(output_tanh_scale)

        self.fourier = FourierFeatures(input_dim, fourier_dim, fourier_scale) if fourier_dim > 0 else None

        layers = []
        in_dim = self.fourier.out_dim if self.fourier is not None else input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        _init_regression_loss_state(self, output_dim, loss_type, huber_delta, loss_weighting)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        if self.fourier is not None:
            params = self.fourier(params)
        return _apply_output_bound(self, self.net(params))

    def training_loss(self, params: torch.Tensor, coeffs: torch.Tensor) -> torch.Tensor:
        """Return the scalar regression objective."""
        return _regression_training_loss(self, params, coeffs)

    def configure_waveform_loss(self, compressor, coeffs_mean, coeffs_std, model_cfg: Mapping):
        _configure_regression_loss_state(self, compressor, coeffs_mean, coeffs_std, model_cfg)

    @torch.no_grad()
    def generate_waveform(self, params: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Predict standardized SVD coefficients for a batch of parameter points."""
        del deterministic
        return self(params)


def _infer_fourier_dim(
    model_cfg: dict,
    state_dict: Optional[Mapping[str, torch.Tensor]],
    buffer_key: str,
    first_weight_key: str,
    input_dim: int = 7,
) -> int:
    """Infer Fourier-feature width for checkpoints with older config schemas."""
    if "fourier_dim" in model_cfg:
        return int(model_cfg.get("fourier_dim") or 0)

    if state_dict is not None:
        if buffer_key in state_dict:
            return int(state_dict[buffer_key].shape[1])
        if first_weight_key in state_dict:
            in_features = int(state_dict[first_weight_key].shape[1])
            extra = in_features - input_dim
            if extra > 0 and extra % 2 == 0:
                return extra // 2

    return 0


def build_model(config: dict, state_dict: Optional[Mapping[str, torch.Tensor]] = None) -> nn.Module:
    """Instantiate a waveform model from a config dictionary.

    Args:
        config: parsed config.yaml dict

    Returns:
        Constructed (but not trained) waveform model
    """
    model_cfg = config["model"]
    comp_cfg  = config["compression"]
    param_dim = int(model_cfg.get("param_dim", config.get("data", {}).get("param_dim", 7)))

    model_type = model_cfg.get("type", "nsf").lower()

    if model_type in {"mode_resnet_regressor"}:
        from src.data.mode_waveforms import mode_keys as _mode_keys
        keys = _mode_keys(int(comp_cfg.get("ell_max", 4)), include_m0=comp_cfg.get("include_m0", True))
        per_mode = comp_cfg["per_mode_n_basis"]
        n_basis = {tuple(int(x) for x in k.split(":")): int(v) for k, v in per_mode.items()} \
            if isinstance(per_mode, dict) else dict(zip(keys, per_mode))
        fourier_dim = int(model_cfg.get("fourier_dim", 128) or 0)
        model = ModeResNetRegressor(
            mode_keys=keys,
            mode_n_basis=n_basis,
            input_dim=param_dim,
            hidden_dim=model_cfg.get("hidden_dims", [512])[0],
            n_blocks=int(model_cfg.get("n_blocks", 6)),
            dropout=model_cfg.get("dropout", 0.0),
            loss_type=model_cfg.get("loss", "huber"),
            huber_delta=model_cfg.get("huber_delta", 0.5),
            loss_weighting=model_cfg.get("loss_weighting", "none"),
            fourier_dim=fourier_dim,
            fourier_scale=model_cfg.get("fourier_scale", 1.0),
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: mode_resnet_regressor | output_dim={model.output_dim} | "
              f"param_dim={param_dim} | {len(keys)} modes | n_blocks={model_cfg.get('n_blocks', 6)} | "
              f"fourier_dim={fourier_dim} | parameters={n_params:,}")
        return model

    data_dim = 2 * comp_cfg["n_basis"]   # Re + Im SVD coefficients

    if model_type in {"nsf", "flow"}:
        fourier_dim = _infer_fourier_dim(
            model_cfg,
            state_dict,
            buffer_key="context_net.fourier.B",
            first_weight_key="context_net.net.0.weight",
            input_dim=param_dim,
        )
        model = WaveformFlow(
            data_dim=data_dim,
            context_dim=model_cfg["context_dim"],
            n_coupling_layers=model_cfg["n_coupling_layers"],
            hidden_features=model_cfg["hidden_dims"][0],
            n_spline_bins=model_cfg["n_spline_bins"],
            tail_bound=model_cfg["tail_bound"],
            conditioning_hidden_dims=model_cfg["hidden_dims"],
            param_dim=param_dim,
            dropout=model_cfg["dropout"],
            fourier_dim=fourier_dim,
            fourier_scale=model_cfg.get("fourier_scale", 10.0),
        )
        model_summary = (
            f"Model: flow | data_dim={data_dim} | "
            f"param_dim={param_dim} | "
            f"context_dim={model_cfg['context_dim']} | "
            f"{model_cfg['n_coupling_layers']} coupling layers | "
            f"fourier_dim={fourier_dim}"
        )
    elif model_type in {"mlp_regressor", "regressor"}:
        fourier_dim = _infer_fourier_dim(
            model_cfg,
            state_dict,
            buffer_key="fourier.B",
            first_weight_key="net.0.weight",
            input_dim=param_dim,
        )
        model = WaveformRegressor(
            input_dim=param_dim,
            output_dim=data_dim,
            hidden_dims=model_cfg["hidden_dims"],
            dropout=model_cfg.get("dropout", 0.0),
            loss_type=model_cfg.get("loss", "huber"),
            huber_delta=model_cfg.get("huber_delta", 0.05),
            loss_weighting=model_cfg.get("loss_weighting", "coeff_std"),
            fourier_dim=fourier_dim,
            fourier_scale=model_cfg.get("fourier_scale", 10.0),
            output_tanh_scale=model_cfg.get("output_tanh_scale", 0.0),
        )
        model_summary = (
            f"Model: regressor | data_dim={data_dim} | "
            f"param_dim={param_dim} | "
            f"hidden_dims={model_cfg['hidden_dims']} | "
            f"loss={model_cfg.get('loss', 'huber')} | "
            f"fourier_dim={fourier_dim}"
        )
    elif model_type in {"resnet_regressor"}:
        fourier_dim = _infer_fourier_dim(
            model_cfg,
            state_dict,
            buffer_key="fourier.B",
            first_weight_key="in_proj.weight",
            input_dim=param_dim,
        )
        model = ResNetRegressor(
            input_dim=param_dim,
            output_dim=data_dim,
            hidden_dim=model_cfg.get("hidden_dims", [512])[0],
            n_blocks=model_cfg.get("n_blocks", 4),
            dropout=model_cfg.get("dropout", 0.0),
            loss_type=model_cfg.get("loss", "huber"),
            huber_delta=model_cfg.get("huber_delta", 0.05),
            loss_weighting=model_cfg.get("loss_weighting", "coeff_std"),
            fourier_dim=fourier_dim,
            fourier_scale=model_cfg.get("fourier_scale", 10.0),
            output_tanh_scale=model_cfg.get("output_tanh_scale", 0.0),
        )
        model_summary = (
            f"Model: resnet_regressor | data_dim={data_dim} | "
            f"param_dim={param_dim} | "
            f"hidden_dim={model_cfg.get('hidden_dims', [512])[0]} | "
            f"n_blocks={model_cfg.get('n_blocks', 4)} | "
            f"loss={model_cfg.get('loss', 'huber')} | "
            f"fourier_dim={fourier_dim}"
        )
    elif model_type in {"split_resnet_regressor"}:
        fourier_dim = _infer_fourier_dim(
            model_cfg,
            state_dict,
            buffer_key="fourier.B",
            first_weight_key="in_proj.weight",
            input_dim=param_dim,
        )
        n_basis = comp_cfg["n_basis"]
        n_shared = int(model_cfg.get("n_shared_blocks", 3))
        n_branch = int(model_cfg.get("n_branch_blocks", 3))
        model = SplitResNetRegressor(
            input_dim=param_dim,
            n_basis=n_basis,
            hidden_dim=model_cfg.get("hidden_dims", [512])[0],
            n_shared_blocks=n_shared,
            n_branch_blocks=n_branch,
            dropout=model_cfg.get("dropout", 0.0),
            loss_type=model_cfg.get("loss", "huber"),
            huber_delta=model_cfg.get("huber_delta", 0.5),
            loss_weighting=model_cfg.get("loss_weighting", "none"),
            fourier_dim=fourier_dim,
            fourier_scale=model_cfg.get("fourier_scale", 8.0),
            output_tanh_scale=model_cfg.get("output_tanh_scale", 0.0),
        )
        model_summary = (
            f"Model: split_resnet_regressor | data_dim={data_dim} | "
            f"param_dim={param_dim} | hidden_dim={model_cfg.get('hidden_dims', [512])[0]} | "
            f"n_shared={n_shared} n_branch={n_branch} | "
            f"loss={model_cfg.get('loss', 'huber')} | fourier_dim={fourier_dim}"
        )
    else:
        raise ValueError(
            "model.type must be one of: 'nsf', 'flow', 'mlp_regressor', 'regressor', "
            "'resnet_regressor', 'split_resnet_regressor'"
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{model_summary} | "
        f"parameters={n_params:,}"
    )

    return model
