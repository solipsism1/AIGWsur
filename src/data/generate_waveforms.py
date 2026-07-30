"""
Generate NRSur7dq4 training data in the frequency domain.

Samples BBH parameter points using a quasi-random Sobol sequence (for good
coverage of the 7D parameter space), calls PyCBC to generate frequency-domain
waveforms, and saves them to an HDF5 file suitable for WaveformDataset.

Parameter space:
    q      ∈ [1, 4]          (mass ratio, ≥1 by convention)
    χ₁, χ₂ ∈ ball(|χ| ≤ 0.8)  (dimensionless spin vectors per body)

Output HDF5 datasets:
    parameters  : (N, 7)  [q, χ₁ₓ, χ₁ᵧ, χ₁_z, χ₂ₓ, χ₂ᵧ, χ₂_z]
    frequencies : (n_freq,)
    hp_real     : (N, n_freq)  Re[h+(f)]
    hp_imag     : (N, n_freq)  Im[h+(f)]

Usage:
    python src/data/generate_waveforms.py --config config.yaml \\
        --output training_data/waveforms_train.h5 --split train
"""

import argparse
import time
import numpy as np
import h5py
import yaml
from scipy.stats import qmc
from tqdm import tqdm


MTSUN_SI = 4.925491025543576e-6


def omega0_to_f_ref_hz(omega0: float, total_mass_msun: float) -> float:
    """Convert dimensionless orbital omega0 to LAL/PyCBC GW reference frequency."""
    return float(omega0) / (np.pi * float(total_mass_msun) * MTSUN_SI)


def f_ref_hz_to_omega0(f_ref_hz: float, total_mass_msun: float) -> float:
    """Convert LAL/PyCBC GW reference frequency to dimensionless orbital omega0."""
    return np.pi * float(total_mass_msun) * MTSUN_SI * float(f_ref_hz)


def uses_reference_omega(config: dict) -> bool:
    """Whether to append omega0 as an eighth parameter."""
    return bool(config["data"].get("sample_reference_omega", False))


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def load_reference_omega_pool(config: dict) -> np.ndarray:
    """Load optional omega0 values from benchmark HDF5 metadata."""
    paths = _as_list(config["data"].get("reference_omega_source_h5"))
    if not paths:
        return np.array([], dtype=np.float64)

    values = []
    for path in paths:
        with h5py.File(path, "r") as f:
            if "metadata" in f and "omega0" in f["metadata"]:
                omega = np.asarray(f["metadata/omega0"][:], dtype=np.float64)
            elif "omega0" in f:
                omega = np.asarray(f["omega0"][:], dtype=np.float64)
            else:
                raise KeyError(f"No omega0 dataset found in {path}")
        values.append(omega)

    pool = np.concatenate(values)
    omega_min, omega_max = config["data"].get("reference_omega_range", [np.min(pool), np.max(pool)])
    pool = pool[(pool >= float(omega_min)) & (pool <= float(omega_max))]
    if len(pool) == 0:
        raise ValueError(
            "reference_omega_source_h5 was provided, but no omega0 values "
            f"survived range {omega_min}..{omega_max}"
        )
    return np.sort(pool.astype(np.float64))


def sample_parameters(n_samples: int, config: dict, seed: int = 42) -> np.ndarray:
    """Sample BBH physical parameters using a quasi-random sequence.

    Uses a Sobol sequence by default for uniform, low-discrepancy coverage of
    the 7D parameter space. Samples outside the spin-magnitude constraint
    (|χ| > chi_max) are rejected and replaced recursively.

    Args:
        n_samples: number of valid parameter vectors to return
        config   : parsed config.yaml dict
        seed     : random seed for reproducibility

    Returns:
        params: (n_samples, 7) array [q, χ₁ₓ, χ₁ᵧ, χ₁_z, χ₂ₓ, χ₂ᵧ, χ₂_z]
    """
    q_min, q_max = config["data"]["q_range"]
    chi_max = config["data"]["chi_mag_max"]
    include_omega = uses_reference_omega(config)
    omega_pool = load_reference_omega_pool(config) if include_omega else np.array([])
    dim = 7 if len(omega_pool) else (8 if include_omega else 7)

    if config["data"]["sampling"] == "sobol":
        sampler = qmc.Sobol(d=dim, seed=seed)
        samples_unit = sampler.random(n_samples)
    elif config["data"]["sampling"] == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        samples_unit = sampler.random(n_samples)
    else:
        rng = np.random.default_rng(seed)
        samples_unit = rng.uniform(0, 1, size=(n_samples, dim))

    # Map unit hypercube samples to physical ranges
    params = np.zeros((samples_unit.shape[0], 8 if include_omega else 7), dtype=samples_unit.dtype)
    params[:, 0] = samples_unit[:, 0] * (q_max - q_min) + q_min   # mass ratio q
    for i in range(1, 7):
        params[:, i] = samples_unit[:, i] * 2 * chi_max - chi_max  # spin components
    if include_omega:
        if len(omega_pool):
            rng = np.random.default_rng(seed + 7919)
            mode = str(config["data"].get("reference_omega_sampling", "choice")).lower()
            if mode == "cycle":
                offset = int(rng.integers(0, len(omega_pool)))
                params[:, 7] = omega_pool[(np.arange(n_samples) + offset) % len(omega_pool)]
            elif mode == "choice":
                params[:, 7] = rng.choice(omega_pool, size=n_samples, replace=True)
            else:
                raise ValueError("reference_omega_sampling must be 'choice' or 'cycle'")
        else:
            omega_min, omega_max = config["data"].get("reference_omega_range", [0.0098, 0.0244])
            params[:, 7] = samples_unit[:, 7] * (omega_max - omega_min) + omega_min

    # Reject samples where either spin magnitude exceeds chi_max
    # (cube sampling over-covers relative to the sphere by a factor ~1/0.52)
    chi1_mag = np.sqrt(params[:, 1]**2 + params[:, 2]**2 + params[:, 3]**2)
    chi2_mag = np.sqrt(params[:, 4]**2 + params[:, 5]**2 + params[:, 6]**2)
    valid = (chi1_mag <= chi_max) & (chi2_mag <= chi_max)
    params = params[valid]

    if len(params) < n_samples:
        print(
            f"  Warning: {len(params)}/{n_samples} valid after spin-magnitude cut. "
            f"Resampling the remainder..."
        )
        extra = sample_parameters(n_samples - len(params), config, seed=seed + 1000)
        params = np.vstack([params, extra])[:n_samples]

    return params[:n_samples]


def _crop_frequency_series(hp, config: dict):
    """Crop a PyCBC FrequencySeries to the configured training grid."""
    delta_f = float(config["data"]["delta_f"])
    f_lower = float(config["data"]["f_lower"])
    f_final = float(config["data"]["f_final"])

    hp_array = np.asarray(hp.data, dtype=np.complex128)
    freqs = np.asarray(hp.sample_frequencies, dtype=np.float64)
    mask = (freqs >= f_lower - 0.5 * delta_f) & (freqs <= f_final + 0.5 * delta_f)
    return freqs[mask], hp_array[mask]


def _time_domain_to_frequency_domain(waveform_kwargs: dict, config: dict):
    """Generate a time-domain waveform and FFT it to the configured grid."""
    from pycbc.waveform import get_td_waveform

    delta_f = float(config["data"]["delta_f"])
    f_final = float(config["data"]["f_final"])
    sample_rate = float(config["data"].get("sample_rate", max(4096.0, 4.0 * f_final)))

    hp_td, _ = get_td_waveform(
        delta_t=1.0 / sample_rate,
        **waveform_kwargs,
    )
    hp_fd = hp_td.to_frequencyseries(delta_f=delta_f)
    return _crop_frequency_series(hp_fd, config)


def generate_single_waveform(q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z,
                              omega0=None, config: dict = None):
    """Generate a single NRSur7dq4 waveform in the frequency domain via PyCBC.

    The total mass is fixed at a reference value (config.data.reference_total_mass)
    to make the waveform scale-free; physical strain amplitudes can be recovered
    analytically by rescaling. Distance is fixed at 1 Mpc and inclination at 0
    (face-on), consistent with the h+ polarisation used throughout.

    Args:
        q, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z: scalar BBH parameters
        omega0: optional dimensionless orbital reference frequency
        config: parsed config.yaml dict

    Returns:
        (freqs, hp_real, hp_imag) on success, or None if PyCBC raises.
    """
    if config is None:
        raise ValueError("config is required")
    from pycbc.waveform import get_fd_waveform

    M_total = float(config["data"]["reference_total_mass"])
    m1 = M_total * q / (1.0 + q)
    m2 = M_total / (1.0 + q)
    f_lower_crop = float(config["data"]["f_lower"])

    waveform_kwargs = dict(
        approximant=config["data"]["approximant"],
        mass1=m1,
        mass2=m2,
        spin1x=chi1x, spin1y=chi1y, spin1z=chi1z,
        spin2x=chi2x, spin2y=chi2y, spin2z=chi2z,
        f_lower=f_lower_crop,
        distance=1.0,        # 1 Mpc reference distance
        inclination=0.0,     # face-on (maximises h+ signal)
        coa_phase=0.0,
    )
    if omega0 is not None:
        f_ref_hz = omega0_to_f_ref_hz(float(omega0), M_total)
        # LAL's NRSur utilities require f_ref >= f_min. Generate from the
        # lower reference frequency when needed, then crop back to f_lower.
        if bool(config["data"].get("use_zero_generation_f_min", False)):
            waveform_f_lower = 0.0
        elif bool(config["data"].get("allow_lower_generation_f_min", True)):
            waveform_f_lower = min(f_lower_crop, f_ref_hz)
        else:
            waveform_f_lower = f_lower_crop
        floor = float(config["data"].get("generation_f_lower_floor", 0.0))
        waveform_kwargs["f_lower"] = waveform_f_lower if waveform_f_lower == 0.0 else max(waveform_f_lower, floor)
        waveform_kwargs["f_ref"] = f_ref_hz

    try:
        hp, _ = get_fd_waveform(
            delta_f=config["data"]["delta_f"],
            f_final=config["data"]["f_final"],
            **waveform_kwargs,
        )
        freqs, hp_array = _crop_frequency_series(hp, config)
    except Exception as e:
        try:
            freqs, hp_array = _time_domain_to_frequency_domain(waveform_kwargs, config)
            return freqs, np.real(hp_array).astype(np.float32), np.imag(hp_array).astype(np.float32)
        except Exception as td_error:
            print(f"  TD fallback also failed after FD error: {td_error}")
        print(
            f"  Failed: q={q:.2f}, χ₁=({chi1x:.2f},{chi1y:.2f},{chi1z:.2f}), "
            f"χ₂=({chi2x:.2f},{chi2y:.2f},{chi2z:.2f}): {e}"
        )
        return None

    return freqs, np.real(hp_array).astype(np.float32), np.imag(hp_array).astype(np.float32)


def generate_dataset(config: dict, n_samples: int, output_path: str, seed: int = 42):
    """Generate a full waveform dataset and save to HDF5.

    Args:
        config     : parsed config.yaml dict
        n_samples  : number of waveforms to generate
        output_path: destination HDF5 file path
        seed       : random seed for parameter sampling
    """
    include_omega = uses_reference_omega(config)
    print(f"Sampling {n_samples} parameter points...")
    params = sample_parameters(n_samples, config, seed=seed)

    # Generate one waveform to determine the frequency array shape
    test_result = None
    for i in range(min(10, len(params))):
        test_result = generate_single_waveform(*params[i], config=config)
        if test_result is not None:
            break

    if test_result is None:
        raise RuntimeError("Could not generate any valid waveform with the current config.")

    freqs, _, _ = test_result
    n_freq = len(freqs)
    print(
        f"Frequency array: {n_freq} bins, "
        f"Δf={freqs[1]-freqs[0]:.4f} Hz, "
        f"{freqs[0]:.1f}–{freqs[-1]:.1f} Hz"
    )

    # Pre-allocate output arrays
    hp_real_all = np.zeros((n_samples, n_freq), dtype=np.float32)
    hp_imag_all = np.zeros((n_samples, n_freq), dtype=np.float32)
    valid_mask  = np.zeros(n_samples, dtype=bool)

    print(f"Generating {n_samples} waveforms...")
    t0 = time.time()

    for i in tqdm(range(n_samples)):
        result = generate_single_waveform(*params[i], config=config)
        if result is None:
            continue

        _, hp_real, hp_imag = result

        # Pad or truncate to the reference frequency-array length
        n = min(len(hp_real), n_freq)
        hp_real_all[i, :n] = hp_real[:n]
        hp_imag_all[i, :n] = hp_imag[:n]
        valid_mask[i] = True

    elapsed = time.time() - t0
    n_valid = valid_mask.sum()
    print(
        f"Generated {n_valid}/{n_samples} valid waveforms "
        f"in {elapsed:.1f}s ({elapsed/max(n_valid,1):.3f}s per waveform)"
    )

    # Save to HDF5
    print(f"Saving to {output_path}...")
    with h5py.File(output_path, "w") as f:
        params_valid = params[valid_mask].astype(np.float32)
        f.create_dataset("parameters", data=params_valid)
        f.create_dataset("frequencies", data=freqs.astype(np.float32))
        f.create_dataset("hp_real", data=hp_real_all[valid_mask])
        f.create_dataset("hp_imag", data=hp_imag_all[valid_mask])
        if include_omega:
            mtot = float(config["data"]["reference_total_mass"])
            f_ref_hz = np.array(
                [omega0_to_f_ref_hz(omega, mtot) for omega in params_valid[:, 7]],
                dtype=np.float32,
            )
            f.create_dataset("omega0", data=params_valid[:, 7])
            f.create_dataset("f_ref_hz", data=f_ref_hz)

        # Metadata for reproducibility
        f.attrs["approximant"]           = config["data"]["approximant"]
        f.attrs["f_lower"]               = config["data"]["f_lower"]
        f.attrs["f_final"]               = config["data"]["f_final"]
        f.attrs["delta_f"]               = config["data"]["delta_f"]
        f.attrs["reference_total_mass"]  = config["data"]["reference_total_mass"]
        f.attrs["n_valid"]               = int(n_valid)
        if include_omega:
            f.attrs["param_names"]       = "q,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z,omega0"
            f.attrs["sample_reference_omega"] = True
            omega_pool = load_reference_omega_pool(config)
            if len(omega_pool):
                f.create_dataset("reference_omega_source_pool", data=omega_pool.astype(np.float32))
                f.attrs["reference_omega_source_count"] = int(len(omega_pool))
                f.attrs["reference_omega_sampling"] = str(
                    config["data"].get("reference_omega_sampling", "choice")
                )
                f.attrs["reference_omega_source_h5"] = ",".join(
                    str(path) for path in _as_list(config["data"].get("reference_omega_source_h5"))
                )
            f.attrs["reference_omega_range"] = np.asarray(
                config["data"].get("reference_omega_range", [0.0098, 0.0244]),
                dtype=np.float64,
            )
            f.attrs["allow_lower_generation_f_min"] = bool(
                config["data"].get("allow_lower_generation_f_min", True)
            )
            f.attrs["use_zero_generation_f_min"] = bool(
                config["data"].get("use_zero_generation_f_min", False)
            )
            f.attrs["generation_f_lower_floor"] = float(
                config["data"].get("generation_f_lower_floor", 0.0)
            )
        else:
            f.attrs["param_names"]       = "q,chi1x,chi1y,chi1z,chi2x,chi2y,chi2z"

    print(f"Done! Saved {n_valid} waveforms to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NRSur7dq4 training data")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--output",    default="training_data/waveforms_train.h5")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override the sample count from config (n_train/n_val/n_test)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--split",     choices=["train", "val", "test"], default="train")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    n_samples = args.n_samples if args.n_samples is not None else config["data"][f"n_{args.split}"]
    generate_dataset(config, n_samples, args.output, seed=args.seed)
