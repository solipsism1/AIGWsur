"""
Compute PSD-weighted match statistics between the trained flow and held-out waveforms.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.getcwd())

from src.data.dataset import WaveformDataset
from src.data.compression import WaveformCompressor
from src.models.flow import build_model
from src.utils.checkpoints import resolve_checkpoint_path, resolve_compressor_path
from src.utils.normalization import compute_normalization_stats, infer_train_data_path


def require_pycbc():
    """Import PyCBC lazily with a clear error if it is unavailable."""
    try:
        from pycbc.filter import match as pycbc_match
        from pycbc.psd import aLIGOZeroDetHighPower
        from pycbc.types import FrequencySeries
    except ImportError as exc:
        raise RuntimeError(
            "PyCBC is required for PSD-weighted match evaluation. "
            "Install it in WSL, Conda, or another compatible environment."
        ) from exc

    return FrequencySeries, pycbc_match, aLIGOZeroDetHighPower


def pad_to_zero_frequency(h_f: np.ndarray, freqs: np.ndarray, delta_f: float) -> np.ndarray:
    """Pad a frequency array whose first stored bin is above 0 Hz."""
    f_start = float(freqs[0])
    if f_start <= 0:
        return h_f
    n_pad = int(round(f_start / delta_f))
    return np.pad(h_f, (n_pad, 0), mode="constant")


def load_trained_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load a trained waveform model from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_state_dict = checkpoint["model_state_dict"]
    model = build_model(config, state_dict=model_state_dict).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    print(
        f"Loaded {checkpoint_path} "
        f"(epoch {checkpoint['epoch']}, val objective={checkpoint['val_loss']:.4f})"
    )
    return model


def compute_matches(
    model,
    test_ds: WaveformDataset,
    device: torch.device,
    n_test: int = 1000,
    deterministic: bool = True,
) -> dict:
    """Generate waveforms and compute PyCBC matches against held-out waveforms."""
    FrequencySeries, pycbc_match, aLIGOZeroDetHighPower = require_pycbc()

    freqs = test_ds.freqs
    delta_f = float(freqs[1] - freqs[0])
    f_lower = float(freqs[0])
    n_pad = int(round(f_lower / delta_f)) if f_lower > 0 else 0
    psd = aLIGOZeroDetHighPower(len(freqs) + n_pad, delta_f, f_lower)

    matches = []
    gen_times = []
    sample_params = []
    n_test = min(n_test, len(test_ds))

    import h5py as _h5py
    _h5f = _h5py.File(test_ds._waveform_h5_path, "r")

    for i in range(n_test):
        params_norm, _ = test_ds[i]
        params_tensor = params_norm.unsqueeze(0).to(device)

        t0 = time.perf_counter()
        with torch.no_grad():
            coeffs_std = model.generate_waveform(params_tensor, deterministic=deterministic)
        gen_times.append(time.perf_counter() - t0)

        real_gen, imag_gen = test_ds.decode_to_waveform(coeffs_std.cpu().numpy())
        real_true = _h5f["hp_real"][i]
        imag_true = _h5f["hp_imag"][i]

        h_true = pad_to_zero_frequency(
            (real_true + 1j * imag_true).astype(np.complex128),
            freqs,
            delta_f,
        )
        h_gen = pad_to_zero_frequency(
            (real_gen[0] + 1j * imag_gen[0]).astype(np.complex128),
            freqs,
            delta_f,
        )

        h1 = FrequencySeries(h_true, delta_f=delta_f)
        h2 = FrequencySeries(h_gen, delta_f=delta_f)
        match_value, _ = pycbc_match(h1, h2, psd=psd, low_frequency_cutoff=f_lower)
        matches.append(match_value)
        sample_params.append(test_ds.params[i])

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_test}: running mean match = {np.mean(matches):.6f}")

    _h5f.close()
    matches = np.array(matches)
    gen_times = np.array(gen_times)

    results = {
        "matches": matches,
        "parameters": np.asarray(sample_params, dtype=np.float32),
        "gen_times_ms": gen_times * 1000,
        "mean_match": float(matches.mean()),
        "median_match": float(np.median(matches)),
        "min_match": float(matches.min()),
        "frac_above_0.99": float((matches > 0.99).mean()),
        "frac_above_0.999": float((matches > 0.999).mean()),
        "mean_gen_time_ms": float(gen_times.mean() * 1000),
        "median_gen_time_ms": float(np.median(gen_times) * 1000),
    }

    print("\n=== Match results ===")
    print(f"  Mean match:    {results['mean_match']:.6f}")
    print(f"  Median match:  {results['median_match']:.6f}")
    print(f"  Min match:     {results['min_match']:.6f}")
    print(f"  > 0.99 :       {results['frac_above_0.99'] * 100:.1f}%")
    print(f"  > 0.999:       {results['frac_above_0.999'] * 100:.1f}%")
    print("\n=== Generation speed ===")
    print(f"  Mean:     {results['mean_gen_time_ms']:.2f} ms/waveform")
    print(f"  Throughput: {1000 / results['mean_gen_time_ms']:.0f} waveforms/s (single)")

    return results


def compute_batch_throughput(model, device: torch.device, param_dim: int, batch_sizes=None):
    """Benchmark batch-generation throughput at various batch sizes."""
    batch_sizes = batch_sizes or [1, 10, 100, 1000, 10000]

    print("\n=== Batch throughput ===")
    for batch_size in batch_sizes:
        params = torch.randn(batch_size, param_dim, device=device)

        with torch.no_grad():
            _ = model.generate_waveform(params, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        n_iters = max(1, 100 // batch_size)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                _ = model.generate_waveform(params, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - t0) / n_iters
        throughput = batch_size / elapsed
        print(f"  Batch {batch_size:>6d}: {elapsed * 1000:.2f} ms -> {throughput:.0f} waveforms/s")


def plot_results(results: dict, output_dir: str):
    """Save match and mismatch distribution histograms."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    matches = results["matches"]
    params = results.get("parameters")

    if params is not None and len(params) == len(matches):
        table = np.column_stack([np.arange(len(matches)), matches, params])
        np.savetxt(
            output_path / "matches.csv",
            table,
            delimiter=",",
            header=(
                "index,match,"
                + ",".join(
                    [
                        "q",
                        "chi1x",
                        "chi1y",
                        "chi1z",
                        "chi2x",
                        "chi2y",
                        "chi2z",
                    ][: params.shape[1]]
                    + (["omega0"] if params.shape[1] > 7 else [])
                )
            ),
            comments="",
        )

    summary = {
        key: value
        for key, value in results.items()
        if key not in {"matches", "parameters", "gen_times_ms"}
    }
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(matches, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0.99, color="red", linestyle="--", label="0.99 threshold")
    ax.axvline(0.999, color="orange", linestyle="--", label="0.999 threshold")
    ax.set_xlabel("Match")
    ax.set_ylabel("Count")
    ax.set_title(f"Match distribution (N={len(matches)}, mean={matches.mean():.5f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "match_histogram.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    mismatch = 1 - matches
    ax.hist(np.log10(mismatch + 1e-10), bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(np.log10(0.01), color="red", linestyle="--", label="match=0.99")
    ax.axvline(np.log10(0.001), color="orange", linestyle="--", label="match=0.999")
    ax.set_xlabel("log10(1 - match)")
    ax.set_ylabel("Count")
    ax.set_title("Mismatch distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "mismatch_histogram.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute match statistics for trained flow")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional legacy config path. The checkpoint's embedded config is used by default.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--test-data", default="training_data/waveforms_test.h5")
    parser.add_argument("--train-data", default=None)
    parser.add_argument("--compressor", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--stats-samples", type=int, default=None)
    parser.add_argument("--stats-chunk-size", type=int, default=512)
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic sampling instead of deterministic generation")
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    compressor_path = resolve_compressor_path(checkpoint_path, args.compressor)
    train_data_path = args.train_data or str(infer_train_data_path(args.test_data))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint["config"]

    if args.config:
        print(
            "Note: ignoring --config and using the checkpoint's embedded config "
            "to avoid architecture mismatches during evaluation."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_trained_model(str(checkpoint_path), config, device)
    compressor = WaveformCompressor.load(str(compressor_path))

    print("Computing training normalization statistics...")
    stats = compute_normalization_stats(
        train_data_path,
        compressor,
        n_samples=args.stats_samples,
        chunk_size=args.stats_chunk_size,
    )

    test_ds = WaveformDataset(args.test_data, compressor=compressor, n_samples=args.n_test)
    test_ds.apply_normalization_stats(
        stats["coeffs_mean"],
        stats["coeffs_std"],
        stats["param_min"],
        stats["param_max"],
    )

    results = compute_matches(
        model,
        test_ds,
        device,
        n_test=args.n_test,
        deterministic=not args.stochastic,
    )
    param_dim = int(config.get("model", {}).get("param_dim", config.get("data", {}).get("param_dim", 7)))
    compute_batch_throughput(model, device, param_dim)
    plot_results(results, args.output_dir)
