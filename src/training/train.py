"""
Training loop for the conditional normalizing flow.

Run from the project root:
    python src/training/train.py --config config.yaml

Supports:
  - Resuming from a checkpoint (--resume PATH)
  - Re-fitting the compressor (--force-refit-compressor)
  - Evaluation-only mode (--evaluate-only)
  - Single custom-waveform generation (--generate "q=2.0,s1z=0.5")
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

matplotlib.use("Agg")  # non-interactive backend (avoids Tk threading issues)

sys.path.append(os.getcwd())

from src.data.dataset import WaveformDataset
from src.models.flow import build_model
from src.training.optimization import (
    LinearWarmupController,
    build_scheduler,
    step_scheduler,
)
from src.utils.checkpoints import resolve_checkpoint_path
from src.utils.normalization import compute_normalization_stats


def atomic_save(state_dict, filepath):
    """Write a checkpoint atomically to prevent corruption on crash.

    On Windows, os.rename cannot overwrite an existing file, so we delete
    the old file BEFORE writing the temp, then rename. This means there is
    a brief window where neither file exists, but it prevents disk-full
    failures when writing the temp alongside the existing checkpoint.
    """
    tmp_path = str(filepath) + ".tmp"
    # Delete any stale temp from a previous crashed save
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    # Delete the old checkpoint first (Windows cannot rename over existing file)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass  # If locked, the subsequent rename will fail and caller catches it
    torch.save(state_dict, tmp_path, _use_new_zipfile_serialization=False)
    os.rename(tmp_path, filepath)


def _model_parameters_are_finite(model) -> bool:
    """Return False if any trainable parameter has become NaN/Inf."""
    return all(torch.isfinite(param).all().item() for param in model.parameters())


def _clone_model_parameters(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
    }


def _restore_model_parameters(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(state[name])


def _scale_optimizer_lr(optimizer, factor: float, min_lr: float):
    for group in optimizer.param_groups:
        group["lr"] = max(float(group["lr"]) * factor, min_lr)


def plot_loss_history(train_losses, val_losses, output_path, metric_name="Loss"):
    """Save a training / validation objective-curve plot."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label=f"Train {metric_name}")
    plt.plot(epochs, val_losses, label=f"Val {metric_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title("Training curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path / "plots" / "loss_history.png")
    plt.close()


def plot_samples(model, dataset, epoch, output_path, device, n_samples=3):
    """Generate deterministic sample waveforms and save comparison plots."""
    model.eval()
    indices = np.random.choice(len(dataset), n_samples, replace=False)

    fig, axes = plt.subplots(n_samples, 4, figsize=(24, 4 * n_samples))
    if n_samples == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for i, idx in enumerate(indices):
            params, coeffs_gt = dataset[idx]
            params = params.unsqueeze(0).to(device)

            coeffs_pred = model.generate_waveform(params, deterministic=True)
            real_gt, imag_gt = dataset.decode_to_waveform(coeffs_gt.unsqueeze(0).cpu().numpy())
            real_pred, imag_pred = dataset.decode_to_waveform(coeffs_pred.cpu().numpy())

            p_raw = dataset.unnormalize_params(params.cpu().numpy())
            title_str = f"q={p_raw[0, 0]:.2f}, chi1z={p_raw[0, 3]:.2f}, chi2z={p_raw[0, 6]:.2f}"

            freqs = dataset.freqs
            delta_f = freqs[1] - freqs[0]
            h_gt_f = real_gt[0] + 1j * imag_gt[0]
            h_pred_f = real_pred[0] + 1j * imag_pred[0]

            amp_gt = np.abs(h_gt_f)
            phase_gt = np.unwrap(np.angle(h_gt_f))
            amp_pred = np.abs(h_pred_f)
            phase_pred = np.unwrap(np.angle(h_pred_f))

            if freqs[0] > delta_f:
                n_pad = int(freqs[0] / delta_f)
                h_gt_f = np.pad(h_gt_f, (n_pad, 0), mode="constant")
                h_pred_f = np.pad(h_pred_f, (n_pad, 0), mode="constant")

            h_gt_t = np.fft.irfft(h_gt_f)
            h_pred_t = np.fft.irfft(h_pred_f)

            roll_shift = len(h_gt_t) // 2
            h_gt_t = np.roll(h_gt_t, roll_shift)
            h_pred_t = np.roll(h_pred_t, roll_shift)

            trim = 100
            gt_max = np.max(np.abs(h_gt_t[trim:-trim]))
            pr_max = np.max(np.abs(h_pred_t[trim:-trim]))
            h_gt_t /= (gt_max + 1e-30)
            h_pred_t /= (pr_max + 1e-30)

            ax = axes[i, 0]
            ax.loglog(freqs, amp_gt, label="GT", alpha=0.7)
            ax.loglog(freqs, amp_pred, "--", label="Pred", alpha=0.7)
            ax.set_title(f"Sample {idx} - Amp ({title_str})")
            ax.legend(fontsize="small")

            ax = axes[i, 1]
            ax.plot(freqs, phase_gt, label="GT", alpha=0.7)
            ax.plot(freqs, phase_pred, "--", label="Pred", alpha=0.7)
            ax.set_title(f"Sample {idx} - Phase")

            ax = axes[i, 2]
            ax.plot(h_gt_t, label="GT", alpha=0.7)
            ax.plot(h_pred_t, "--", label="Pred", alpha=0.7)
            ax.set_title(f"Sample {idx} - Time (full)")
            ax.set_ylim(-1.1, 1.1)

            ax = axes[i, 3]
            ax.plot(h_gt_t, label="GT", alpha=0.7)
            ax.plot(h_pred_t, "--", label="Pred", alpha=0.7)
            ax.set_title(f"Sample {idx} - Time (zoomed)")
            ax.set_xlim(roll_shift - 1000, roll_shift + 500)
            ax.set_ylim(-1.1, 1.1)

    save_path = output_path / "plots" / f"samples_epoch_{epoch}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"  Saved sample plot -> {save_path.name}")
    plt.close()


def generate_custom(model, dataset, params_str, output_path, device):
    """Generate and plot a deterministic waveform for a parameter string."""
    print(f"Generating custom waveform: {params_str}")
    model.eval()

    target_params = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    try:
        if params_str:
            mapping = {
                "q": 0,
                "s1x": 1,
                "s1y": 2,
                "s1z": 3,
                "s2x": 4,
                "s2y": 5,
                "s2z": 6,
            }
            for pair in params_str.split(","):
                key, val = pair.split("=")
                if key in mapping:
                    target_params[mapping[key]] = float(val)
    except Exception as exc:
        print(f"Error parsing parameters: {exc}")
        return

    p_min = dataset.param_min
    p_max = dataset.param_max
    norm_params = 2.0 * (target_params - p_min) / (p_max - p_min + 1e-10) - 1.0
    params_tensor = torch.from_numpy(norm_params.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        coeffs_pred = model.generate_waveform(params_tensor, deterministic=True)

    real_pred, imag_pred = dataset.decode_to_waveform(coeffs_pred.cpu().numpy())

    freqs = dataset.freqs
    delta_f = freqs[1] - freqs[0]
    h_pred_f = real_pred[0] + 1j * imag_pred[0]
    amp_pred = np.abs(h_pred_f)
    phase_pred = np.unwrap(np.angle(h_pred_f))

    if freqs[0] > delta_f:
        n_pad = int(freqs[0] / delta_f)
        h_pred_f = np.pad(h_pred_f, (n_pad, 0), mode="constant")

    h_pred_t = np.fft.irfft(h_pred_f)
    h_pred_t = np.roll(h_pred_t, len(h_pred_t) // 2)
    max_amp = np.max(np.abs(h_pred_t))
    print(f"Peak strain amplitude: {max_amp:.4e}")
    h_pred_t /= (max_amp + 1e-30)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].loglog(freqs, amp_pred)
    axes[0].set_title(f"Amplitude (q={target_params[0]:.2f})")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[1].plot(freqs, phase_pred)
    axes[1].set_title("Phase")
    axes[1].grid(True)
    axes[2].plot(h_pred_t)
    axes[2].set_title("Time domain (normalized)")
    axes[2].grid(True)

    safe_str = params_str.replace("=", "").replace(",", "_") if params_str else "default"
    save_path = output_path / "plots" / f"generated_{safe_str}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved custom generation -> {save_path}")


def _scan_existing_best(output_path: Path, current_best: float) -> tuple[float, int]:
    """Find the best validation loss already present on disk."""
    best_val_loss = current_best
    best_epoch = 0
    for path in output_path.glob("best_model_epoch*_loss*.pt"):
        try:
            stem = path.stem
            epoch = int(stem.split("_loss")[0].split("epoch")[-1])
            val = float(stem.split("loss")[-1])
            if val < best_val_loss:
                best_val_loss = val
                best_epoch = epoch
                print(f"  Found existing best: {path.name} (val objective={best_val_loss:.4f})")
        except Exception:
            pass
    return best_val_loss, best_epoch


def _split_limit(config: dict, key: str):
    """Return an optional split limit from config.data."""
    value = config["data"].get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == "all":
        return None
    return int(value)


def train_loop(
    model,
    optimizer,
    warmup,
    scheduler,
    scheduler_name,
    train_loader,
    val_loader,
    val_ds,
    config,
    output_path,
    device,
    start_epoch,
    best_val_loss,
    all_train_losses,
    all_val_losses,
    patience_counter=0,
    best_epoch=0,
):
    """Main epoch-level training loop."""
    train_cfg = config["training"]
    metric_name = getattr(model, "metric_name", "Loss")
    early_stopping_patience = max(0, int(train_cfg.get("patience", 0)))
    tb_writer = SummaryWriter(log_dir=str(output_path / "tb_logs"))
    early_stopping_min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    min_epochs = max(0, int(train_cfg.get("min_epochs", 0)))
    rollback_nonfinite_step = bool(train_cfg.get("rollback_nonfinite_step", False))
    bad_step_lr_factor = float(train_cfg.get("bad_step_lr_factor", 0.5))
    bad_step_min_lr = float(train_cfg.get("bad_step_min_lr", train_cfg.get("scheduler_min_lr", 0.0)))

    for epoch in range(start_epoch, train_cfg["max_epochs"] + 1):
        model.train()
        train_losses = []
        bad_batches = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, mininterval=1.0)
        for params, coeffs in pbar:
            params = params.to(device)
            coeffs = coeffs.to(device)
            param_noise_std = float(train_cfg.get("param_noise_std", 0.0))
            if param_noise_std > 0:
                params = torch.clamp(
                    params + torch.randn_like(params) * param_noise_std,
                    -1.0,
                    1.0,
                )

            loss = model.training_loss(params, coeffs)
            if not torch.isfinite(loss):
                bad_batches += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({metric_name: "nonfinite"})
                continue

            optimizer.zero_grad()
            loss.backward()
            if train_cfg["grad_clip"] > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            else:
                grads = [p.grad.detach().flatten() for p in model.parameters() if p.grad is not None]
                grad_norm = torch.linalg.vector_norm(torch.cat(grads)) if grads else torch.zeros((), device=device)
            if not torch.isfinite(grad_norm):
                bad_batches += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({metric_name: "bad_grad"})
                continue
            pre_step_state = _clone_model_parameters(model) if rollback_nonfinite_step else None
            optimizer.step()
            if rollback_nonfinite_step and not _model_parameters_are_finite(model):
                bad_batches += 1
                _restore_model_parameters(model, pre_step_state)
                optimizer.state.clear()
                _scale_optimizer_lr(optimizer, bad_step_lr_factor, bad_step_min_lr)
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({metric_name: "bad_step"})
                continue
            warmup.step()

            train_losses.append(loss.item())
            pbar.set_postfix({metric_name: f"{loss.item():.3f}"})

        if not train_losses:
            print(f"Epoch {epoch}: no finite training losses; stopping.")
            break

        avg_train = float(np.mean(train_losses))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for params, coeffs in val_loader:
                params = params.to(device)
                coeffs = coeffs.to(device)
                val_loss = model.training_loss(params, coeffs)
                if torch.isfinite(val_loss):
                    val_losses.append(val_loss.item())

        if not val_losses:
            print(f"Epoch {epoch}: no finite validation losses; stopping.")
            break

        avg_val = float(np.mean(val_losses))
        lr_current = optimizer.param_groups[0]["lr"]

        all_train_losses.append(avg_train)
        all_val_losses.append(avg_val)
        print(
            f"Epoch {epoch}: train={avg_train:.4f}, val={avg_val:.4f}, "
            f"lr={lr_current:.2e}"
            + (f", skipped_nonfinite={bad_batches}" if bad_batches else "")
        )

        tb_writer.add_scalar(f"{metric_name}/train", avg_train, epoch)
        tb_writer.add_scalar(f"{metric_name}/val", avg_val, epoch)
        tb_writer.add_scalar("LR", lr_current, epoch)

        plot_loss_history(all_train_losses, all_val_losses, output_path, metric_name=metric_name)

        if scheduler is not None and not warmup.is_active():
            step_scheduler(scheduler, scheduler_name, metric=avg_val)

        improved = avg_val < (best_val_loss - early_stopping_min_delta)
        if improved:
            best_val_loss = avg_val
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "warmup_state_dict": warmup.state_dict(),
            "val_loss": avg_val,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "patience_counter": patience_counter,
            "config": config,
            "train_loss_history": all_train_losses,
            "val_loss_history": all_val_losses,
        }

        if improved:
            target_path = output_path / f"best_model_epoch{epoch}_loss{avg_val:.4f}.pt"
            alias_path = output_path / "best_model.pt"
            keep_best_history = bool(train_cfg.get("keep_best_history", False))

            best_history_pattern = "best_model_epoch*_loss*.pt"
            for path in output_path.glob(best_history_pattern):
                try:
                    os.remove(path)
                except Exception as exc:
                    print(f"  Warning: could not remove {path.name}: {exc}")

            try:
                atomic_save(checkpoint_data, alias_path)
                if keep_best_history:
                    atomic_save(checkpoint_data, target_path)
                    print(f"  -> Best model saved: {alias_path.name} ({target_path.name})")
                else:
                    print(f"  -> Best model saved: {alias_path.name}")
            except Exception as exc:
                print(f"  Warning: failed to save best model: {exc}")

        checkpoint_interval = int(train_cfg.get("keep_checkpoint_interval", 0))
        keep_periodic_checkpoint = checkpoint_interval > 0 and epoch % checkpoint_interval == 0
        current_checkpoint = output_path / f"checkpoint_epoch{epoch}_loss{avg_val:.4f}.pt"
        periodic_checkpoint = output_path / f"periodic_epoch{epoch}_loss{avg_val:.4f}.pt"
        for path in output_path.glob("checkpoint_epoch*_loss*.pt"):
            try:
                os.remove(path)
            except OSError:
                pass
        for path in output_path.glob("*.tmp"):
            try:
                os.remove(path)
            except OSError:
                pass
        try:
            atomic_save(checkpoint_data, current_checkpoint)
            if keep_periodic_checkpoint:
                atomic_save(checkpoint_data, periodic_checkpoint)
        except Exception as exc:
            print(f"  Warning: failed to save rolling checkpoint: {exc}")

        # cosine_restart_best: every restart_interval epochs, reload best model weights
        # and rebuild the scheduler so LR resets to the configured peak.
        _restart_interval = int(train_cfg.get("restart_interval", 0))
        if (scheduler_name == "cosine_restart_best"
                and _restart_interval > 0
                and epoch > 0
                and epoch % _restart_interval == 0):
            _best_path = output_path / "best_model.pt"
            if _best_path.exists():
                _best_ckpt = torch.load(str(_best_path), map_location=device)
                model.load_state_dict(_best_ckpt["model_state_dict"])
                print(
                    f"  Cosine restart at epoch {epoch}: reloaded best model "
                    f"(val {_best_ckpt['best_val_loss']:.4f}), "
                    f"resetting LR to {float(train_cfg['lr']):.2e}"
                )
            for pg in optimizer.param_groups:
                pg["lr"] = float(train_cfg["lr"])
            scheduler, _ = build_scheduler(optimizer, config["training"])
            patience_counter = 0

        plot_interval = int(train_cfg.get("plot_interval", 10))
        if plot_interval > 0 and epoch % plot_interval == 0:
            try:
                plot_samples(model, val_ds, epoch, output_path, device)
            except Exception as exc:
                print(f"  Warning: plot generation failed: {exc}")

        if (
            early_stopping_patience > 0
            and epoch >= min_epochs
            and patience_counter >= early_stopping_patience
        ):
            print(
                "Early stopping triggered "
                f"after {patience_counter} stale epochs. "
                f"Best epoch: {best_epoch}, best val {metric_name}: {best_val_loss:.4f}"
            )
            break

    tb_writer.close()
    print(f"Training complete. Best val {metric_name}: {best_val_loss:.4f} (epoch {best_epoch})")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data-dir", default="training_data")
    parser.add_argument("--train-data", default=None, help="Optional explicit training HDF5 path")
    parser.add_argument("--val-data", default=None, help="Optional explicit validation HDF5 path")
    parser.add_argument(
        "--stats-data",
        default=None,
        help="Optional HDF5 path used for coefficient/parameter normalization stats",
    )
    parser.add_argument("--output-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--force-refit-compressor",
        action="store_true",
        help="Refit compressor.h5 even if one already exists in the output directory",
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help="Generate evaluation plots and exit",
    )
    parser.add_argument(
        "--generate",
        type=str,
        default=None,
        help="Generate a single waveform, e.g. 'q=2.0,s1z=0.5'",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)

    data_dir = config["data"].get("data_dir", "training_data")
    if not os.path.exists(data_dir) and os.path.exists("training_data"):
        data_dir = "training_data"

    train_path = Path(args.train_data) if args.train_data else Path(data_dir) / "waveforms_train.h5"
    val_path = Path(args.val_data) if args.val_data else Path(data_dir) / "waveforms_val.h5"

    compression_cfg = config["compression"]
    compressor_path = output_path / "compressor.h5"
    if compressor_path.exists() and not args.force_refit_compressor:
        print(f"Loading pre-fitted compressor from {compressor_path}...")
        train_ds = WaveformDataset(
            str(train_path),
            compressor_path=str(compressor_path),
            n_samples=_split_limit(config, "n_train"),
        )
    else:
        if compressor_path.exists() and args.force_refit_compressor:
            print(f"Refitting compressor and overwriting {compressor_path}...")
        print("Fitting new SVD compressor...")
        train_ds = WaveformDataset(
            str(train_path),
            fit_compressor=True,
            n_basis=compression_cfg["n_basis"],
            n_samples=_split_limit(config, "n_train"),
            compressor_fit_n_samples=compression_cfg.get("fit_n_samples"),
            compressor_fit_strategy=compression_cfg.get("fit_strategy", "all"),
            compressor_fit_seed=compression_cfg.get("fit_seed", 42),
            compressor_svd_solver=compression_cfg.get("svd_solver", "auto"),
            compressor_representation=compression_cfg.get("representation", "amp_phase"),
            compressor_greedy_extra_basis=compression_cfg.get("greedy_extra_basis", 0),
        )
        train_ds.compressor.save(compressor_path)
        print(f"Compressor saved to {compressor_path}")

    if args.stats_data:
        print(f"Applying normalization statistics from {args.stats_data}...")
        stats = compute_normalization_stats(
            args.stats_data,
            train_ds.compressor,
            n_samples=_split_limit(config, "n_train"),
        )
        train_ds.apply_normalization_stats(
            stats["coeffs_mean"],
            stats["coeffs_std"],
            stats["param_min"],
            stats["param_max"],
        )

    print("Initializing validation dataset...")
    val_ds = WaveformDataset(
        str(val_path),
        compressor=train_ds.compressor,
        n_samples=_split_limit(config, "n_val"),
    )

    val_ds.apply_normalization_stats(
        train_ds.coeffs_mean,
        train_ds.coeffs_std,
        train_ds.param_min,
        train_ds.param_max,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    resume_state_dict_for_arch = None
    if args.resume and os.path.exists(args.resume):
        resume_state_dict_for_arch = torch.load(args.resume, map_location="cpu").get("model_state_dict")

    model = build_model(config, state_dict=resume_state_dict_for_arch).to(device)
    if hasattr(model, "configure_waveform_loss"):
        model.configure_waveform_loss(
            train_ds.compressor,
            train_ds.coeffs_mean,
            train_ds.coeffs_std,
            config["model"],
        )
    elif hasattr(model, "coeffs_std"):
        model.coeffs_std.copy_(torch.from_numpy(train_ds.coeffs_std).float().to(device))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    warmup = LinearWarmupController(
        optimizer,
        warmup_steps=config["training"].get("warmup_steps", 0),
    )
    scheduler, scheduler_name = build_scheduler(optimizer, config["training"])
    print(
        "LR controls: "
        f"warmup_steps={warmup.warmup_steps}, scheduler={scheduler_name}"
    )

    start_epoch = 1
    best_val_loss = float("inf")
    all_train_losses = []
    all_val_losses = []
    patience_counter = 0
    best_epoch = 0

    if args.evaluate_only and not args.resume:
        try:
            args.resume = str(resolve_checkpoint_path(search_dir=output_path))
        except FileNotFoundError:
            args.resume = None

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if bool(config["training"].get("reset_optimizer_on_resume", False)):
            warmup.sync_optimizer()
            print("Loaded model weights only; optimizer/scheduler/warmup were reset from this config.")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                # Re-apply config patience: state_dict restores all attributes including
                # patience, which overwrites the value we set from config. Fix it back.
                if hasattr(scheduler, "patience"):
                    scheduler.patience = max(1, int(train_cfg.get("scheduler_patience", 10)))
            warmup.load_state_dict(ckpt.get("warmup_state_dict"))
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
            all_train_losses = ckpt.get("train_loss_history", [])
            all_val_losses = ckpt.get("val_loss_history", [])
            patience_counter = ckpt.get("patience_counter", 0)
            best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
            print(f"Loaded epoch {start_epoch - 1}, best validation objective: {best_val_loss:.4f}")
        if hasattr(model, "configure_waveform_loss"):
            model.configure_waveform_loss(
                train_ds.compressor,
                train_ds.coeffs_mean,
                train_ds.coeffs_std,
                config["model"],
            )
    else:
        warmup.sync_optimizer()

    disk_best, disk_best_epoch = _scan_existing_best(output_path, best_val_loss)
    if disk_best < best_val_loss:
        best_val_loss = disk_best
        best_epoch = disk_best_epoch

    if args.evaluate_only:
        print("\n--- Evaluation mode ---")
        model.eval()
        plot_samples(model, val_ds, "EVAL", output_path, device, n_samples=3)
        print("Saved to checkpoints/plots/samples_epoch_EVAL.png")
        return

    if args.generate:
        print(f"\n--- Custom generation: {args.generate} ---")
        generate_custom(model, val_ds, args.generate, output_path, device)
        return

    train_loop(
        model,
        optimizer,
        warmup,
        scheduler,
        scheduler_name,
        train_loader,
        val_loader,
        val_ds,
        config,
        output_path,
        device,
        start_epoch,
        best_val_loss,
        all_train_losses,
        all_val_losses,
        patience_counter=patience_counter,
        best_epoch=best_epoch,
    )


if __name__ == "__main__":
    main()
