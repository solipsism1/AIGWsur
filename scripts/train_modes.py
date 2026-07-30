"""Train the inertial-frame mode regressor on per-mode SVD coefficients.

Reuses the generic epoch loop in src/training/train.py (train_loop). Plotting is
disabled (plot_interval=0) since mode -> waveform decoding needs an orientation.

    python scripts/train_modes.py --config config.modes_v1.yaml \
        --train-data D:/gw_waveform_data/modes_ds_smoke.h5 --val-frac 0.15 \
        --output-dir runs/modes_smoke
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from src.data.mode_dataset import ModeCoeffDataset
from src.models.flow import build_model
from src.training.optimization import LinearWarmupController, build_scheduler
from src.training.train import train_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.modes_v1.yaml")
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--output-dir", default="runs/modes_smoke")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--loss-weighting", default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    if args.max_epochs is not None:
        config["training"]["max_epochs"] = args.max_epochs
    if args.loss_weighting is not None:
        config["model"]["loss_weighting"] = args.loss_weighting
    if args.dropout is not None:
        config["model"]["dropout"] = args.dropout
    if args.lr is not None:
        config["training"]["lr"] = args.lr
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out = Path(args.output_dir); (out / "plots").mkdir(parents=True, exist_ok=True)

    if args.val_data:
        train_ds = ModeCoeffDataset(args.train_data)
        val_ds = ModeCoeffDataset(args.val_data)
    else:
        with __import__("h5py").File(args.train_data, "r") as f:
            n = f["coeffs"].shape[0]
        n_val = int(args.val_frac * n)
        train_ds = ModeCoeffDataset(args.train_data, index_range=(0, n - n_val))
        val_ds = ModeCoeffDataset(args.train_data, index_range=(n - n_val, n))
    val_ds.apply_normalization_stats(train_ds.coeffs_mean, train_ds.coeffs_std,
                                     train_ds.param_min, train_ds.param_max)
    print(f"train={len(train_ds)} val={len(val_ds)} coeff_dim={train_ds.coeff_dim}")
    # Persist normalization stats for evaluation (model predicts standardized coeffs).
    np.savez(out / "norm_stats.npz", coeffs_mean=train_ds.coeffs_mean,
             coeffs_std=train_ds.coeffs_std, param_min=train_ds.param_min,
             param_max=train_ds.param_max)

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["training"]["batch_size"],
                            shuffle=False, num_workers=0, pin_memory=True)

    model = build_model(config).to(device)
    model.configure_waveform_loss(None, train_ds.coeffs_mean, train_ds.coeffs_std, config["model"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    warmup = LinearWarmupController(optimizer, warmup_steps=config["training"].get("warmup_steps", 0))
    scheduler, scheduler_name = build_scheduler(optimizer, config["training"])

    start_epoch, best = 1, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model_state_dict"]); warmup.sync_optimizer()
        start_epoch = ck["epoch"] + 1; best = ck.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch-1}, best {best:.4f}")
    else:
        warmup.sync_optimizer()

    train_loop(model, optimizer, warmup, scheduler, scheduler_name,
               train_loader, val_loader, val_ds, config, out, device,
               start_epoch, best, [], [], patience_counter=0, best_epoch=0)


if __name__ == "__main__":
    main()
