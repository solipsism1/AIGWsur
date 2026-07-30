"""M5 stage 2: match-based fine-tuning of the mode regressor.

Because the projection is differentiable, we backprop a PSD-weighted mismatch directly.
For each batch we sample random orientations (iota, phi), project both the predicted
and the true (dataset) coefficients to h_+(f), and minimize 1 - phase-maximized overlap.
This targets the reported metric instead of the coefficient-space Huber surrogate.

Target uses the SVD-compressed true coeffs (the achievable ceiling, ~0.9957), so this
pushes the network toward that ceiling at arbitrary orientation.

    python scripts/finetune_modes_match.py --config config.modes_v1.yaml \
        --train-data D:/gw_waveform_data/modes_ds_250k.h5 --init runs/modes_m5/best_model.pt \
        --output-dir runs/modes_m5_ft --epochs 60
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
from src.data.mode_compression import PerModeCompressor, resolve_data_path
from src.models.flow import build_model
from src.physics.projection import TorchModeDecoder, TorchStrainModel


def analytic_aligo_psd(freqs: np.ndarray) -> np.ndarray:
    f = np.asarray(freqs, dtype=np.float64)
    x = f / 215.0
    s0 = 1.0e-49
    return s0 * (284.9 * x ** -4.8 + 17.11 * x ** -1.7
                 + x ** 3 * ((2.36 * x ** 2 - 3.84 * x + 1.25) / (x ** 2 + x + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.modes_v1.yaml")
    ap.add_argument("--train-data", required=True)
    ap.add_argument("--init", required=True, help="coeff-trained checkpoint to start from")
    ap.add_argument("--stats", default=None, help="norm_stats.npz (defaults next to --init)")
    ap.add_argument("--output-dir", default="runs/modes_m5_ft")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--orient-per-sample", type=int, default=4)
    ap.add_argument("--norm-weight", type=float, default=0.5,
                    help="weight on the log-norm anchor; the match loss is scale-free, so "
                         "without this the overall amplitude drifts (was ~300x off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--psd-weights", default="data/psd_weights_band.npz",
                    help="precomputed real-PSD band weights (.npz); falls back to analytic")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    init = Path(args.init)
    stats_path = args.stats or (init.parent / "norm_stats.npz")
    stats = np.load(stats_path)

    import h5py
    with h5py.File(resolve_data_path(args.train_data), "r") as f:
        n = f["coeffs"].shape[0]
    n_val = int(args.val_frac * n)
    train_ds = ModeCoeffDataset(args.train_data, index_range=(0, n - n_val))
    val_ds = ModeCoeffDataset(args.train_data, index_range=(n - n_val, n))
    for ds in (train_ds, val_ds):
        ds.apply_normalization_stats(stats["coeffs_mean"], stats["coeffs_std"],
                                     stats["param_min"], stats["param_max"])
    print(f"train={len(train_ds)} val={len(val_ds)}")

    net = build_model(config).to(device)
    ck = torch.load(init, map_location=device)
    net.load_state_dict(ck["model_state_dict"])
    comp = PerModeCompressor.load(config["compression"]["compressor_path"])
    dec = TorchModeDecoder(comp, stats["coeffs_mean"], stats["coeffs_std"]).to(device)
    grid_n = comp.svd[comp.keys[0]][0].mean.shape[0]
    tsm = TorchStrainModel(net, dec, comp.keys, config["data"]["sample_rate"],
                           config["data"]["f_lower"], config["data"]["f_final"], grid_n).to(device)

    # PSD weights: prefer the precomputed REAL aLIGO weights (the analytic approx weights
    # the wrong frequencies and inflates the internal match vs the eval match). Normalize
    # to max 1 in float64 before casting (raw 1/PSD ~1e42 overflows float32).
    wpath = resolve_data_path(args.psd_weights)
    if os.path.exists(wpath):
        wz = np.load(wpath)
        w_np = wz["weights"].astype(np.float64)
        assert len(w_np) == tsm.freqs.shape[0], f"psd-weights len {len(w_np)} != band {tsm.freqs.shape[0]}"
        print(f"using real PSD weights from {wpath}")
    else:
        psd_np = analytic_aligo_psd(tsm.freqs.cpu().numpy()).astype(np.float64)
        w_np = np.where(np.isfinite(psd_np) & (psd_np > 0), 1.0 / psd_np, 0.0)
        print("using analytic PSD weights (real weights file not found)")
    w_np = w_np / w_np.max()
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    STRAIN_SCALE = 1.0e21  # lift ~1e-21 strain to O(1) so squares don't underflow float32

    def match_loss(params_norm, true_std):
        B = params_norm.shape[0]
        iota = torch.arccos(torch.empty(B, device=device).uniform_(-1, 1))
        phi = torch.empty(B, device=device).uniform_(0, 2 * np.pi)
        pred = tsm.strain_fd(params_norm, iota, phi) * STRAIN_SCALE
        with torch.no_grad():
            targ = tsm.strain_from_std_coeffs(true_std, iota, phi) * STRAIN_SCALE
        pred_n2 = torch.sum(torch.abs(pred) ** 2 * w, dim=1).clamp_min(1e-30)
        targ_n2 = torch.sum(torch.abs(targ) ** 2 * w, dim=1).clamp_min(1e-30)
        num = torch.abs(torch.sum(pred * torch.conj(targ) * w, dim=1))
        match = num / torch.sqrt(pred_n2 * targ_n2)
        # The match is scale-invariant in pred, so anchor the amplitude with a log-norm term,
        # else the overall strain scale drifts (observed ~300x) without hurting the match.
        norm_term = (0.5 * torch.log(pred_n2) - 0.5 * torch.log(targ_n2)) ** 2
        return torch.mean(1.0 - match) + args.norm_weight * torch.mean(norm_term)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        net.train(); tl = []
        for p, c in train_loader:
            p, c = p.to(device), c.to(device)
            for _ in range(args.orient_per_sample):
                opt.zero_grad()
                loss = match_loss(p, c)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); tl.append(loss.item())
        sched.step()
        net.eval(); vl = []
        with torch.no_grad():
            for p, c in val_loader:
                p, c = p.to(device), c.to(device)
                vl.append(match_loss(p, c).item())
        tm, vm = float(np.mean(tl)), float(np.mean(vl))
        print(f"Epoch {ep}: train mismatch={tm:.5f} (match {1-tm:.5f}) | "
              f"val mismatch={vm:.5f} (match {1-vm:.5f}) lr={opt.param_groups[0]['lr']:.2e}")
        if vm < best:
            best = vm
            torch.save({"model_state_dict": net.state_dict(), "epoch": ep,
                        "config": config, "val_mismatch": vm}, out / "best_model.pt")
            np.savez(out / "norm_stats.npz", **{k: stats[k] for k in stats.files})
    print(f"Done. Best val match: {1-best:.5f}")


if __name__ == "__main__":
    main()
