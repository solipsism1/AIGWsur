# AIGWsur — Differentiable Neural-Network Surrogate for NRSur7dq4

A fast, fully differentiable neural-network surrogate for the
[NRSur7dq4](https://arxiv.org/abs/1905.09300) precessing binary black-hole
waveform model. The network predicts all 21 inertial-frame spherical-harmonic
modes h_lm (l ≤ 4) over the 8-dimensional intrinsic parameter space
`λ = (q, χ₁ₓ, χ₁ᵧ, χ₁z, χ₂ₓ, χ₂ᵧ, χ₂z, ω₀)`; polarizations at any orientation
(ι, φ) follow from an analytic, differentiable spin-weighted spherical-harmonic
projection.

[![arXiv](https://img.shields.io/badge/arXiv-2608.09978-b31b1b.svg)](https://arxiv.org/abs/2608.09978)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Paper:** [_A fast, differentiable neural-network surrogate for precessing
binary black-hole waveforms_](https://arxiv.org/abs/2608.09978), B. Modrekiladze (2026), arXiv:2608.09978.
The `paper/` directory contains the source of the submitted version.

This is step one of a two-step program. Matched filtering only finds signals
that the template bank already contains ("we will only see what we expect to
see"). A fast, differentiable surrogate frees the computational budget for step
two: learning the distribution of gravitational-wave signals from data alone,
without anchoring to a theoretical template (companion paper in preparation).

---

## Highlights

| Property | Value |
|---|---|
| Outputs | all 21 h_lm modes (l ≤ 4), both polarizations, any orientation |
| Architecture | shared RFF + ResNet trunk, per-mode linear heads (5.1 M params) |
| Compression | per-mode SVD, 3548 coefficients total |
| Training data | 6×10⁵ NRSur7dq4 waveforms (Sobol; q∈[1,4], \|χ\|≤0.8) |
| Fixed-orientation match | mean 0.975, median 0.988 |
| Orientation-averaged match | mean 0.940, median 0.975 |
| Strain amplitude ratio | median 0.98 (physical, no scale drift) |
| Speed | ~12 ms single waveform; ~3.5×10⁴ wf/s batched (~430× faster than direct evaluation) |
| Differentiable | yes — full 13-parameter Fisher matrix and NUTS/HMC sampling |

Trained weights (`runs/modes_m5_600k_ft/`) and the PSD weights used for match
fine-tuning and PE (`data/psd_weights_band.npz`) are included in this
repository. The per-mode SVD basis (66 MB) is attached to the
[v1.0 release](https://github.com/solipsism1/AIGWsur/releases/tag/v1.0) —
download it once before using the model:

```bash
curl -L -o data/mode_compressor_v1.h5 \
  https://github.com/solipsism1/AIGWsur/releases/download/v1.0/mode_compressor_v1.h5
```

---

## Repository layout

```
├── src/
│   ├── data/
│   │   ├── mode_waveforms.py     # TD mode generation + conditioning (PyCBC)
│   │   ├── mode_compression.py   # Per-mode SVD compressor
│   │   ├── mode_dataset.py       # Coefficient dataset
│   │   └── generate_waveforms.py # Legacy single-polarization data generation
│   ├── models/flow.py            # ModeResNetRegressor + build_model()
│   ├── physics/
│   │   ├── swsh.py               # Differentiable spin-weighted spherical harmonics
│   │   └── projection.py         # Differentiable coeffs → h₊(f), strain model
│   ├── training/                 # Training loop, schedulers
│   └── evaluation/               # PSD-weighted match evaluation
├── scripts/
│   ├── generate_mode_dataset.py  # Data generation (PyCBC/WSL env)
│   ├── train_modes.py            # Coefficient-space (Huber) pretraining
│   ├── finetune_modes_match.py   # PSD-weighted match fine-tuning (norm-anchored)
│   ├── eval_modes.py             # Match statistics + figures
│   ├── fisher_modes.py           # 13-parameter Fisher matrix via autodiff
│   ├── bench_modes_speed.py      # Latency / throughput benchmark
│   └── pe_demo/                  # NUTS parameter-estimation demo (paper Appendix A)
├── paper/                        # Paper source as submitted to arXiv
├── runs/modes_m5_600k_ft/        # Trained weights + eval artifacts
├── data/                         # SVD compressor + PSD weights
└── config.modes_v1.yaml          # Production config
```

---

## Setup

```bash
pip install -r requirements.txt
```

Two environments are involved:

- **PyTorch only** — inference, training, Fisher, PE demo. Any OS.
- **PyCBC + LALSuite** — data generation and match evaluation
  (`environment.pycbc-wsl.yml`; on Windows use WSL).

---

## Quick start: generate waveforms with the pretrained model

```python
import torch, yaml
from src.physics.projection import load_torch_strain_model

cfg = yaml.safe_load(open("config.modes_v1.yaml"))
tsm, stats = load_torch_strain_model("runs/modes_m5_600k_ft", cfg)

# params_norm: (batch, 8) intrinsic parameters normalized to [-1, 1]
#   raw λ = (q, χ1x, χ1y, χ1z, χ2x, χ2y, χ2z, ω0)
pmin, pmax = torch.tensor(stats["param_min"]), torch.tensor(stats["param_max"])
params_norm = 2.0 * (params_raw - pmin) / (pmax - pmin + 1e-10) - 1.0

iota = torch.tensor([0.5]); phi = torch.tensor([0.0])
h_fd = tsm.strain_fd(params_norm, iota, phi)   # differentiable h₊(f)
```

Everything is differentiable end-to-end: gradients flow through the network,
the SVD decoder, and the SWSH projection, so `torch.func.jacfwd` /
`jacrev` give waveform derivatives for Fisher forecasts or gradient-based
samplers directly.

---

## Reproducing the paper pipeline

1. **Generate training data** (PyCBC env; the shipped SVD basis
   `data/mode_compressor_v1.h5` compresses each waveform on the fly):
   ```bash
   python scripts/generate_mode_dataset.py \
       --compressor data/mode_compressor_v1.h5 \
       --n-valid 600000 --out data/modes_ds.h5
   ```
   (To refit the SVD basis from scratch instead, see
   `scripts/generate_mode_cache.py`, `scripts/size_modes_from_cache.py`,
   and `scripts/fit_mode_compressors.py`.)
2. **Pretrain on coefficients** (Huber loss):
   ```bash
   python scripts/train_modes.py --config config.modes_v1.yaml \
       --train-data data/modes_ds.h5 --output-dir runs/my_run
   ```
3. **Match fine-tune** (PSD-weighted, norm-anchored):
   ```bash
   python scripts/finetune_modes_match.py --train-data data/modes_ds.h5 \
       --init runs/my_run/best_model.pt --output-dir runs/my_run_ft \
       --norm-weight 1.0
   ```
4. **Evaluate** (PyCBC env):
   ```bash
   python scripts/eval_modes.py --run runs/modes_m5_600k_ft \
       --config config.modes_v1.yaml --n-test 150 --seed 999
   ```
   The shipped `runs/modes_m5_600k_ft/eval/` contains the paper's evaluation
   arrays and summary (seed 999, n=150).

---

## Fisher matrix

```bash
python scripts/fisher_modes.py --run runs/modes_m5_600k_ft --config config.modes_v1.yaml
```

Autodiff derivatives validated against finite differences (overlap ≥ 0.9999
for all 13 parameters).

---

## Parameter-estimation demo (paper Appendix A)

Cross-injection test: a true NRSur7dq4 signal (`scripts/pe_demo/injection_true_modes.npz`)
recovered with the surrogate likelihood via Pyro NUTS (zero noise, SNR 25):

```bash
python scripts/pe_demo/run_nuts.py      # ~16 min CPU for 200 samples
python scripts/pe_demo/plot_corner.py   # → paper/pe_corner.png
```

---

## Citation

```bibtex
@article{Modrekiladze:2026sur,
  author        = {Modrekiladze, Beka},
  title         = {A fast, differentiable neural-network surrogate for precessing binary black-hole waveforms},
  year          = {2026},
  eprint        = {2608.09978},
  archivePrefix = {arXiv},
  primaryClass  = {gr-qc}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Author

[Beka Modrekiladze](https://inspirehep.net/authors/1749389), DESY Hamburg · beka.modrekiladze@desy.de
