"""Benchmark the mode surrogate's waveform-generation throughput (params -> h+(f)).

Times the full differentiable pipeline (net -> per-mode SVD decode -> SWSH projection
-> rFFT) on GPU at several batch sizes, plus the reverse-mode gradient cost.
"""
import os, sys, time, numpy as np, torch, yaml
sys.path.append(os.getcwd())
from src.physics.projection import load_torch_strain_model

config = yaml.safe_load(open("config.modes_v1.yaml"))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tsm, stats = load_torch_strain_model("runs/modes_m5_600k_ft", config, device=dev)
tsm.eval()
print(f"device={dev}, params={sum(p.numel() for p in tsm.net.parameters()):,}")

def bench(bs, iters=50):
    p = torch.randn(bs, 8, device=dev)
    io = torch.rand(bs, device=dev) * np.pi
    ph = torch.rand(bs, device=dev) * 2 * np.pi
    with torch.no_grad():
        tsm.strain_fd(p, io, ph)
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            tsm.strain_fd(p, io, ph)
    if dev.type == "cuda": torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return dt * 1e3, bs / dt

print(f"{'batch':>8} {'latency_ms':>11} {'wf/s':>14}")
for bs in [1, 10, 100, 1000, 10000]:
    ms, tp = bench(bs, iters=(100 if bs <= 100 else 20))
    print(f"{bs:>8} {ms:>11.2f} {tp:>14.0f}")

# reverse-mode gradient cost (single)
p = torch.randn(1, 8, device=dev, requires_grad=True)
io = torch.rand(1, device=dev) * np.pi; ph = torch.rand(1, device=dev) * 2 * np.pi
h = tsm.strain_fd(p, io, ph); loss = (h.real**2 + h.imag**2).sum()
loss.backward()
if dev.type == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    p.grad = None
    h = tsm.strain_fd(p, io, ph); loss = (h.real**2 + h.imag**2).sum(); loss.backward()
if dev.type == "cuda": torch.cuda.synchronize()
print(f"reverse-mode grad (single): {(time.perf_counter()-t0)/50*1e3:.2f} ms")
