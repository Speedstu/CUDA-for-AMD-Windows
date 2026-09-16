#!/usr/bin/env python3
"""Warm common CUDA-facing PyTorch training kernels into ZLUDA's persistent cache."""
from __future__ import annotations

import json
import sys
import time
from typing import Callable


def main() -> int:
    import torch
    import torch.nn as nn

    torch.manual_seed(20260916)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-facing device is unavailable")

    device = torch.device("cuda")
    results: list[dict[str, object]] = []

    def timed(name: str, step: Callable[[], None]) -> None:
        samples: list[float] = []
        for iteration in range(2):
            torch.cuda.synchronize()
            started = time.perf_counter()
            step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
        results.append({
            "name": name,
            "first_ms": samples[0],
            "second_ms": samples[1],
        })
        print(f"{name}: first={samples[0]:.3f} ms second={samples[1]:.3f} ms", flush=True)

    x_relu = torch.randn(8192, device=device, requires_grad=True)
    def relu_step() -> None:
        x_relu.grad = None
        torch.relu(x_relu).sum().backward()
    timed("relu_backward", relu_step)

    x_clip = torch.randn(8192, device=device, requires_grad=True)
    def clamp_step() -> None:
        x_clip.grad = None
        x_clip.clamp(-0.1, 0.1).square().mean().backward()
    timed("clamp_backward", clamp_step)

    x_min = torch.randn(8192, device=device, requires_grad=True)
    y_min = torch.randn(8192, device=device)
    def minimum_step() -> None:
        x_min.grad = None
        torch.minimum(x_min, y_min).square().mean().backward()
    timed("minimum_backward", minimum_step)

    x_max = torch.randn(8192, device=device, requires_grad=True)
    y_max = torch.randn(8192, device=device)
    def maximum_step() -> None:
        x_max.grad = None
        torch.maximum(x_max, y_max).square().mean().backward()
    timed("maximum_backward", maximum_step)

    linear = nn.Linear(512, 126).to(device)
    x_linear = torch.randn(64, 512, device=device, requires_grad=True)
    def linear_step() -> None:
        linear.zero_grad(set_to_none=True)
        x_linear.grad = None
        out = linear(x_linear)
        out.square().mean().backward()
    timed("linear_backward", linear_step)

    norm = nn.LayerNorm(512).to(device)
    x_norm = torch.randn(64, 512, device=device, requires_grad=True)
    def norm_step() -> None:
        norm.zero_grad(set_to_none=True)
        x_norm.grad = None
        norm(x_norm).square().mean().backward()
    timed("layernorm_backward", norm_step)

    model = nn.Sequential(
        nn.Linear(256, 512), nn.ReLU(), nn.LayerNorm(512), nn.Linear(512, 64)
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x_opt = torch.randn(128, 256, device=device)
    target = torch.randn(128, 64, device=device)
    def optimizer_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x_opt) - target).square().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    timed("adamw_training_step", optimizer_step)

    payload = {
        "schema": 1,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
        "ok": True,
    }
    print("CUDAAMD_WARMUP:" + json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("CUDAAMD_WARMUP:" + json.dumps({"schema": 1, "ok": False, "error": str(exc)}), file=sys.stderr, flush=True)
        raise
