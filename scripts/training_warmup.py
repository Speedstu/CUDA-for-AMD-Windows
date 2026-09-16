#!/usr/bin/env python3
"""Warm common CUDA-facing PyTorch training kernels into ZLUDA's persistent cache."""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Callable


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true", help="Warm slower general-purpose PyTorch kernels too")
    args = parser.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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

    if args.extended:
        sort_x = torch.randn(8, 2048, device=device)
        def sort_topk_step() -> None:
            torch.sort(sort_x, dim=-1)
            torch.topk(sort_x, 37, dim=-1)
        timed("sort_topk", sort_topk_step)

        det_x = torch.eye(12, device=device) + 0.05 * torch.randn(12, 12, device=device)
        def det_step() -> None:
            torch.linalg.det(det_x)
            torch.linalg.slogdet(det_x)
        timed("det_slogdet", det_step)

        batch_norm = nn.BatchNorm2d(16, affine=True, track_running_stats=False).to(device)
        x_bn = torch.randn(8, 16, 16, 16, device=device, requires_grad=True)
        def batch_norm_step() -> None:
            batch_norm.zero_grad(set_to_none=True)
            x_bn.grad = None
            batch_norm(x_bn).square().mean().backward()
        timed("batchnorm_backward", batch_norm_step)

        group_norm = nn.GroupNorm(4, 16).to(device)
        x_gn = torch.randn(8, 16, 16, 16, device=device, requires_grad=True)
        def group_norm_step() -> None:
            group_norm.zero_grad(set_to_none=True)
            x_gn.grad = None
            group_norm(x_gn).square().mean().backward()
        timed("groupnorm_backward", group_norm_step)

        x_drop = torch.ones(100000, device=device, requires_grad=True)
        def dropout_step() -> None:
            x_drop.grad = None
            F.dropout(x_drop, p=0.25, training=True).sum().backward()
        timed("dropout_backward", dropout_step)

        x_grid = torch.randn(2, 3, 16, 18, device=device, requires_grad=True)
        grid = torch.empty(2, 12, 14, 2, device=device).uniform_(-1, 1).requires_grad_(True)
        def grid_sample_step() -> None:
            x_grid.grad = None
            grid.grad = None
            F.grid_sample(x_grid, grid, mode="bilinear", padding_mode="zeros", align_corners=False).square().mean().backward()
        timed("grid_sample_backward", grid_sample_step)

        x_deconv = torch.randn(2, 4, 16, 16, device=device)
        w_deconv = torch.randn(4, 6, 3, 3, device=device)
        def conv_transpose_step() -> None:
            F.conv_transpose2d(x_deconv, w_deconv, stride=2, padding=1)
        timed("conv_transpose2d", conv_transpose_step)

    payload = {
        "schema": 1,
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "extended": args.extended,
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
