"""Reference GPU matrix for the experimental cuDNN -> MIOpen bridge.

Run only inside a locally staged experimental runtime. This test covers:
- legacy cuDNN forward via aten::cudnn_convolution,
- public torch.nn.functional.conv2d,
- autograd backward-data and backward-filter through the public Conv2d path.

All GPU results are compared against a CPU PyTorch reference.
"""

import torch
import torch.nn.functional as F

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
# The bridge intentionally does not claim cuDNN determinism until the selected
# MIOpen solutions expose an equivalent guarantee.
torch.backends.cudnn.deterministic = False

print("torch", torch.__version__)
print("device", torch.cuda.get_device_name(0))
print("cudnn_version", torch.backends.cudnn.version())

CASES = [
    dict(name="basic", n=1, cin=1, cout=1, hw=5, k=3, pad=1, stride=1, dilation=1, groups=1),
    dict(name="pad3x3", n=2, cin=3, cout=4, hw=8, k=3, pad=1, stride=1, dilation=1, groups=1),
    dict(name="stride2", n=1, cin=2, cout=3, hw=9, k=3, pad=1, stride=2, dilation=1, groups=1),
    dict(name="dilation2", n=1, cin=2, cout=2, hw=9, k=3, pad=2, stride=1, dilation=2, groups=1),
    dict(name="groups2", n=1, cin=4, cout=6, hw=7, k=3, pad=1, stride=1, dilation=1, groups=2),
]


def values(count: int, scale: float) -> torch.Tensor:
    return (torch.arange(count, dtype=torch.float32) % 17 - 8) * scale


def tensors(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    n, cin, cout, hw, k = (
        case["n"],
        case["cin"],
        case["cout"],
        case["hw"],
        case["k"],
    )
    groups = case["groups"]
    x = values(n * cin * hw * hw, 0.125).reshape(n, cin, hw, hw)
    w = values(cout * (cin // groups) * k * k, 0.0625).reshape(
        cout, cin // groups, k, k
    )
    return x, w


def cpu_conv(case: dict, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return F.conv2d(
        x,
        w,
        stride=case["stride"],
        padding=case["pad"],
        dilation=case["dilation"],
        groups=case["groups"],
    )


def forward_check(case: dict, dtype: torch.dtype) -> None:
    x_cpu, w_cpu = tensors(case)
    expected = cpu_conv(case, x_cpu, w_cpu)
    x = x_cpu.to(device="cuda", dtype=dtype)
    w = w_cpu.to(device="cuda", dtype=dtype)

    direct = torch.ops.aten.cudnn_convolution.default(
        x,
        w,
        [case["pad"], case["pad"]],
        [case["stride"], case["stride"]],
        [case["dilation"], case["dilation"]],
        case["groups"],
        False,
        False,
        False,
    )
    public = cpu_conv(case, x, w)
    torch.cuda.synchronize()

    direct_err = (direct.float().cpu() - expected).abs().max().item()
    public_err = (public.float().cpu() - expected).abs().max().item()
    tolerance = 1e-5 if dtype == torch.float32 else 3e-2
    if direct_err > tolerance or public_err > tolerance:
        raise RuntimeError(
            f"{case['name']} {dtype} forward: direct={direct_err}, "
            f"public={public_err}, tolerance={tolerance}"
        )

    print(
        "FORWARD PASS",
        case["name"],
        dtype,
        "shape",
        tuple(direct.shape),
        "direct_max_abs",
        direct_err,
        "public_max_abs",
        public_err,
    )


def backward_check(case: dict, dtype: torch.dtype) -> None:
    x0, w0 = tensors(case)

    x_cpu = x0.clone().requires_grad_(True)
    w_cpu = w0.clone().requires_grad_(True)
    y_cpu = cpu_conv(case, x_cpu, w_cpu)
    seed = values(y_cpu.numel(), 0.03125).reshape_as(y_cpu)
    y_cpu.backward(seed)
    dx_ref = x_cpu.grad.detach()
    dw_ref = w_cpu.grad.detach()

    x = x0.to(device="cuda", dtype=dtype).requires_grad_(True)
    w = w0.to(device="cuda", dtype=dtype).requires_grad_(True)
    y = cpu_conv(case, x, w)
    y.backward(seed.to(device="cuda", dtype=dtype))
    torch.cuda.synchronize()

    y_err = (y.float().cpu() - y_cpu.detach()).abs().max().item()
    dx_err = (x.grad.float().cpu() - dx_ref).abs().max().item()
    dw_err = (w.grad.float().cpu() - dw_ref).abs().max().item()

    if dtype == torch.float32:
        y_tol = dx_tol = dw_tol = 1e-5
    else:
        y_tol = 3e-2
        dx_tol = 3e-2
        dw_tol = 5e-2

    if y_err > y_tol or dx_err > dx_tol or dw_err > dw_tol:
        raise RuntimeError(
            f"{case['name']} {dtype} backward: y={y_err}/{y_tol}, "
            f"dx={dx_err}/{dx_tol}, dw={dw_err}/{dw_tol}"
        )

    print(
        "BACKWARD PASS",
        case["name"],
        dtype,
        "y_max_abs",
        y_err,
        "dx_max_abs",
        dx_err,
        "dw_max_abs",
        dw_err,
    )


def bias_training_check(dtype: torch.dtype) -> None:
    case = CASES[1]
    x0, w0 = tensors(case)
    bias0 = values(case["cout"], 0.125)

    x_cpu = x0.clone().requires_grad_(True)
    w_cpu = w0.clone().requires_grad_(True)
    b_cpu = bias0.clone().requires_grad_(True)
    y_cpu = F.conv2d(
        x_cpu,
        w_cpu,
        b_cpu,
        stride=case["stride"],
        padding=case["pad"],
        dilation=case["dilation"],
        groups=case["groups"],
    )
    seed = values(y_cpu.numel(), 0.015625).reshape_as(y_cpu)
    y_cpu.backward(seed)

    x = x0.to(device="cuda", dtype=dtype).requires_grad_(True)
    w = w0.to(device="cuda", dtype=dtype).requires_grad_(True)
    b = bias0.to(device="cuda", dtype=dtype).requires_grad_(True)
    y = F.conv2d(
        x,
        w,
        b,
        stride=case["stride"],
        padding=case["pad"],
        dilation=case["dilation"],
        groups=case["groups"],
    )
    y.backward(seed.to(device="cuda", dtype=dtype))
    torch.cuda.synchronize()

    errs = {
        "y": (y.float().cpu() - y_cpu.detach()).abs().max().item(),
        "dx": (x.grad.float().cpu() - x_cpu.grad).abs().max().item(),
        "dw": (w.grad.float().cpu() - w_cpu.grad).abs().max().item(),
        "db": (b.grad.float().cpu() - b_cpu.grad).abs().max().item(),
    }
    tolerance = 1e-5 if dtype == torch.float32 else 5e-2
    if max(errs.values()) > tolerance:
        raise RuntimeError(
            f"bias training {dtype}: {errs}, tolerance={tolerance}"
        )
    print("BIAS TRAINING PASS", dtype, errs)


for item in CASES:
    forward_check(item, torch.float32)
for item in CASES:
    backward_check(item, torch.float32)

# FP16 is validated across the same matrix. Values are chosen to be small and
# stable enough to compare against the FP32 CPU reference with explicit bounds.
for item in CASES:
    forward_check(item, torch.float16)
for item in CASES:
    backward_check(item, torch.float16)

bias_training_check(torch.float32)
bias_training_check(torch.float16)

print("PASS: PyTorch cuDNN bridge forward + backward + bias matrix")