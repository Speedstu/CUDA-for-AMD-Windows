#!/usr/bin/env python3
"""Summarize ZLUDA trace output into kernel-oriented diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

FUNC_RE = re.compile(
    r'cuModuleGetFunction\(hfunc:\s*(0x[0-9A-Fa-f]+).*?name:\s*"([^"]+)"\).*?->\s*([A-Z0-9_]+)'
)
LAUNCH_RE = re.compile(
    r'(cuLaunchKernel(?:Ex(?:_ptsz)?|_ptsz)?|cuLaunchCooperativeKernel(?:_ptsz)?)\((.*?)\)\s*->\s*([A-Z0-9_]+)'
)
HANDLE_RE = re.compile(r'(?:^|[,\s])f:\s*(0x[0-9A-Fa-f]+)')
RESULT_RE = re.compile(r'\)\s*->\s*([A-Z][A-Z0-9_]+)\s*$')
TRACE_PREFIX_RE = re.compile(r'^\[ZLUDA_TRACE\]\s*')


def find_log(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(path)
    logs = list(path.rglob("log.txt"))
    if not logs:
        raise FileNotFoundError(f"no log.txt found under {path}")
    return max(logs, key=lambda p: p.stat().st_mtime_ns)


def clean_line(line: str) -> str:
    return TRACE_PREFIX_RE.sub("", line.strip())


def summarize_lines(lines: list[str], tail: int = 30) -> dict[str, Any]:
    functions: dict[str, str] = {}
    launches: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for index, raw in enumerate(lines, 1):
        line = clean_line(raw)
        if not line:
            continue

        m = FUNC_RE.search(line)
        if m:
            handle, name, status = m.groups()
            if status == "CUDA_SUCCESS":
                functions[handle.lower()] = name

        lm = LAUNCH_RE.search(line)
        if lm:
            api, args, status = lm.groups()
            hm = HANDLE_RE.search(args)
            handle = hm.group(1).lower() if hm else None
            launches.append(
                {
                    "line": index,
                    "api": api,
                    "function_handle": handle,
                    "function_name": functions.get(handle) if handle else None,
                    "status": status,
                    "text": line,
                }
            )

        rm = RESULT_RE.search(line)
        if rm:
            status = rm.group(1)
            if status not in {
                "CUDA_SUCCESS",
                "CUBLAS_STATUS_SUCCESS",
                "CUSPARSE_STATUS_SUCCESS",
                "CUFFT_SUCCESS",
                "CUDNN_STATUS_SUCCESS",
                "NVML_SUCCESS",
            }:
                failures.append({"line": index, "status": status, "text": line})

    tail_lines = [clean_line(x) for x in lines[-tail:] if clean_line(x)]
    return {
        "schema": 1,
        "resolved_functions": len(functions),
        "launch_count": len(launches),
        "launches": launches,
        "non_success_count": len(failures),
        "non_success_calls": failures,
        "last_launch": launches[-1] if launches else None,
        "tail": tail_lines,
    }


def summarize(path: Path, tail: int) -> dict[str, Any]:
    log_path = find_log(path)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = summarize_lines(lines, tail)
    run_dir = log_path.parent
    compiler_logs = sorted(run_dir.glob("module_*.log"))
    ptx_modules = sorted(run_dir.glob("module_*.ptx"))
    result.update(
        {
            "trace_log": str(log_path),
            "trace_dir": str(run_dir),
            "compiler_logs": [
                {
                    "path": str(p),
                    "tail": p.read_text(encoding="utf-8", errors="replace").splitlines()[-20:],
                }
                for p in compiler_logs
            ],
            "ptx_modules": [str(p) for p in ptx_modules],
        }
    )
    return result


def self_test() -> None:
    sample = [
        '[ZLUDA_TRACE] cuModuleGetFunction(hfunc: 0x00000123, hmod: 0x9, name: "kernel_ok") -> CUDA_SUCCESS',
        '[ZLUDA_TRACE] cuLaunchKernel(f: 0x00000123, gridDimX: 1, sharedMemBytes: 0) -> CUDA_SUCCESS',
        '[ZLUDA_TRACE] cuModuleGetFunction(hfunc: 0x00000456, hmod: 0x9, name: "kernel_bad") -> CUDA_SUCCESS',
        '[ZLUDA_TRACE] cuLaunchKernel(f: 0x00000456, gridDimX: 2, sharedMemBytes: 0) -> CUDA_ERROR_LAUNCH_FAILED',
    ]
    result = summarize_lines(sample, 4)
    assert result["launch_count"] == 2
    assert result["launches"][0]["function_name"] == "kernel_ok"
    assert result["launches"][1]["function_name"] == "kernel_bad"
    assert result["launches"][1]["status"] == "CUDA_ERROR_LAUNCH_FAILED"
    assert result["non_success_count"] == 1
    print("self-test: ok")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize a ZLUDA --zluda-trace run and correlate kernel handles with launch calls."
    )
    parser.add_argument("path", nargs="?", help="Trace run directory, parent directory, or log.txt")
    parser.add_argument("--tail", type=int, default=30, help="Number of final trace lines to include")
    parser.add_argument("--json", dest="json_path", help="Optional JSON report output path")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if not args.path:
        parser.error("path is required unless --self-test is used")
    if args.tail < 1:
        parser.error("--tail must be at least 1")

    try:
        result = summarize(Path(args.path), args.tail)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json_path:
        out = Path(args.json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"Trace: {result['trace_log']}")
    print(f"Resolved functions: {result['resolved_functions']}")
    print(f"Kernel launches: {result['launch_count']}")
    print(f"Non-success calls: {result['non_success_count']}")
    last = result.get("last_launch")
    if last:
        name = last.get("function_name") or "<unresolved>"
        print(f"Last launch: line {last['line']} {last['api']} {name} -> {last['status']}")
    if result["compiler_logs"]:
        print("Compiler logs:")
        for item in result["compiler_logs"]:
            print(f"  {item['path']}")
    if result["non_success_calls"]:
        print("Recent non-success calls:")
        for item in result["non_success_calls"][-10:]:
            print(f"  L{item['line']}: {item['status']} | {item['text']}")
    if args.json_path:
        print(f"JSON: {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
