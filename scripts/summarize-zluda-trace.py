#!/usr/bin/env python3
"""Summarize ZLUDA trace output into kernel-oriented diagnostics."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
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
LAUNCH_BLOCKING_RE = re.compile(
    r'^\[zluda-launch\]\s+#(?P<id>\d+)\s+'
    r'(?P<event>begin|done|launch-error|sync-error)\s+'
    r'kernel=(?P<kernel>\S+)(?:\s+(?P<detail>.*))?$'
)


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
    blocking_events: list[dict[str, Any]] = []
    blocking_open: dict[int, dict[str, Any]] = {}
    blocking_errors: list[dict[str, Any]] = []

    for index, raw in enumerate(lines, 1):
        line = clean_line(raw)
        if not line:
            continue

        bm = LAUNCH_BLOCKING_RE.match(line)
        if bm:
            launch_id = int(bm.group("id"))
            event = bm.group("event")
            item = {
                "line": index,
                "launch_id": launch_id,
                "event": event,
                "kernel": bm.group("kernel"),
                "detail": bm.group("detail") or "",
                "text": line,
            }
            blocking_events.append(item)
            if event == "begin":
                blocking_open[launch_id] = item
            else:
                blocking_open.pop(launch_id, None)
                if event in {"launch-error", "sync-error"}:
                    blocking_errors.append(item)

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

    unmatched = sorted(blocking_open.values(), key=lambda item: item["launch_id"])
    completed_ids = {
        item["launch_id"]
        for item in blocking_events
        if item["event"] in {"done", "launch-error", "sync-error"}
    }
    begin_ids = {
        item["launch_id"]
        for item in blocking_events
        if item["event"] == "begin"
    }
    tail_lines = [clean_line(x) for x in lines[-tail:] if clean_line(x)]
    return {
        "schema": 1,
        "resolved_functions": len(functions),
        "launch_count": len(launches),
        "launches": launches,
        "non_success_count": len(failures),
        "non_success_calls": failures,
        "last_launch": launches[-1] if launches else None,
        "launch_blocking": {
            "event_count": len(blocking_events),
            "begin_count": len(begin_ids),
            "terminal_count": len(completed_ids),
            "error_count": len(blocking_errors),
            "events": blocking_events,
            "errors": blocking_errors,
            "unmatched_count": len(unmatched),
            "unmatched": unmatched,
            "last_unmatched": unmatched[-1] if unmatched else None,
            "last_event": blocking_events[-1] if blocking_events else None,
        },
        "tail": tail_lines,
    }


def summarize(path: Path, tail: int) -> dict[str, Any]:
    log_path = find_log(path)
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    result = summarize_lines(lines, tail)
    run_dir = log_path.parent
    compiler_logs = sorted(run_dir.glob("module_*.log"))
    ptx_modules = sorted(run_dir.glob("module_*.ptx"))

    launched_names = {
        item["function_name"]
        for item in result["launches"]
        if item.get("function_name")
    }
    function_ptx_modules: dict[str, list[str]] = {name: [] for name in launched_names}
    for ptx_path in ptx_modules:
        ptx_text = ptx_path.read_text(encoding="utf-8", errors="replace")
        for name in launched_names:
            if name in ptx_text:
                function_ptx_modules[name].append(str(ptx_path))

    last_launch = result.get("last_launch")
    if last_launch and last_launch.get("function_name"):
        last_launch["ptx_modules"] = function_ptx_modules.get(
            last_launch["function_name"], []
        )

    result.update(
        {
            "trace_log": str(log_path),
            "trace_dir": str(run_dir),
            "function_ptx_modules": function_ptx_modules,
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
        '[zluda-launch] #10 begin kernel=kernel_ok grid=1x1x1 block=1x1x1 shared=0 stream=0x1',
        '[zluda-launch] #10 done kernel=kernel_ok',
        '[ZLUDA_TRACE] cuModuleGetFunction(hfunc: 0x00000456, hmod: 0x9, name: "kernel_bad") -> CUDA_SUCCESS',
        '[ZLUDA_TRACE] cuLaunchKernel(f: 0x00000456, gridDimX: 2, sharedMemBytes: 0) -> CUDA_ERROR_LAUNCH_FAILED',
        '[zluda-launch] #11 begin kernel=kernel_bad grid=2x1x1 block=1x1x1 shared=0 stream=0x2',
    ]
    result = summarize_lines(sample, 4)
    assert result["launch_count"] == 2
    assert result["launches"][0]["function_name"] == "kernel_ok"
    assert result["launches"][1]["function_name"] == "kernel_bad"
    assert result["launches"][1]["status"] == "CUDA_ERROR_LAUNCH_FAILED"
    assert result["non_success_count"] == 1
    assert result["launch_blocking"]["begin_count"] == 2
    assert result["launch_blocking"]["terminal_count"] == 1
    assert result["launch_blocking"]["unmatched_count"] == 1
    assert result["launch_blocking"]["last_unmatched"]["launch_id"] == 11
    assert result["launch_blocking"]["last_unmatched"]["kernel"] == "kernel_bad"

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        (run_dir / "log.txt").write_text("\n".join(sample) + "\n", encoding="utf-8")
        (run_dir / "module_0001_01.ptx").write_text(
            '.version 8.4\n.target sm_90\n.visible .entry kernel_bad() { ret; }\n',
            encoding="utf-8",
        )
        file_result = summarize(run_dir, 4)
        assert file_result["last_launch"]["function_name"] == "kernel_bad"
        assert file_result["last_launch"]["ptx_modules"]
        assert file_result["last_launch"]["ptx_modules"][0].endswith(
            "module_0001_01.ptx"
        )
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
    blocking = result.get("launch_blocking") or {}
    if blocking.get("event_count"):
        print(
            "Launch-blocking trace: "
            f"{blocking['begin_count']} begins, "
            f"{blocking['terminal_count']} terminals, "
            f"{blocking['error_count']} errors, "
            f"{blocking['unmatched_count']} unmatched"
        )
        suspect = blocking.get("last_unmatched")
        if suspect:
            print(
                "Crash suspect: "
                f"#{suspect['launch_id']} {suspect['kernel']} "
                f"(begin at line {suspect['line']}, no terminal record)"
            )
        elif blocking.get("last_event"):
            event = blocking["last_event"]
            print(
                "Last blocking event: "
                f"#{event['launch_id']} {event['event']} {event['kernel']} "
                f"(line {event['line']})"
            )
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
