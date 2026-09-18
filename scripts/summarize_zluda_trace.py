#!/usr/bin/env python3
"""Summarize a ZLUDA trace log into kernel/function-level diagnostics.

The parser is intentionally tolerant of additional trace fields. It extracts
function-handle/name mappings, kernel launches, and non-success CUDA results
without requiring a GPU or ZLUDA installation.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

GET_FUNCTION_RE = re.compile(
    r"\bcuModuleGetFunction\([^\n]*?hfunc:\s*(0x[0-9A-Fa-f]+)"
    r"[^\n]*?name:\s*\"([^\"]+)\"[^\n]*?\)\s*->\s*([A-Z0-9_]+)"
)
LAUNCH_RE = re.compile(
    r"\b(cuLaunchKernel(?:Ex)?)\([^\n]*?\b(?:f|kernel):\s*(0x[0-9A-Fa-f]+)"
    r"[^\n]*?\)\s*->\s*([A-Z0-9_]+)"
)
CUDA_RESULT_RE = re.compile(r"\)\s*->\s*(CUDA_[A-Z0-9_]+)\s*$")
MODULE_LOAD_RE = re.compile(
    r"\b(cuModuleLoad(?:Data(?:Ex)?|FatBinary)?)\([^\n]*\)\s*->\s*(CUDA_[A-Z0-9_]+)"
)


def parse_trace(text: str, tail: int) -> dict[str, Any]:
    lines = text.splitlines()
    functions: dict[str, str] = {}
    launches: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    module_loads: list[dict[str, Any]] = []

    for lineno, raw in enumerate(lines, 1):
        line = raw.strip()

        m = GET_FUNCTION_RE.search(line)
        if m:
            handle, name, status = m.groups()
            if status == "CUDA_SUCCESS":
                functions[handle.lower()] = name

        m = LAUNCH_RE.search(line)
        if m:
            api, handle, status = m.groups()
            launches.append(
                {
                    "line": lineno,
                    "api": api,
                    "function_handle": handle,
                    "function_name": functions.get(handle.lower()),
                    "status": status,
                    "raw": line,
                }
            )

        m = MODULE_LOAD_RE.search(line)
        if m:
            api, status = m.groups()
            module_loads.append(
                {
                    "line": lineno,
                    "api": api,
                    "status": status,
                    "raw": line,
                }
            )

        m = CUDA_RESULT_RE.search(line)
        if m and m.group(1) != "CUDA_SUCCESS":
            errors.append(
                {
                    "line": lineno,
                    "status": m.group(1),
                    "raw": line,
                }
            )

    failing_launches = [x for x in launches if x["status"] != "CUDA_SUCCESS"]
    return {
        "schema": 1,
        "line_count": len(lines),
        "function_mappings": len(functions),
        "launch_count": len(launches),
        "failing_launch_count": len(failing_launches),
        "error_count": len(errors),
        "module_load_count": len(module_loads),
        "first_error": errors[0] if errors else None,
        "last_error": errors[-1] if errors else None,
        "first_failing_launch": failing_launches[0] if failing_launches else None,
        "last_launch": launches[-1] if launches else None,
        "launches_tail": launches[-max(1, tail) :],
        "errors_tail": errors[-max(1, tail) :],
        "module_loads_tail": module_loads[-max(1, min(tail, 20)) :],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path, help="ZLUDA trace log.txt")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--tail", type=int, default=12)
    args = parser.parse_args()

    if args.tail < 1:
        parser.error("--tail must be at least 1")
    if not args.log.is_file():
        parser.error(f"trace log not found: {args.log}")

    text = args.log.read_text(encoding="utf-8", errors="replace")
    report = {
        "source": str(args.log),
        **parse_trace(text, args.tail),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
