#!/usr/bin/env python3
"""Fail-closed, read-only check of measured evidence before promoting a driver.

This gate can rule out incomplete/mismatched measurements. Passing it does
not establish universal CUDA compatibility or replace independent review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

DRIVER_REQUIRED = (
    "driver_function_metadata",
    "driver_ptx_selection",
    "cub_sort_scan",
    "driver_buffer_clear",
)
FUNCTIONAL_REQUIRED = ("matmul", "sdpa_math")
BAD_STATUSES = {"incorrect", "error", "timeout", "hang_after_result",
                "crash_after_result", "unavailable"}


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def evaluate(cap: dict, fun: dict, *, driver_sha: str, arch: str,
             revision: str, now: datetime, max_age_hours: int = 168) -> list[str]:
    failures: list[str] = []
    if not re.fullmatch(r"[0-9a-fA-F]{64}", driver_sha):
        failures.append("Invalid driver SHA-256")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        failures.append("Invalid project revision")
    if not re.fullmatch(r"gfx[0-9a-fA-F]+", arch):
        failures.append("Invalid GPU architecture")
    for name, report in (("capability", cap), ("functional", fun)):
        if report.get("schema") != 1:
            failures.append(f"{name}: unsupported report schema")
        if str(report.get("project_revision", "")).lower() != revision.lower():
            failures.append(f"{name}: project revision mismatch")
        if (report.get("gpu") or {}).get("arch") != arch:
            failures.append(f"{name}: GPU architecture mismatch")
        hashes = report.get("runtime_hashes") or {}
        if str(hashes.get("nvcuda_sha256") or "").lower() != driver_sha.lower():
            failures.append(f"{name}: tested driver SHA-256 mismatch or missing")
        created = report.get("generated_utc")
        try:
            stamped = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            if stamped.tzinfo is None:
                raise ValueError("timezone missing")
            hours = (now - stamped.astimezone(timezone.utc)).total_seconds() / 3600
            if not 0 <= hours <= max_age_hours:
                failures.append(f"{name}: report expired or from the future")
        except (ValueError, TypeError, OverflowError):
            failures.append(f"{name}: missing or invalid UTC timestamp")

    if cap.get("no_silent_corruption") is not True:
        failures.append("capability: silent corruption not ruled out")
    if cap.get("no_hangs") is not True or cap.get("no_process_crashes") is not True:
        failures.append("capability: a GPU probe hung or crashed")
    for key in ("incorrect", "timeouts", "process_hangs", "process_crashes", "errors", "timeout_retries"):
        if type(cap.get(key)) is not int or cap[key] != 0:
            failures.append(f"capability: {key} is not zero")
    results = cap.get("results")
    if not isinstance(results, list) or not results:
        failures.append("capability: missing probe results")
        results = []
    selected = cap.get("selected_tests")
    if not isinstance(selected, list) or len(selected) != len(set(selected)):
        failures.append("capability: selected_tests missing or duplicated")
    elif set(selected) != {x.get("test") for x in results if isinstance(x, dict)}:
        failures.append("capability: selected tests differ from actual results")

    by_name: dict[str, dict] = {}
    for item in results:
        if not isinstance(item, dict) or not isinstance(item.get("test"), str):
            failures.append("capability: invalid result item")
            continue
        name = item["test"]
        if name in by_name:
            failures.append(f"capability: duplicate result {name}")
        by_name[name] = item
        if item.get("status") in BAD_STATUSES:
            failures.append(f"capability: {name} returned {item['status']}")
        if item.get("status") == "pass" and (
                (item.get("result") or {}).get("ok") is not True or
                item.get("process_exit") != 0 or
                item.get("timed_out") is not False or
                item.get("process_hung_after_result") is not False or
                item.get("process_crashed_after_result") is not False):
            failures.append(f"capability: {name} lacks a clean PASS")
    for test in DRIVER_REQUIRED:
        entry = by_name.get(test) or {}
        if (entry.get("status") != "pass" or
                (entry.get("result") or {}).get("ok") is not True or
                entry.get("process_exit") != 0 or
                entry.get("timed_out") is not False or
                entry.get("process_hung_after_result") is not False or
                entry.get("process_crashed_after_result") is not False):
            failures.append(f"capability: required {test} has no clean numerical PASS")

    if fun.get("core_correctness_ok") is not True:
        failures.append("functional: core correctness failed")
    if type(fun.get("hard_failures")) is not int or fun["hard_failures"] != 0:
        failures.append("functional: hard_failures is not zero")
    tests = fun.get("tests")
    if not isinstance(tests, list) or not tests:
        failures.append("functional: missing test results")
        tests = []
    by_test: dict[str, dict] = {}
    for item in tests:
        if not isinstance(item, dict) or not isinstance(item.get("test"), str):
            failures.append("functional: invalid test item")
            continue
        if item["test"] in by_test:
            failures.append(f"functional: duplicate {item['test']}")
        by_test[item["test"]] = item
        if item.get("status") in BAD_STATUSES:
            failures.append(f"functional: {item['test']} returned {item['status']}")
        if item.get("status") == "pass" and (
                (item.get("result") or {}).get("ok") is not True or
                item.get("process_exit") != 0 or
                item.get("timed_out") is not False or
                item.get("process_crashed_after_result") is not False):
            failures.append(f"functional: {item['test']} lacks a clean PASS")
    for test in FUNCTIONAL_REQUIRED:
        row = by_test.get(test) or {}
        if (row.get("status") != "pass" or
                (row.get("result") or {}).get("ok") is not True or
                row.get("process_exit") != 0 or
                row.get("timed_out") is not False or
                row.get("process_crashed_after_result") is not False):
            failures.append(f"functional: required {test} has no clean numerical PASS")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capabilities", type=Path, required=True)
    parser.add_argument("--functional", type=Path, required=True)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--arch", required=True, help="Actual AMD target, e.g. gfx1150")
    parser.add_argument("--revision", required=True, help="Full 40-hex project commit ID")
    parser.add_argument("--max-age-hours", type=int, default=168)
    args = parser.parse_args(argv)
    if args.max_age_hours < 1:
        parser.error("--max-age-hours must be positive")
    try:
        cap = json.loads(args.capabilities.read_text(encoding="utf-8-sig"))
        fun = json.loads(args.functional.read_text(encoding="utf-8-sig"))
        digest = sha256(args.driver)
        issues = evaluate(cap, fun, driver_sha=digest, arch=args.arch,
                          revision=args.revision, now=datetime.now(timezone.utc),
                          max_age_hours=args.max_age_hours)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({"eligible_for_review": False, "failures": [str(exc)]}))
        return 2
    payload = {"eligible_for_review": not issues, "driver_sha256": digest,
               "gpu_arch": args.arch, "checked_revision": args.revision,
               "failures": sorted(set(issues)),
               "note": "Pass means evidence is consistent, not universal CUDA support"}
    print(json.dumps(payload, indent=2))
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
