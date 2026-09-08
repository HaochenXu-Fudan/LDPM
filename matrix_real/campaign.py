#!/usr/bin/env python3
"""Run and summarize the ten-repetition School experiment."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
DRIVER = HERE / "experiment.py"
SEEDS = tuple(range(2026, 2036))
METHODS = ("LDPM-CS", "LDPM-CS-C")
METRICS = (
    "time",
    "validation_error",
    "test_error",
    "test_error_infeasibility",
    "feasibility",
)


def _parse_seeds(raw: str) -> List[int]:
    seeds = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if len(seeds) != len(SEEDS) or len(set(seeds)) != len(SEEDS):
        raise ValueError("the School manuscript protocol requires ten distinct seeds")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "school.mat")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results" / "school")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--time-limit", type=float, default=1800.0)
    return parser


def _run_dir(root: Path, seed: int) -> Path:
    return root / ("seed%d" % int(seed))


def _is_complete(directory: Path, seed: int) -> bool:
    protocol_path = directory / "protocol.json"
    summary_path = directory / "summary.json"
    if not protocol_path.exists() or not summary_path.exists():
        return False
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    config = protocol.get("ldpm_config", {})
    return bool(
        int(protocol.get("seed", -1)) == int(seed)
        and {row.get("method") for row in rows} == set(METHODS)
        and float(config.get("tol", np.nan)) == 1e-5
        and float(config.get("beta0", np.nan)) == 1.0
        and float(config.get("beta_power", np.nan)) == 0.2
        and float(config.get("beta_max", np.nan)) == 5.0
        and float(config.get("gamma", np.nan)) == 10.0
    )


def _command(args: argparse.Namespace, seed: int, directory: Path) -> List[str]:
    return [
        sys.executable,
        str(DRIVER),
        "--data",
        str(args.data),
        "--output-dir",
        str(directory),
        "--methods",
        "ldpm,ldpm-capped",
        "--seed",
        str(seed),
        "--tol",
        "1e-5",
        "--beta0",
        "1",
        "--beta-power",
        "0.2",
        "--beta-max",
        "5",
        "--gamma",
        "10",
        "--consensus-tol",
        "1e-5",
        "--max-time",
        "%.17g" % float(args.time_limit),
        "--skip-heatmap",
    ]


def _run_one(args: argparse.Namespace, seed: int) -> Dict[str, object]:
    directory = _run_dir(args.results_dir, seed)
    if not args.overwrite and _is_complete(directory, seed):
        return {"seed": seed, "status": "reused", "directory": str(directory)}
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / "run.log"
    command = _command(args, seed, directory)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: %s\n" % " ".join(command))
        log.flush()
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError("School seed %d failed; see %s" % (seed, log_path))
    if not _is_complete(directory, seed):
        raise RuntimeError("School seed %d produced an incomplete protocol" % seed)
    return {"seed": seed, "status": "completed", "directory": str(directory)}


def _read_rows(root: Path, seeds: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        path = _run_dir(root, seed) / "summary.json"
        if not path.exists():
            raise FileNotFoundError("missing formal result: %s" % path)
        values = json.loads(path.read_text(encoding="utf-8"))
        by_method = {str(row.get("method")): row for row in values}
        if set(by_method) != set(METHODS):
            raise RuntimeError("unexpected method set in %s" % path)
        for method in METHODS:
            source = by_method[method]
            row: Dict[str, object] = {
                "seed": int(seed),
                "method": method,
                "status": source.get("status"),
                "iterations": source.get("iterations"),
                "cap_reached": source.get("cap_reached"),
            }
            row.update({metric: source.get(metric) for metric in METRICS})
            rows.append(row)
    return rows


def _summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        summary: Dict[str, object] = {
            "method": method,
            "runs": len(selected),
            "statuses": ";".join(sorted({str(row["status"]) for row in selected})),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            summary[metric + "_mean"] = float(np.mean(values))
            summary[metric + "_std"] = float(np.std(values, ddof=1))
        output.append(summary)
    return output


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty result table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    seeds = _parse_seeds(args.seeds)
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    run_status: List[Dict[str, object]] = []
    if not args.summarize_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(_run_one, args, seed): seed for seed in seeds}
            for future in as_completed(futures):
                status = future.result()
                run_status.append(status)
                print("School seed=%d %s" % (status["seed"], status["status"]), flush=True)

    rows = _read_rows(args.results_dir, seeds)
    summary = _summarize(rows)
    _write_csv(args.results_dir / "all_runs.csv", rows)
    _write_csv(args.results_dir / "summary.csv", summary)
    protocol = {
        "dataset": "School",
        "repetitions": 10,
        "seeds": list(seeds),
        "methods": list(METHODS),
        "tol": 1e-5,
        "beta": {"beta0": 1.0, "power": 0.2, "gamma": 10.0, "capped_max": 5.0},
        "run_status": sorted(run_status, key=lambda row: int(row["seed"])),
    }
    (args.results_dir / "campaign.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Saved School ten-repetition summary under %s" % args.results_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
