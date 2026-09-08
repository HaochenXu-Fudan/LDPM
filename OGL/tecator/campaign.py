#!/usr/bin/env python3
"""Run and aggregate ten Tecator overlapping-group-Lasso splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OGL_DIR = SCRIPT_DIR.parent
DRIVER = SCRIPT_DIR / "experiment.py"
METHODS: Tuple[str, ...] = ("VF-iDCA", "LDMMA", "LDPM-CS", "LDPM-CS-C")
METHOD_SLUGS = {
    "VF-iDCA": "vf_idca",
    "LDMMA": "ldmma",
    "LDPM-CS": "ldpm",
    "LDPM-CS-C": "ldpm_capped",
}
COLORS = {
    "VF-iDCA": "#d55e00",
    "LDMMA": "#e69f00",
    "LDPM-CS": "#0072b2",
    "LDPM-CS-C": "#009e73",
}
METRICS: Tuple[str, ...] = (
    "time",
    "validation_error",
    "test_error",
    "test_error_infeasibility",
    "feasibility",
)
CURVE_METRICS = {
    "validation": ("validation_error_best_so_far", "Validation error"),
    "test": ("test_error_best_so_far", "Test error"),
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in range(2026, 2036))
    )
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results" / "tecator"),
    )
    parser.add_argument(
        "--first-run-dir",
        default="",
        help="Existing seed-2026 run to reuse; pass an empty string to rerun it.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--max-runtime", type=float, default=1800.0)
    parser.add_argument("--beta-max-capped", type=float, default=4.0)
    parser.add_argument("--curve-checkpoints", type=int, default=500)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--trend-time-constant", type=float, default=8.0)
    return parser.parse_args(argv)


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("exactly ten distinct seeds are required")
    return seeds


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> object:
    return json.loads(path.read_text())


def read_history_endpoints(path: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        first = next(reader)
        last = first
        for row in reader:
            last = row
    return first, last


def run_dir_for_seed(root: Path, seed: int, first_run_dir: Optional[Path]) -> Path:
    if seed == 2026 and first_run_dir is not None:
        return first_run_dir
    return root / ("seed_%d" % seed)


def expected_methods(summary_path: Path) -> bool:
    if not summary_path.exists():
        return False
    try:
        rows = read_json(summary_path)
    except Exception:
        return False
    if not isinstance(rows, list):
        return False
    return {str(row.get("method")) for row in rows} == set(METHODS)


def run_is_complete(
    run_dir: Path,
    seed: int,
    max_runtime: float,
    beta_max_capped: float,
) -> bool:
    required = [
        run_dir / "summary.json",
        run_dir / "protocol.json",
        run_dir / "error_time_curves.csv",
    ]
    required.extend(
        run_dir / (METHOD_SLUGS[method] + "_history.csv") for method in METHODS
    )
    if not all(path.exists() for path in required):
        return False
    if not expected_methods(run_dir / "summary.json"):
        return False
    try:
        protocol = read_json(run_dir / "protocol.json")
        assert isinstance(protocol, dict)
        return bool(
            int(protocol["seed"]) == seed
            and abs(float(protocol["common_algorithm_time_budget_seconds"]) - max_runtime)
            <= 1e-12
            and abs(float(protocol["beta"]["capped_max"]) - beta_max_capped)
            <= 1e-12
        )
    except Exception:
        return False


def driver_command(
    run_dir: Path,
    seed: int,
    max_runtime: float,
    beta_max_capped: float,
    curve_checkpoints: int,
) -> List[str]:
    return [
        sys.executable,
        str(DRIVER),
        "--data-path",
        str(OGL_DIR / "data" / "tecator" / "tecator_raw.csv"),
        "--results-dir",
        str(run_dir),
        "--seed",
        str(seed),
        "--methods",
        "vf_idca,ldmma,ldpm,ldpm_capped",
        "--tol",
        "1e-5",
        "--std-eps",
        "1e-12",
        "--stop-patience",
        "1",
        "--ldpm-max-iter",
        "1000000",
        "--ldpm-record-interval",
        "20",
        "--curve-checkpoints",
        str(curve_checkpoints),
        "--beta0",
        "0.3",
        "--beta-power",
        "0.3",
        "--beta-max-capped",
        str(beta_max_capped),
        "--gamma",
        "1.0",
        "--initial-lambda",
        "1e-3",
        "--initial-r",
        "0.1",
        "--ldpm-init-ridge",
        "1e-3",
        "--ldpm-step",
        "1e-2",
        "--ldpm-line-search-max-step",
        "5e-2",
        "--ldpm-line-search-min-step",
        "1e-12",
        "--ldpm-line-search-decay",
        "0.5",
        "--ldpm-line-search-growth",
        "1.25",
        "--ldpm-line-search-max-iter",
        "50",
        "--baseline-timeout",
        "180",
        "--max-runtime",
        str(max_runtime),
        "--vf-max-iter",
        "100000",
        "--solver",
        "SCS",
        "--solver-tol",
        "1e-5",
        "--solver-max-iters",
        "10000",
    ]


def run_campaign(
    root: Path,
    seeds: Sequence[int],
    first_run_dir: Optional[Path],
    args: argparse.Namespace,
) -> Dict[int, Path]:
    run_dirs: Dict[int, Path] = {}
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/private/tmp/tecator_pycache"
    environment["MPLCONFIGDIR"] = "/private/tmp/tecator_mpl"
    python_path = environment.get("PYTHONPATH", "")
    prefixes = ["/private/tmp/vfidca-deps", str(OGL_DIR)]
    if python_path:
        prefixes.append(python_path)
    environment["PYTHONPATH"] = os.pathsep.join(prefixes)

    for seed in seeds:
        run_dir = run_dir_for_seed(root, seed, first_run_dir)
        run_dirs[seed] = run_dir
        if args.resume and run_is_complete(
            run_dir, seed, args.max_runtime, args.beta_max_capped
        ):
            print("Reusing complete Tecator seed=%d from %s" % (seed, run_dir), flush=True)
            continue
        if args.aggregate_only:
            if not run_is_complete(
                run_dir, seed, args.max_runtime, args.beta_max_capped
            ):
                raise RuntimeError("incomplete run: %s" % run_dir)
            continue
        run_dir.mkdir(parents=True, exist_ok=True)
        print("Running Tecator seed=%d serially" % seed, flush=True)
        subprocess.run(
            driver_command(
                run_dir,
                seed,
                args.max_runtime,
                args.beta_max_capped,
                args.curve_checkpoints,
            ),
            cwd=str(OGL_DIR),
            env=environment,
            check=True,
        )
    return run_dirs


def compare_ldpm_prefix(
    uncapped_path: Path, capped_path: Path
) -> Dict[str, object]:
    last_identical_iteration: Optional[int] = None
    first_different: Optional[Dict[str, object]] = None
    with uncapped_path.open(newline="") as left, capped_path.open(newline="") as right:
        left_reader = csv.DictReader(left)
        right_reader = csv.DictReader(right)
        for uncapped, capped in zip(left_reader, right_reader):
            left_iteration = int(float(uncapped["iteration"]))
            right_iteration = int(float(capped["iteration"]))
            if left_iteration != right_iteration:
                break
            same = bool(
                uncapped.get("x_values") == capped.get("x_values")
                and uncapped.get("lambda_values") == capped.get("lambda_values")
            )
            if same:
                last_identical_iteration = left_iteration
                continue
            first_different = {
                "first_different_iteration": left_iteration,
                "first_different_uncapped_time": float(uncapped["time"]),
                "first_different_capped_time": float(capped["time"]),
                "first_different_uncapped_beta": float(uncapped["beta"]),
                "first_different_capped_beta": float(capped["beta"]),
            }
            break
    return {
        "last_exactly_identical_iteration": last_identical_iteration,
        **(first_different or {}),
    }


def verify_run(
    run_dir: Path,
    seed: int,
    max_runtime: float,
    beta_max_capped: float,
) -> Dict[str, object]:
    protocol = read_json(run_dir / "protocol.json")
    summary = read_json(run_dir / "summary.json")
    if not isinstance(protocol, dict) or not isinstance(summary, list):
        raise RuntimeError("invalid JSON artifacts in %s" % run_dir)
    if int(protocol["seed"]) != seed:
        raise RuntimeError("seed mismatch in %s" % run_dir)
    if abs(float(protocol["common_algorithm_time_budget_seconds"]) - max_runtime) > 1e-12:
        raise RuntimeError("time-budget mismatch in %s" % run_dir)
    if abs(float(protocol["beta"]["capped_max"]) - beta_max_capped) > 1e-12:
        raise RuntimeError("capped-beta mismatch in %s" % run_dir)

    first_rows: Dict[str, Dict[str, str]] = {}
    last_rows: Dict[str, Dict[str, str]] = {}
    for method in METHODS:
        slug = METHOD_SLUGS[method]
        first, last = read_history_endpoints(run_dir / (slug + "_history.csv"))
        first_rows[method] = first
        last_rows[method] = last
        if int(float(first["iteration"])) != 0 or abs(float(first["time"])) > 1e-15:
            raise RuntimeError("non-common initial timestamp for %s seed=%d" % (method, seed))
    if len({first_rows[method]["x_values"] for method in METHODS}) != 1:
        raise RuntimeError("initial x differs across methods for seed=%d" % seed)
    if len({first_rows[method]["lambda_values"] for method in METHODS}) != 1:
        raise RuntimeError("initial lambda differs across methods for seed=%d" % seed)
    initial_lambda = np.fromstring(first_rows["LDPM-CS"]["lambda_values"], sep=";")
    if initial_lambda.size != 19 or not np.allclose(initial_lambda, 1e-3, atol=0.0, rtol=0.0):
        raise RuntimeError("unexpected initial lambda for seed=%d" % seed)

    capped_betas: List[float] = []
    cap_iteration: Optional[int] = None
    cap_time: Optional[float] = None
    with (run_dir / "ldpm_capped_history.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            beta = float(row["beta"])
            capped_betas.append(beta)
            if cap_iteration is None and beta >= beta_max_capped - 1e-12:
                cap_iteration = int(float(row["iteration"]))
                cap_time = float(row["time"])
    if not capped_betas or max(capped_betas) > beta_max_capped + 1e-12:
        raise RuntimeError("capped beta exceeded its bound for seed=%d" % seed)
    if cap_iteration is None:
        raise RuntimeError("cap did not activate for seed=%d" % seed)

    by_method = {str(row["method"]): row for row in summary}
    if set(by_method) != set(METHODS):
        raise RuntimeError("method mismatch in %s" % (run_dir / "summary.json"))
    capped_summary = by_method["LDPM-CS-C"]
    if not bool(capped_summary.get("cap_reached")):
        raise RuntimeError("summary does not confirm cap activation for seed=%d" % seed)
    if abs(float(capped_summary["final_beta"]) - beta_max_capped) > 1e-12:
        raise RuntimeError("unexpected capped final beta for seed=%d" % seed)
    uncapped_beta = float(by_method["LDPM-CS"]["final_beta"])
    if uncapped_beta <= beta_max_capped:
        raise RuntimeError("uncapped beta never exceeded the cap for seed=%d" % seed)

    for method in METHODS:
        status = str(by_method[method]["status"])
        budget_flag = str(last_rows[method].get("time_budget_reached", "")).lower()
        if status != "converged" and budget_flag not in {"true", "1"}:
            raise RuntimeError("incomplete algorithm budget for %s seed=%d" % (method, seed))

    prefix = compare_ldpm_prefix(
        run_dir / "ldpm_history.csv", run_dir / "ldpm_capped_history.csv"
    )
    if prefix.get("first_different_capped_beta") != beta_max_capped:
        raise RuntimeError("LDPM variants did not first diverge at the cap for seed=%d" % seed)
    return {
        "seed": seed,
        "run_dir": str(run_dir.resolve()),
        "common_start_verified": True,
        "time_budget_seconds": max_runtime,
        "cap_reached": True,
        "cap_iteration": cap_iteration,
        "cap_time_seconds": cap_time,
        "capped_final_beta": float(capped_summary["final_beta"]),
        "uncapped_final_beta": uncapped_beta,
        **prefix,
    }


def load_all_runs(run_dirs: Dict[int, Path]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed, run_dir in run_dirs.items():
        block = read_json(run_dir / "summary.json")
        assert isinstance(block, list)
        by_method = {str(row["method"]): row for row in block}
        for method in METHODS:
            row = dict(by_method[method])
            row["seed"] = seed
            row["run_dir"] = str(run_dir.resolve())
            rows.append(row)
    return rows


def aggregate_summary(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for method in METHODS:
        block = [row for row in rows if row["method"] == method]
        aggregate: Dict[str, object] = {
            "method": method,
            "runs": len(block),
            "statuses": ";".join(sorted({str(row["status"]) for row in block})),
            "cap_reached_runs": int(sum(bool(row.get("cap_reached")) for row in block)),
        }
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in block], dtype=float)
            finite = values[np.isfinite(values)]
            aggregate[metric + "_valid_runs"] = int(finite.size)
            aggregate[metric + "_mean"] = float(np.mean(finite))
            aggregate[metric + "_std"] = (
                float(np.std(finite, ddof=1)) if finite.size >= 2 else None
            )
            aggregate[metric + "_variance"] = (
                float(np.var(finite, ddof=1)) if finite.size >= 2 else None
            )
        output.append(aggregate)
    return output


def load_curve_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def causal_continuous_trend(
    times: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
    time_constant: float,
) -> np.ndarray:
    order = np.argsort(times, kind="stable")
    times = times[order]
    values = np.minimum.accumulate(values[order])
    indices = np.searchsorted(times, grid, side="right") - 1
    indices = np.clip(indices, 0, len(times) - 1)
    target = values[indices]
    if time_constant <= 0:
        return target
    trend = np.empty_like(target)
    trend[0] = target[0]
    for index in range(1, len(grid)):
        delta = grid[index] - grid[index - 1]
        weight = 1.0 - math.exp(-delta / time_constant)
        trend[index] = trend[index - 1] + weight * (target[index] - trend[index - 1])
    return np.minimum.accumulate(trend)


def aggregate_curves(
    run_dirs: Dict[int, Path],
    max_runtime: float,
    time_step: float,
    time_constant: float,
) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray], List[Dict[str, object]]]:
    grid = np.round(
        np.arange(0.0, max_runtime + 0.5 * time_step, time_step), 10
    )
    trends: Dict[Tuple[str, str], np.ndarray] = {}
    output_rows: List[Dict[str, object]] = []
    per_seed_rows = {
        seed: load_curve_rows(run_dir / "error_time_curves.csv")
        for seed, run_dir in run_dirs.items()
    }
    per_seed_summary = {}
    for seed, run_dir in run_dirs.items():
        summary_block = read_json(run_dir / "summary.json")
        assert isinstance(summary_block, list)
        per_seed_summary[seed] = {
            str(row["method"]): row for row in summary_block
        }
    for metric_name, (column, _) in CURVE_METRICS.items():
        for method in METHODS:
            method_trends: List[np.ndarray] = []
            for seed in run_dirs:
                full_block = [
                    row
                    for row in per_seed_rows[seed]
                    if row["method"] == method
                ]
                block = [
                    row
                    for row in full_block
                    if float(row["time"]) <= max_runtime + 1e-12
                ]
                if not block:
                    raise RuntimeError("missing %s curve for seed=%d" % (method, seed))
                times = np.asarray([float(row["time"]) for row in block], dtype=float)
                values = np.asarray([float(row[column]) for row in block], dtype=float)
                trend = causal_continuous_trend(
                    times, values, grid, time_constant=time_constant
                )
                status = str(per_seed_summary[seed][method]["status"])
                terminal_row = max(full_block, key=lambda row: float(row["time"]))
                terminal_time = float(terminal_row["time"])
                if status == "converged" and terminal_time <= max_runtime:
                    terminal_value = float(terminal_row[column])
                    terminal_mask = grid >= terminal_time
                    trend[terminal_mask] = terminal_value
                    if not np.all(trend[terminal_mask] == terminal_value):
                        raise RuntimeError(
                            "non-flat post-convergence tail for %s seed=%d"
                            % (method, seed)
                        )
                method_trends.append(trend)
            stacked = np.vstack(method_trends)
            trends[(metric_name, method)] = stacked
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0, ddof=1)
            for time_value, mean_value, std_value in zip(grid, mean, std):
                output_rows.append(
                    {
                        "metric": metric_name,
                        "method": method,
                        "time": float(time_value),
                        "mean": float(mean_value),
                        "std": float(std_value),
                        "n": int(stacked.shape[0]),
                    }
                )
    return grid, trends, output_rows


def plot_metric(
    output_dir: Path,
    metric_name: str,
    grid: np.ndarray,
    trends: Dict[Tuple[str, str], np.ndarray],
    max_runtime: float,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tecator_mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, ylabel = CURVE_METRICS[metric_name]
    fig, axis = plt.subplots(figsize=(10.2, 6.0))
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.13, top=0.80)
    handles = []
    for method in METHODS:
        stacked = trends[(metric_name, method)]
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0, ddof=1)
        line = axis.plot(
            grid,
            mean,
            color=COLORS[method],
            linestyle="-",
            linewidth=2.35,
            label=method,
        )[0]
        handles.append(line)
        axis.fill_between(
            grid,
            mean - std,
            mean + std,
            color=COLORS[method],
            alpha=0.11,
            linewidth=0.0,
        )

    axis.set_xlim(0.0, max_runtime)
    axis.set_xlabel("Running time (seconds)")
    axis.set_ylabel(ylabel)
    axis.set_title("Tecator", pad=11)
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.58)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    fig.legend(
        handles,
        list(METHODS),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=4,
        frameon=False,
        handlelength=3.0,
        columnspacing=2.0,
    )

    stem = "%s_error_vs_time_mean_std" % metric_name
    fig.savefig(output_dir / (stem + ".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / (stem + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def metric_cell(row: Dict[str, object], metric: str) -> str:
    return "%.6f +/- %.6f [%d/10]" % (
        float(row[metric + "_mean"]),
        float(row[metric + "_std"]),
        int(row[metric + "_valid_runs"]),
    )


def write_report(
    output_dir: Path,
    seeds: Sequence[int],
    summary: Sequence[Dict[str, object]],
    audits: Sequence[Dict[str, object]],
    max_runtime: float,
    beta_max_capped: float,
    time_constant: float,
) -> None:
    by_method = {str(row["method"]): row for row in summary}
    lines = [
        "# Tecator overlapping group Lasso: ten seeded splits",
        "",
        "Seeds: %s. Every cell is mean +/- sample standard deviation [valid runs/10]."
        % ", ".join(map(str, seeds)),
        "",
        "| Method | Time | Validation error | Test error | Test error infeasible | Feasibility |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        row = by_method[method]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                method,
                metric_cell(row, "time"),
                metric_cell(row, "validation_error"),
                metric_cell(row, "test_error"),
                metric_cell(row, "test_error_infeasibility"),
                metric_cell(row, "feasibility"),
            )
        )
    cap_times = np.asarray([float(row["cap_time_seconds"]) for row in audits])
    lines.extend(
        [
            "",
            "Protocol notes:",
            "",
            "- Ten different seeded 107/54/54 splits; algorithms are deterministic conditional on each split.",
            "- Each seed uses one common initial x/lambda/r state across all methods.",
            "- Every method has the same %.1f-second algorithm-time budget; budget-end timeout diagnostics are included in the ten-run statistics."
            % max_runtime,
            "- LDPM-CS-C uses beta_max=%.3g and reached the cap in %d/10 runs; mean cap time %.3f s (sample SD %.3f s)."
            % (
                beta_max_capped,
                sum(bool(row["cap_reached"]) for row in audits),
                float(np.mean(cap_times)),
                float(np.std(cap_times, ddof=1)),
            ),
            "- Curves aggregate lower-resolved feasible best-so-far errors on a 0.1-second grid. Each run is converted to a causal continuously decreasing trend with time constant %.1f seconds before the ten-run mean and sample-SD band are computed."
            % time_constant,
            "- If a run converges before the common time limit, its final incumbent is held exactly constant from the actual stopping time onward.",
            "- All four mean curves are solid. Shaded regions are mean +/- one sample standard deviation.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    seeds = parse_seeds(args.seeds)
    output_dir = Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    first_run_dir = Path(args.first_run_dir) if args.first_run_dir else None
    if first_run_dir is not None and not first_run_dir.is_absolute():
        first_run_dir = (OGL_DIR / first_run_dir).resolve()

    run_dirs = run_campaign(output_dir, seeds, first_run_dir, args)
    audits = [
        verify_run(
            run_dirs[seed], seed, args.max_runtime, args.beta_max_capped
        )
        for seed in seeds
    ]
    all_runs = load_all_runs(run_dirs)
    summary = aggregate_summary(all_runs)
    grid, trends, curve_rows = aggregate_curves(
        run_dirs,
        max_runtime=args.max_runtime,
        time_step=args.time_step,
        time_constant=args.trend_time_constant,
    )

    write_csv(output_dir / "all_runs.csv", all_runs)
    write_csv(output_dir / "summary_mean_std.csv", summary)
    write_csv(output_dir / "curve_mean_std.csv", curve_rows)
    (output_dir / "run_audit.json").write_text(
        json.dumps(audits, indent=2, allow_nan=False) + "\n"
    )
    for metric_name in CURVE_METRICS:
        plot_metric(output_dir, metric_name, grid, trends, args.max_runtime)
    write_report(
        output_dir,
        seeds,
        summary,
        audits,
        args.max_runtime,
        args.beta_max_capped,
        args.trend_time_constant,
    )
    protocol = {
        "dataset": "Tecator",
        "seeds": seeds,
        "repetitions": "ten seeded 107/54/54 splits",
        "methods": list(METHODS),
        "ldmma_run": False,
        "common_algorithm_time_budget_seconds": args.max_runtime,
        "beta_max_capped": args.beta_max_capped,
        "curve_checkpoints_per_run": args.curve_checkpoints,
        "curve_grid_seconds": args.time_step,
        "curve_input": "per-run lower-resolved feasible best-so-far validation/test error",
        "curve_display": {
            "trend": "causal exponential relaxation of the right-continuous per-run incumbent",
            "time_constant_seconds": args.trend_time_constant,
            "converged_tail": "snap to the final incumbent at the actual stopping time and carry it forward exactly",
            "line": "ten-run mean; solid for all methods",
            "band": "mean +/- one sample standard deviation (ddof=1)",
            "colors": COLORS,
        },
        "run_directories": {
            str(seed): str(path.resolve()) for seed, path in run_dirs.items()
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("Ten-run aggregate written to %s" % output_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
