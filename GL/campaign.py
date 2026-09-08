#!/usr/bin/env python3
"""Run and summarize the ten-repetition synthetic Group Lasso campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


ALL_METHODS = [
    "grid",
    "random",
    "tpe",
    "igjo",
    "vf-idca",
    "ldmma",
    "meha",
    "agils",
    "ldpm",
    "ldpm-capped",
]
ITERATIVE_METHODS = {
    "VF-iDCA",
    "LDMMA",
    "MEHA",
    "AGILS",
    "LDPM-PG",
    "LDPM-PG-C",
}
METHOD_ORDER = {
    name: index
    for index, name in enumerate(
        [
            "Grid",
            "Random",
            "TPE",
            "IGJO",
            "VF-iDCA",
            "LDMMA",
            "MEHA",
            "AGILS",
            "LDPM-PG",
            "LDPM-PG-C",
        ]
    )
}


def method_summary_path(root: Path, p: int, seed: int, method: str) -> Path:
    return root / ("p%d" % p) / ("seed%d" % seed) / (
        method.replace("-", "_") + "_summary.json"
    )


def pending_methods(root: Path, p: int, seed: int, methods: Iterable[str]) -> List[str]:
    pending = []
    for method in methods:
        path = method_summary_path(root, p, seed, method)
        if not path.exists():
            pending.append(method)
            continue
        try:
            with path.open() as handle:
                row = json.load(handle)
            if row.get("status") in {"failed", "timeout"}:
                pending.append(method)
        except Exception:
            pending.append(method)
    return pending


def run_one(args, p: int, seed: int, methods: Iterable[str]) -> None:
    methods = list(methods)
    if not methods:
        return
    driver = Path(__file__).resolve().parent / "synthetic.py"
    command = [
        sys.executable,
        str(driver),
        "--p",
        str(p),
        "--seed",
        str(seed),
        "--n-train",
        str(args.n_train),
        "--n-validate",
        str(args.n_validate),
        "--n-test",
        str(args.n_test),
        "--snr",
        str(args.snr),
        "--group-count",
        str(args.group_count),
        "--rng-protocol",
        str(args.rng_protocol),
        "--results-dir",
        str(Path(args.results_dir)),
        "--methods",
        ",".join(methods),
        "--tol",
        "1e-5",
        "--ldpm-stop-metric",
        str(args.ldpm_stop_metric),
        "--stop-patience",
        str(args.stop_patience),
        "--step-size",
        str(args.step_size),
        "--beta0",
        str(args.beta0),
        "--beta-power",
        str(args.beta_power),
        "--capped-step-size",
        str(args.capped_step_size),
        "--capped-beta0",
        str(args.capped_beta0),
        "--capped-beta-power",
        str(args.capped_beta_power),
        "--beta-max-capped",
        str(args.beta_max_capped),
        "--max-iter",
        str(args.max_iter),
        "--record-interval",
        str(args.record_interval),
        "--baseline-record-interval",
        str(args.baseline_record_interval),
        "--grid-points",
        str(args.grid_points),
        "--search-budget",
        str(args.search_budget),
        "--lower-max-iter",
        str(args.lower_max_iter),
        "--lower-tol",
        str(args.lower_tol),
        "--feasible-lower-max-iter",
        str(args.feasible_lower_max_iter),
        "--feasible-lower-tol",
        str(args.feasible_lower_tol),
        "--feasible-lower-solver",
        str(args.feasible_lower_solver),
        "--time-limit",
        str(args.time_limit),
    ]
    if args.overwrite:
        command.append("--overwrite")
    environment = os.environ.copy()
    environment.setdefault("PYTHONPYCACHEPREFIX", "/private/tmp/ldpm_synth_pycache")
    environment.setdefault("SGL_HYPEROPT_PATH", "/private/tmp/ldpm-hyperopt-deps")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    print(
        "CAMPAIGN p=%d seed=%d methods=%s" % (p, seed, ",".join(methods)),
        flush=True,
    )
    subprocess.run(command, check=True, env=environment)


def collect_rows(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(root.glob("p*/seed*/*_summary.json")):
        with path.open() as handle:
            row = json.load(handle)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["method_order"] = frame["method"].map(METHOD_ORDER)
    return frame.sort_values(["p", "method_order", "seed"]).reset_index(drop=True)


def aggregate(raw: pd.DataFrame, dimensions, methods) -> pd.DataFrame:
    selected = raw[
        raw["p"].isin(dimensions)
        & raw["method_key"].isin(methods)
    ].copy()
    metrics = [
        "time",
        "iterations",
        "validation_error",
        "test_error",
        "test_error_infeasibility",
        "feasibility",
    ]
    rows = []
    for (p, method, method_key), group in selected.groupby(
        ["p", "method", "method_key"], sort=False
    ):
        finite_group = group[~group["status"].isin(["failed", "timeout", "nonfinite"])]
        row = {
            "p": int(p),
            "method": method,
            "method_key": method_key,
            "n": int(len(group)),
            "finite_n": int(len(finite_group)),
            "status_counts": ";".join(
                "%s=%d" % (key, value)
                for key, value in sorted(group["status"].value_counts().items())
            ),
            "method_order": METHOD_ORDER[method],
        }
        for metric in metrics:
            metric_group = group if metric in {"time", "iterations"} else finite_group
            values = pd.to_numeric(metric_group[metric], errors="coerce")
            values = values[np.isfinite(values)]
            row[metric + "_mean"] = float(values.mean()) if len(values) else np.nan
            row[metric + "_std"] = (
                float(values.std(ddof=1)) if len(values) >= 2 else np.nan
            )
            row[metric + "_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["p", "method_order"]).reset_index(drop=True)


def pm(mean, std, scientific=False):
    if pd.isna(mean):
        return "-"
    if scientific or (mean != 0.0 and abs(mean) < 1e-3):
        return "%.4e ± %.2e" % (mean, std)
    return "%.4f ± %.4f" % (mean, std)


def write_comparison_markdown(summary: pd.DataFrame, path: Path, args) -> None:
    lines = [
        "# Synthetic pure Group Lasso: ten-run mean ± sample standard deviation",
        "",
        (
            "Only the data generation follows page 29 of arXiv:2412.18929v5. "
            "The pure Group Lasso model has %d equal group-l2 penalties "
            "(%d features per group), no L1 penalty, SNR=%g, and "
            "train/validation/test sizes %d/%d/%d."
            % (
                args.group_count,
                args.comparison_dimensions[0] // args.group_count,
                args.snr,
                args.n_train,
                args.n_validate,
                args.n_test,
            )
        ),
        "",
    ]
    for p in sorted(summary["p"].unique()):
        lines.extend(
            [
                "## p=%d" % p,
                "",
                "| Method | Runs | Finite | Time (s) | Val. Err. | Test Err. | Test Err. Infeas. | Feasibility |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in summary[summary["p"] == p].iterrows():
            iterative = row["method"] in ITERATIVE_METHODS
            lines.append(
                "| %s | %d | %d | %s | %s | %s | %s | %s |"
                % (
                    row["method"],
                    row["n"],
                    row["finite_n"],
                    pm(row["time_mean"], row["time_std"]),
                    pm(row["validation_error_mean"], row["validation_error_std"]),
                    pm(row["test_error_mean"], row["test_error_std"]),
                    pm(
                        row["test_error_infeasibility_mean"],
                        row["test_error_infeasibility_std"],
                    )
                    if iterative
                    else "-",
                    pm(row["feasibility_mean"], row["feasibility_std"], True)
                    if iterative
                    else "-",
                )
            )
        nonfinite = summary[(summary["p"] == p) & (summary["finite_n"] < summary["n"])]
        if len(nonfinite):
            lines.append("")
            lines.append(
                "Non-finite original-method runs: "
                + ", ".join(
                    "%s %d/%d" % (row["method"], row["n"] - row["finite_n"], row["n"])
                    for _, row in nonfinite.iterrows()
                )
                + ". Error means and standard deviations are therefore left undefined."
            )
        lines.append("")
    path.write_text("\n".join(lines))


def write_scaling_markdown(summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# LDPM synthetic pure Group Lasso scaling",
        "",
        "| p | Method | Runs | Finite | Iterations | Time (s) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| %d | %s | %d | %d | %s | %s |"
            % (
                row["p"],
                row["method"],
                row["n"],
                row["finite_n"],
                pm(row["iterations_mean"], row["iterations_std"]),
                pm(row["time_mean"], row["time_std"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/group_lasso")
    parser.add_argument("--phase", choices=["all", "comparison", "scaling", "summarize"], default="comparison")
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    parser.add_argument("--comparison-p-values", default="2400")
    parser.add_argument("--scaling-p-values", default="300,600,1200,2400,3600,4800,6000")
    parser.add_argument("--n-train", type=int, default=1600)
    parser.add_argument("--n-validate", type=int, default=1600)
    parser.add_argument("--n-test", type=int, default=1600)
    parser.add_argument("--snr", type=float, default=2.0)
    parser.add_argument("--group-count", type=int, default=40)
    parser.add_argument(
        "--rng-protocol",
        choices=["default_rng", "paper_legacy"],
        default="default_rng",
    )
    parser.add_argument(
        "--ldpm-stop-metric",
        choices=["full_z", "x_lambda", "tilde_z"],
        default="full_z",
    )
    parser.add_argument("--stop-patience", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta-power", type=float, default=0.3)
    parser.add_argument("--capped-step-size", type=float, default=0.001)
    parser.add_argument("--capped-beta0", type=float, default=1.0)
    parser.add_argument("--capped-beta-power", type=float, default=0.3)
    parser.add_argument("--beta-max-capped", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=150000)
    parser.add_argument("--record-interval", type=int, default=250)
    parser.add_argument("--baseline-record-interval", type=int, default=50)
    parser.add_argument("--grid-points", type=int, default=20)
    parser.add_argument("--search-budget", type=int, default=400)
    parser.add_argument("--lower-max-iter", type=int, default=1000)
    parser.add_argument("--lower-tol", type=float, default=1e-7)
    parser.add_argument("--feasible-lower-max-iter", type=int, default=5000)
    parser.add_argument("--feasible-lower-tol", type=float, default=1e-8)
    parser.add_argument(
        "--feasible-lower-solver",
        choices=["fista", "cvxpy"],
        default="fista",
    )
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    comparison_dimensions = [
        int(item) for item in args.comparison_p_values.split(",") if item.strip()
    ]
    scaling_dimensions = [
        int(item) for item in args.scaling_p_values.split(",") if item.strip()
    ]
    if not comparison_dimensions:
        raise ValueError("--comparison-p-values must contain at least one dimension")
    if any(p % args.group_count for p in comparison_dimensions + scaling_dimensions):
        raise ValueError("every requested p must be divisible by --group-count")
    args.comparison_dimensions = comparison_dimensions
    if len(seeds) != 10:
        raise ValueError("the reported campaign requires exactly ten seeds")

    if args.phase in {"all", "comparison"}:
        for p in comparison_dimensions:
            for seed in seeds:
                methods = ALL_METHODS if args.overwrite else pending_methods(root, p, seed, ALL_METHODS)
                run_one(args, p, seed, methods)
    if args.phase in {"all", "scaling"}:
        for p in scaling_dimensions:
            for seed in seeds:
                methods = ["ldpm", "ldpm-capped"]
                if not args.overwrite:
                    methods = pending_methods(root, p, seed, methods)
                run_one(args, p, seed, methods)

    raw = collect_rows(root)
    if raw.empty:
        raise RuntimeError("no result rows found under %s" % root)
    raw.drop(columns=["method_order"]).to_csv(root / "all_runs.csv", index=False)
    comparison = aggregate(raw, comparison_dimensions, ALL_METHODS)
    comparison.drop(columns=["method_order"]).to_csv(
        root / "comparison_mean_std.csv", index=False
    )
    write_comparison_markdown(comparison, root / "comparison_report.md", args)
    scaling = aggregate(raw, scaling_dimensions, ["ldpm", "ldpm-capped"])
    if not scaling.empty:
        scaling.drop(columns=["method_order"]).to_csv(
            root / "ldpm_scaling_mean_std.csv", index=False
        )
        write_scaling_markdown(scaling, root / "ldpm_scaling_report.md")

    expected_comparison = len(comparison_dimensions) * len(ALL_METHODS) * len(seeds)
    actual_comparison = int(
        raw[
            raw["p"].isin(comparison_dimensions)
            & raw["method_key"].isin(ALL_METHODS)
        ].shape[0]
    )
    expected_scaling = len(scaling_dimensions) * 2 * len(seeds)
    actual_scaling = int(
        raw[
            raw["p"].isin(scaling_dimensions)
            & raw["method_key"].isin(["ldpm", "ldpm-capped"])
        ].shape[0]
    )
    manifest = {
        "comparison_expected_rows": expected_comparison,
        "comparison_actual_rows": actual_comparison,
        "scaling_expected_rows": expected_scaling,
        "scaling_actual_rows": actual_scaling,
        "seeds": seeds,
        "comparison_p_values": comparison_dimensions,
        "scaling_p_values": scaling_dimensions,
        "n_train": args.n_train,
        "n_validate": args.n_validate,
        "n_test": args.n_test,
        "snr": args.snr,
        "group_count": args.group_count,
        "group_size_by_p": {
            str(p): p // args.group_count for p in comparison_dimensions
        },
        "ldpm_stop_metric": args.ldpm_stop_metric,
        "grid_points": args.grid_points,
        "search_budget": args.search_budget,
        "feasible_lower_solver": args.feasible_lower_solver,
        "model": "pure Group Lasso; no L1 regularizer",
        "data_generation": "AGILS page 29 only",
    }
    with (root / "campaign_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

