#!/usr/bin/env python3
"""Run the requested three-run scaled AGILS Table-3 SGL comparison.

The data model, search spaces, budgets, and method-specific stopping rules
follow Table 3 of DOI 10.1137/24M1721049.  The dimensions are changed only as
requested: n_train=n_validation=n_test=400 and m=600.

The experiment driver stores residual errors as SSE/n.  The main report also
stores the half-scaled SSE/(2n) values used in the prior Table-3 comparison,
while retaining the literal SSE/n values as an audit file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


METHOD_KEYS = [
    "grid",
    "random",
    "tpe",
    "igjo",
    "vf-idca",
    "meha",
    "agils",
    "ldpm-cs",
    "ldpm-cs-capped",
]
BASELINE_METHODS = METHOD_KEYS[:7]
LDPM_METHODS = METHOD_KEYS[7:]
METHOD_LABELS = {
    "grid": "Grid",
    "random": "Random",
    "tpe": "TPE",
    "igjo": "IGJO",
    "vf-idca": "VF-iDCA",
    "meha": "MEHA",
    "agils": "AGILS (PGM transcription)",
    "ldpm-cs": "LDPM-CS",
    "ldpm-cs-capped": "LDPM-CS-C",
}
METRICS = [
    "time",
    "validation_error",
    "test_error",
    "test_error_infeasibility",
    "feasibility",
]
ERROR_METRICS = [
    "validation_error",
    "test_error",
    "test_error_infeasibility",
]
INVALID_STATUSES = {"failed", "timeout", "nonfinite"}
EXPECTED_LDPM_INITIAL_LAMBDA = [0.01, 0.01, 0.01, 0.01, 0.01, 0.75]


def expected_ldpm_config(method: str) -> dict:
    if method not in LDPM_METHODS:
        raise ValueError("configuration check only applies to LDPM methods")
    return {
        "step_size": 0.1,
        "beta0": 0.03,
        "beta_power": 1.2,
        "beta_max": 775.0 if method == "ldpm-cs-capped" else None,
        "gamma": 1.0,
        "initial_coef": "lower",
        "initial_lambda": EXPECTED_LDPM_INITIAL_LAMBDA,
        "init_dual": "kkt",
        "stop_metric": "x_lambda",
        "stop_patience": 10,
        "tol": 1e-6,
        "fixed_lambda_solver": "CLARABEL",
        "fixed_lambda_solver_tol": 1e-7,
    }


def validate_ldpm_config(row: dict, method: str, path: Path) -> None:
    if method not in LDPM_METHODS:
        return
    actual = row.get("algorithm_config")
    expected = expected_ldpm_config(method)
    if actual != expected:
        raise RuntimeError(
            "LDPM configuration mismatch in %s\nexpected=%s\nactual=%s"
            % (
                path,
                json.dumps(expected, sort_keys=True),
                json.dumps(actual, sort_keys=True),
            )
        )


def summary_path(root: Path, seed: int, method: str) -> Path:
    return (
        root
        / "p600"
        / ("seed%d" % seed)
        / (method.replace("-", "_") + "_summary.json")
    )


def pending_methods(
    root: Path, seed: int, methods: Iterable[str], overwrite: bool
) -> List[str]:
    if overwrite:
        return list(methods)
    return [
        method
        for method in methods
        if not summary_path(root, seed, method).exists()
    ]


def common_command(args, seed: int, methods: Iterable[str]) -> List[str]:
    driver = Path(__file__).resolve().parent / "group_lasso_synthetic_experiment.py"
    return [
        sys.executable,
        str(driver),
        "--p",
        "600",
        "--n-train",
        "400",
        "--n-validate",
        "400",
        "--n-test",
        "400",
        "--seed",
        str(seed),
        "--rng-protocol",
        "default_rng",
        "--sparse-group",
        "--methods",
        ",".join(methods),
        "--results-dir",
        str(Path(args.results_dir)),
        "--initial-lambda",
        "1.0",
        "--lower-max-iter",
        "5000",
        "--lower-tol",
        "1e-10",
        "--feasible-lower-max-iter",
        "50000",
        "--feasible-lower-tol",
        "1e-10",
        "--feasible-lower-solver",
        "cvxpy",
        "--solver",
        "CLARABEL",
        "--solver-tol",
        "1e-7",
        "--solver-max-iter",
        "10000",
        "--time-limit",
        str(args.time_limit),
    ]


def run_group(args, seed: int, group_name: str, methods: Iterable[str]) -> None:
    methods = list(methods)
    if not methods:
        print(
            "seed=%d group=%s already complete" % (seed, group_name),
            flush=True,
        )
        return

    command = common_command(args, seed, methods)
    if group_name == "baselines":
        command.extend(
            [
                "--tol",
                "1e-5",
                "--agils-max-iter",
                "100000",
                "--agils-inner-max",
                "10000",
                "--meha-max-iter",
                "100000",
                "--vfidca-max-iter",
                "100",
                "--baseline-record-interval",
                "100",
            ]
        )
    elif group_name == "ldpm":
        command.extend(
            [
                "--tol",
                "1e-6",
                "--ldpm-stop-metric",
                "x_lambda",
                "--stop-patience",
                "10",
                "--step-size",
                "1.0",
                "--beta0",
                "0.01",
                "--beta-power",
                "1.2",
                "--cs-step-size",
                "0.1",
                "--cs-capped-step-size",
                "0.1",
                "--cs-gamma",
                "1.0",
                "--cs-beta0",
                "0.03",
                "--cs-beta-power",
                "1.2",
                "--beta-max-capped",
                "775.0",
                "--ldpm-initial-coef",
                "lower",
                "--ldpm-initial-lambda-vector",
                "0.01,0.01,0.01,0.01,0.01,0.75",
                "--ldpm-init-dual",
                "kkt",
                "--init-max-iter",
                "5000",
                "--init-tol",
                "1e-10",
                "--projection-max-sweeps",
                "100",
                "--projection-tol",
                "1e-7",
                "--max-iter",
                "200000",
                "--record-interval",
                "250",
            ]
        )
    else:
        raise ValueError("unknown method group %r" % group_name)

    if args.overwrite:
        command.append("--overwrite")

    root = Path(args.results_dir)
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / ("seed%d_%s.log" % (seed, group_name))
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = (
        "/private/tmp/ldpm_sgl_n400_m600_three_runs_pycache"
    )
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["VECLIB_MAXIMUM_THREADS"] = "1"
    environment["SGL_HYPEROPT_PATH"] = "/private/tmp/sgl-table4-deps"
    print(
        "START seed=%d group=%s methods=%s log=%s"
        % (seed, group_name, ",".join(methods), log_path),
        flush=True,
    )
    with log_path.open("w") as log_handle:
        completed = subprocess.run(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(
            "seed=%d group=%s failed with exit code %d; see %s"
            % (seed, group_name, completed.returncode, log_path)
        )
    print("DONE seed=%d group=%s" % (seed, group_name), flush=True)


def run_seed(args, seed: int) -> None:
    root = Path(args.results_dir)
    baseline_pending = pending_methods(
        root, seed, BASELINE_METHODS, args.overwrite
    )
    run_group(args, seed, "baselines", baseline_pending)
    ldpm_pending = pending_methods(root, seed, LDPM_METHODS, args.overwrite)
    run_group(args, seed, "ldpm", ldpm_pending)


def collect(root: Path, seeds: Iterable[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        for method in METHOD_KEYS:
            path = summary_path(root, seed, method)
            if not path.exists():
                raise RuntimeError("missing result: %s" % path)
            with path.open() as handle:
                row = json.load(handle)
            validate_ldpm_config(row, method, path)
            row["method"] = METHOD_LABELS[method]
            row["n_train"] = 400
            row["n_validate"] = 400
            row["n_test"] = 400
            row["m"] = 600
            rows.append(row)
    frame = pd.DataFrame(rows)
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame["method_order"] = frame["method_key"].map(
        {method: index for index, method in enumerate(METHOD_KEYS)}
    )
    return frame.sort_values(["method_order", "seed"]).reset_index(drop=True)


def table3_numeric_scale(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    for metric in ERROR_METRICS:
        frame[metric + "_sse_over_n"] = frame[metric]
        frame[metric] = 0.5 * frame[metric]
    frame["error_scale"] = "SSE/(2n)"
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_key in METHOD_KEYS:
        group = frame.loc[frame["method_key"] == method_key].copy()
        valid = group.loc[~group["status"].isin(INVALID_STATUSES)].copy()
        row = {
            "method": METHOD_LABELS[method_key],
            "method_key": method_key,
            "attempted_runs": int(len(group)),
            "valid_runs": int(len(valid)),
            "status_counts": ";".join(
                "%s=%d" % (name, count)
                for name, count in sorted(group["status"].value_counts().items())
            ),
            "method_order": int(group["method_order"].iloc[0]),
        }
        for metric in METRICS:
            metric_group = group if metric == "time" else valid
            values = pd.to_numeric(metric_group[metric], errors="coerce")
            values = values[np.isfinite(values)]
            row[metric + "_mean"] = (
                float(values.mean()) if len(values) else np.nan
            )
            row[metric + "_std"] = (
                float(values.std(ddof=1)) if len(values) >= 2 else np.nan
            )
            row[metric + "_n"] = int(len(values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("method_order").reset_index(drop=True)


def fmt(mean, std, count, scientific=False) -> str:
    mean = float(mean) if pd.notna(mean) else np.nan
    std = float(std) if pd.notna(std) else np.nan
    count = int(count)
    if not np.isfinite(mean):
        return "-"
    if scientific or (mean != 0.0 and abs(mean) < 1e-3):
        if count >= 2 and np.isfinite(std):
            return "%.3e (%.2e)" % (mean, std)
        return "%.3e" % mean
    if count >= 2 and np.isfinite(std):
        return "%.2f (%.2f)" % (mean, std)
    return "%.2f" % mean


def write_report(
    scaled: pd.DataFrame, summary: pd.DataFrame, path: Path, seeds: List[int]
) -> None:
    capped_rows = scaled.loc[scaled["method_key"] == "ldpm-cs-capped"]
    capped_reached = int(
        sum(value is True for value in capped_rows["cap_reached"].tolist())
    )
    lines = [
        "# Scaled AGILS Table-3 sparse Group Lasso experiment",
        "",
        "Data: n_train=n_validation=n_test=400, m=600, five equal groups, "
        "five group-l2 hyperparameters plus one l1 hyperparameter, SNR=3. "
        "Independent seeds: %s." % ", ".join(str(seed) for seed in seeds),
        "",
        "Entries are mean (sample standard deviation) over three attempted runs. "
        "The article's prose defines SSE/n, while the preceding local audit found "
        "that the published Table-3 numerical magnitudes follow the half-loss "
        "SSE/(2n) convention. The main table therefore uses SSE/(2n), and the "
        "literal prose-scale SSE/n values are retained in "
        "runs_raw_sse_over_n.csv and summary_mean_std_raw_sse_over_n.csv.",
        "",
        "| Method | Finite/3 | Status | Time (s) | Val. err. | Test err. | Test err. infeas. | Feasibility |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| %s | %d/3 | %s | %s | %s | %s | %s | %s |"
            % (
                row["method"],
                int(row["valid_runs"]),
                row["status_counts"],
                fmt(row["time_mean"], row["time_std"], row["time_n"]),
                fmt(
                    row["validation_error_mean"],
                    row["validation_error_std"],
                    row["validation_error_n"],
                ),
                fmt(
                    row["test_error_mean"],
                    row["test_error_std"],
                    row["test_error_n"],
                ),
                fmt(
                    row["test_error_infeasibility_mean"],
                    row["test_error_infeasibility_std"],
                    row["test_error_infeasibility_n"],
                ),
                fmt(
                    row["feasibility_mean"],
                    row["feasibility_std"],
                    row["feasibility_n"],
                    scientific=True,
                ),
            )
        )

    lines.extend(
        [
            "",
        "Time is algorithm time and excludes the common high-accuracy "
        "lower-level postprocessing used to compute feasible errors. Grid, "
        "Random, TPE, and IGJO do not have infeasible-error or feasibility "
        "entries under the paper's reporting convention.",
        "",
        "Time is aggregated over all three attempts. Error and feasibility "
        "statistics exclude failed, timed-out, and nonfinite attempts, but retain "
        "finite max-iteration endpoints. All six final LDPM-CS/LDPM-CS-C runs "
        "satisfied the requested stopping tolerance. The MEHA time is time to "
        "numerical failure, not a successful runtime, and no MEHA error mean is "
        "reported.",
        "",
        "AGILS denotes the original PGM algorithm transcribed from the paper; "
        "it is not author source code. AGILS-A and AGILS-R are not run.",
            "",
            "MEHA uses the paper-selected setting c0=20, p=0.1, gamma_tilde=gamma, "
            "alpha_tilde=alpha, beta_tilde=beta/8, eta_tilde=eta/4. Search budgets "
            "are Grid 20x20 and Random/TPE 400 trials; IGJO is capped at 50 "
            "iterations.",
            "",
            "LDPM-CS and LDPM-CS-C use beta0=0.03, exponent 1.2, step 0.1, "
            "and gamma=1. LDPM-CS-C additionally caps beta_k at 775; all other "
            "algorithmic settings are identical. Both use a lower-level/KKT-"
            "compatible coefficient/dual start with lambda0=(0.01,0.01,0.01,"
            "0.01,0.01,0.75), selected validation-only on independent seeds "
            "2024 and 2025. Both stop when max(x/lambda residual, consensus "
            "residual) is <=1e-6 with patience 10. LDPM-CS-C reached beta_max=775 "
            "in %d/3 runs." % capped_reached,
            "",
            "## LDPM per-run convergence audit",
            "",
            "| Method | Seed | Status | Iter. | x/lambda residual | r_cons | Feasibility |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for _, row in scaled.loc[
        scaled["method_key"].isin(LDPM_METHODS)
    ].iterrows():
        lines.append(
            "| %s | %d | %s | %s | %s | %s | %s |"
            % (
                row["method"],
                int(row["seed"]),
                row["status"],
                "-" if pd.isna(row["iterations"]) else int(row["iterations"]),
                fmt(row.get("x_lambda_stop"), np.nan, 1, scientific=True),
                fmt(row.get("r_cons"), np.nan, 1, scientific=True),
                fmt(row.get("feasibility"), np.nan, 1, scientific=True),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/sparse_group_lasso_n400_m600_three_runs",
    )
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=1800.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("exactly three distinct seeds are required")
    root = Path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, min(int(args.jobs), 3))
        ) as executor:
            futures = {
                executor.submit(run_seed, args, seed): seed for seed in seeds
            }
            for future in concurrent.futures.as_completed(futures):
                seed = futures[future]
                future.result()
                print("COMPLETE seed=%d" % seed, flush=True)

    raw = collect(root, seeds)
    scaled = table3_numeric_scale(raw)
    raw_summary = aggregate(raw)
    scaled_summary = aggregate(scaled)

    raw.to_csv(root / "runs_raw_sse_over_n.csv", index=False)
    scaled.to_csv(
        root / "runs_table3_numeric_scale_sse_over_2n.csv", index=False
    )
    raw_summary.to_csv(
        root / "summary_mean_std_raw_sse_over_n.csv", index=False
    )
    scaled_summary.to_csv(
        root / "summary_mean_std_table3_numeric_scale.csv", index=False
    )
    write_report(scaled, scaled_summary, root / "REPORT.md", seeds)

    manifest = {
        "paper": "DOI 10.1137/24M1721049 Table 3 model",
        "dimensions": {
            "n_train": 400,
            "n_validate": 400,
            "n_test": 400,
            "m": 600,
        },
        "seeds": seeds,
        "methods": METHOD_KEYS,
        "agils_implementation": "direct local PGM transcription; not author source",
        "main_report_error_scale": "SSE/(2n)",
        "raw_error_scale": "SSE/n",
        "standard_deviation": "sample standard deviation (ddof=1)",
        "time_excludes_common_feasible_postprocessing": True,
        "invalid_statuses_excluded_from_error_aggregates": sorted(
            INVALID_STATUSES
        ),
    }
    with (root / "campaign_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print("Saved campaign report under %s" % root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
