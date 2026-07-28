#!/usr/bin/env python3
"""Run and aggregate the five-seed a9a/covtype Group Lasso campaign."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


METHODS = ["vf-idca", "ldmma", "meha", "agils", "ldpm", "ldpm-capped"]
LABELS = {
    "vf-idca": "VF-iDCA",
    "ldmma": "LDMMA",
    "meha": "MEHA",
    "agils": "AGILS",
    "ldpm": "LDPM-PG",
    "ldpm-capped": "LDPM-PG-C",
}
HISTORY_TOKENS = {method: method.replace("-", "_") for method in METHODS}
COLORS = {
    "VF-iDCA": "#E45756",
    "LDMMA": "#54A24B",
    "MEHA": "#B279A2",
    "AGILS": "#F58518",
    "LDPM-PG": "#4C78A8",
    "LDPM-PG-C": "#72B7B2",
}

# Selected once on validation-only pilot seed 2025.  That seed is not used in
# the five reported repetitions.  Capped entries are filled from the separate
# capped pilot search and are intentionally independent of the uncapped ones.
LDPM_CONFIGS = {
    "a9a": {
        "step_size": 0.05,
        "beta0": 1.0,
        "beta_power": 0.0,
        "capped_step_size": 0.05,
        "capped_beta0": 0.5,
        "capped_beta_power": 0.3,
        "beta_max_capped": 1.0,
    },
    "covtype": {
        "step_size": 0.01,
        "beta0": 0.05,
        "beta_power": 0.0,
        "capped_step_size": 0.01,
        "capped_beta0": 0.01,
        "capped_beta_power": 0.3,
        "beta_max_capped": 0.05,
    },
}


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results/group_lasso_uniform_upstream_five_runs",
    )
    parser.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    parser.add_argument("--datasets", default="a9a,covtype")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args(argv)


def tol_token(tol: float) -> str:
    return ("%.0e" % tol).replace("-", "m").replace("+", "")


def run_directory(root: Path, dataset: str, seed: int, tol: float = 1e-5) -> Path:
    return root / dataset / ("seed%d_tol%s" % (seed, tol_token(tol)))


def summary_is_complete(path: Path, methods: Sequence[str]) -> bool:
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return False
    return {row.get("method") for row in rows} == {LABELS[method] for method in methods}


def run_campaign(root: Path, datasets: Iterable[str], seeds: Iterable[int], time_limit: float, resume: bool):
    env = os.environ.copy()
    env["LIBSVMDATA_HOME"] = str(Path("data/libsvmdata").resolve())
    env["PYTHONPYCACHEPREFIX"] = "/private/tmp/ldpm_group_campaign_pycache"
    env["MPLCONFIGDIR"] = "/private/tmp/ldpm_group_campaign_mpl"
    for dataset in datasets:
        for seed in seeds:
            summary_path = run_directory(root, dataset, seed) / "summary.json"
            if resume and summary_is_complete(summary_path, METHODS):
                print("Skipping complete run %s seed=%d" % (dataset, seed), flush=True)
                continue
            command = [
                sys.executable,
                "group_lasso_real_experiment.py",
                "--dataset",
                dataset,
                "--seed",
                str(seed),
                "--tol",
                "1e-5",
                "--methods",
                ",".join(METHODS),
                "--max-iter",
                "100000",
                "--time-limit",
                str(time_limit),
                "--record-interval",
                "500",
                "--baseline-record-interval",
                "100",
                "--stop-patience",
                "1",
                "--beta-max-capped",
                str(LDPM_CONFIGS[dataset]["beta_max_capped"]),
                "--step-size",
                str(LDPM_CONFIGS[dataset]["step_size"]),
                "--beta0",
                str(LDPM_CONFIGS[dataset]["beta0"]),
                "--beta-power",
                str(LDPM_CONFIGS[dataset]["beta_power"]),
                "--capped-step-size",
                str(LDPM_CONFIGS[dataset]["capped_step_size"]),
                "--capped-beta0",
                str(LDPM_CONFIGS[dataset]["capped_beta0"]),
                "--capped-beta-power",
                str(LDPM_CONFIGS[dataset]["capped_beta_power"]),
                "--initial-lambda",
                "1",
                "--ldmma-epsilon",
                "1e-3",
                "--feasible-curve-points",
                "20",
                "--feasible-lower-max-iter",
                "20000",
                "--feasible-lower-tol",
                "1e-9",
                "--results-dir",
                str(root),
            ]
            print("Running %s seed=%d" % (dataset, seed), flush=True)
            subprocess.run(command, check=True, env=env)


def load_runs(
    root: Path,
    datasets: Iterable[str],
    seeds: Iterable[int],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in datasets:
        for seed in seeds:
            path = run_directory(root, dataset, seed) / "summary.json"
            if not path.exists():
                raise FileNotFoundError(path)
            for row in json.loads(path.read_text()):
                if row.get("method") not in {LABELS[method] for method in METHODS}:
                    continue
                output = dict(row)
                output["dataset"] = dataset
                output["seed"] = seed
                output["result_source"] = "uniform upstream-protocol rerun"
                rows.append(output)
    frame = pd.DataFrame(rows)
    expected = len(list(datasets)) * len(list(seeds)) * len(METHODS)
    if len(frame) != expected:
        raise RuntimeError("expected %d run rows, found %d" % (expected, len(frame)))
    return frame


def aggregate_runs(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "time",
        "val_loss",
        "test_loss",
        "val_misclassification",
        "test_misclassification",
        "iterate_val_loss",
        "iterate_test_loss",
        "postprocess_time",
        "iterations",
        "x_lambda_stop",
    ]
    output = []
    for (dataset, method), block in frame.groupby(["dataset", "method"], sort=False):
        row: Dict[str, object] = {
            "dataset": dataset,
            "method": method,
            "runs": int(len(block)),
            "converged_runs": int(np.sum(block["status"] == "converged")),
            "statuses": ";".join(sorted(block["status"].astype(str).unique())),
        }
        for metric in metrics:
            values = pd.to_numeric(block[metric], errors="coerce")
            row[metric + "_mean"] = float(values.mean())
            row[metric + "_variance"] = float(values.var(ddof=1))
        cap_values = block["cap_reached"].dropna()
        row["cap_reached_runs"] = int(np.sum(cap_values.astype(bool))) if len(cap_values) else 0
        output.append(row)
    return pd.DataFrame(output)


def step_interpolate(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(times)
    times = np.asarray(times[order], dtype=float)
    values = np.asarray(values[order], dtype=float)
    keep = np.isfinite(times) & np.isfinite(values)
    times, values = times[keep], values[keep]
    result = np.full(grid.shape, np.nan, dtype=float)
    if not len(times):
        return result
    indices = np.searchsorted(times, grid, side="right") - 1
    valid = indices >= 0
    result[valid] = values[indices[valid]]
    return result


def plot_curves(root: Path, datasets: Sequence[str], seeds: Sequence[int]):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    curve_rows = []
    for dataset in datasets:
        histories: Dict[str, List[pd.DataFrame]] = {LABELS[key]: [] for key in METHODS}
        for seed in seeds:
            directory = run_directory(root, dataset, seed)
            for method in METHODS:
                path = directory / (HISTORY_TOKENS[method] + "_feasible_history.csv")
                if path.exists():
                    frame = pd.read_csv(path)
                    frame["plot_validation_error"] = frame["feasible_validation_error"]
                    frame["plot_test_error"] = frame["feasible_test_error"]
                    histories[LABELS[method]].append(frame)
        finite_times = [
            float(value)
            for method_histories in histories.values()
            for history in method_histories
            for value in history["time"].to_numpy(dtype=float)
            if np.isfinite(value) and value >= 0.0
        ]
        if not finite_times:
            continue
        grid = np.linspace(0.0, max(finite_times), 180)
        for metric, title_token, ylabel in (
            ("plot_validation_error", "validation_error_time", "Validation error (half MSE)"),
            ("plot_test_error", "test_error_time", "Test error (half MSE)"),
        ):
            fig, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
            for label in [LABELS[key] for key in METHODS]:
                interpolated = []
                for history in histories[label]:
                    interpolated.append(
                        step_interpolate(
                            history["time"].to_numpy(dtype=float),
                            history[metric].to_numpy(dtype=float),
                            grid,
                        )
                    )
                if not interpolated:
                    continue
                matrix = np.vstack(interpolated)
                counts = np.sum(np.isfinite(matrix), axis=0)
                mean = np.full(grid.shape, np.nan, dtype=float)
                std = np.full(grid.shape, np.nan, dtype=float)
                available = counts > 0
                repeated = counts > 1
                with np.errstate(invalid="ignore"):
                    mean[available] = np.nanmean(matrix[:, available], axis=0)
                    std[repeated] = np.nanstd(matrix[:, repeated], axis=0, ddof=1)
                valid = counts >= 3
                if not np.any(valid):
                    valid = counts >= 1
                axis.plot(grid[valid], mean[valid], label=label, color=COLORS[label], linewidth=1.8)
                if np.any(counts[valid] >= 2):
                    lower = mean[valid] - np.nan_to_num(std[valid], nan=0.0)
                    upper = mean[valid] + np.nan_to_num(std[valid], nan=0.0)
                    axis.fill_between(grid[valid], lower, upper, color=COLORS[label], alpha=0.12)
                for time_value, mean_value, std_value, count in zip(
                    grid[valid], mean[valid], std[valid], counts[valid]
                ):
                    curve_rows.append(
                        {
                            "dataset": dataset,
                            "metric": metric,
                            "method": label,
                            "time": float(time_value),
                            "mean": float(mean_value),
                            "std": float(std_value) if np.isfinite(std_value) else None,
                            "runs_available": int(count),
                        }
                    )
            axis.set_xlabel("Running time (seconds)", fontsize=8)
            axis.set_ylabel(ylabel, fontsize=8)
            axis.set_title(dataset, fontsize=10)
            axis.tick_params(axis="both", labelsize=7)
            axis.grid(True, alpha=0.22, linewidth=0.6)
            axis.legend(frameon=False, ncol=2, fontsize=7)
            png_path = figure_dir / (dataset + "_" + title_token + ".png")
            pdf_path = figure_dir / (dataset + "_" + title_token + ".pdf")
            fig.savefig(png_path, dpi=220)
            fig.savefig(pdf_path)
            plt.close(fig)
    curve_frame = pd.DataFrame(curve_rows)
    curve_frame.to_csv(root / "error_time_curves.csv", index=False)


def format_mean_variance(mean: float, variance: float) -> str:
    return "%.6g (%.3g)" % (mean, variance)


def write_report(root: Path, summary: pd.DataFrame, seeds: Sequence[int]):
    lines = [
        "# Group Lasso real-data five-run results",
        "",
        "Protocol: seeds %s; all six methods are rerun from the identical physical start "
        "$\\lambda^0=\\mathbf{1}, w^0=\\mathbf{1}$. VF-iDCA, LDMMA and MEHA use their "
        "authors' released code paths specialized only by removing the absent L1 block; "
        "AGILS follows Algorithms 1--2 and the SGL settings in arXiv:2412.18929v5 because "
        "no public author repository was found. LDPM parameters were selected using validation-only "
        "pilot seed 2025, outside the five reported seeds; "
        "first-hit stopping rule $R_k \\le 10^{-5}$ "
        "with no additional feasibility stop. Errors are computed after re-solving the lower problem "
        "at the final hyperparameters for every method; "
        "Parentheses contain the sample variance across five runs."
        % ", ".join(map(str, seeds)),
        "",
    ]
    for dataset in [item for item in ("a9a", "covtype") if item in set(summary["dataset"])]:
        lines.extend(
            [
                "## %s" % dataset,
                "",
                "| Method | Converged | Time mean (var) | Validation error mean (var) | Test error mean (var) |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in summary[summary["dataset"] == dataset].iterrows():
            lines.append(
                "| %s | %d/%d | %s | %s | %s |"
                % (
                    row["method"],
                    row["converged_runs"],
                    row["runs"],
                    format_mean_variance(row["time_mean"], row["time_variance"]),
                    format_mean_variance(row["val_loss_mean"], row["val_loss_variance"]),
                    format_mean_variance(row["test_loss_mean"], row["test_loss_variance"]),
                )
            )
        lines.extend(
            [
                "",
                "Classification-error supplement (misclassification rate):",
                "",
                "| Method | Validation miscl. mean (var) | Test miscl. mean (var) |",
                "|---|---:|---:|",
            ]
        )
        for _, row in summary[summary["dataset"] == dataset].iterrows():
            lines.append(
                "| %s | %s | %s |"
                % (
                    row["method"],
                    format_mean_variance(
                        row["val_misclassification_mean"],
                        row["val_misclassification_variance"],
                    ),
                    format_mean_variance(
                        row["test_misclassification_mean"],
                        row["test_misclassification_variance"],
                    ),
                )
            )
        lines.append("")
    (root / "report.md").write_text("\n".join(lines) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = Path(args.results_dir)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    if len(seeds) != 5:
        raise ValueError("the reporting protocol requires exactly five seeds")
    if not args.aggregate_only:
        run_campaign(root, datasets, seeds, args.time_limit, args.resume)
    runs = load_runs(root, datasets, seeds)
    summary = aggregate_runs(runs)
    runs.to_csv(root / "all_runs.csv", index=False)
    summary.to_csv(root / "summary_mean_variance.csv", index=False)
    plot_curves(root, datasets, seeds)
    write_report(root, summary, seeds)
    protocol = {
        "datasets": datasets,
        "seeds": seeds,
        "methods": [LABELS[key] for key in METHODS],
        "rerun_methods": [LABELS[key] for key in METHODS],
        "stopping_rule": "||x_new-x_old||/max(||x_old||,1) + ||lambda_new-lambda_old||/max(||lambda_old||,1) <= 1e-5",
        "stopping_confirmation": "first hit; no extra patience",
        "baseline_implementation": "VF-iDCA, LDMMA, and MEHA use author-released source specialized only to the pure Group Lasso model and common initialization; AGILS follows arXiv:2412.18929v5 Algorithms 1-2 and Section 6.2 because no public author code was found",
        "error_evaluation": "feasible post-processing: re-solve the lower group-Lasso problem at each sampled lambda; excluded from reported algorithm runtime",
        "initial_lambda": {label: 1.0 for label in LABELS.values()},
        "initial_coef": {label: "all ones" for label in LABELS.values()},
        "ldmma_epsilon": 1e-3,
        "ldpm_validation_only_pilot_seed": 2025,
        "ldpm_configs": LDPM_CONFIGS,
        "nominal_time_guard_seconds": args.time_limit,
        "time_guard_behavior": "A native conic solver call returns before Python can handle SIGALRM; actual wall time is retained and no completed run was truncated.",
        "variance": "sample variance across five runs (ddof=1)",
        "reported_error": "half mean squared prediction error",
        "supplementary_metrics": "validation/test misclassification are retained in all_runs.csv and summary_mean_variance.csv",
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print("Aggregated results written to %s" % root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
