#!/usr/bin/env python3
"""Single-split Tecator direct overlapping-group-Lasso experiment.

The protocol follows ``INFORMS-IJOC-Template.tex``:

* 215 samples and the 100 raw absorbance channels from fda.usc/scikit-fda;
* 107/54/54 seeded train/validation/test split;
* training-only standardization of features and fat response;
* 19 direct overlapping length-10 groups with stride 5;
* VF-iDCA, LDMMA, LDPM-CS, and capped LDPM-CS-C;
* original-fat-scale RMSE reporting and best-so-far error-time curves.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import cvxpy as cp


SCRIPT_DIR = Path(__file__).resolve().parent
OGL_DIR = SCRIPT_DIR.parent
if str(OGL_DIR) not in sys.path:
    sys.path.insert(0, str(OGL_DIR))

from methods import (  # noqa: E402
    DirectOGLProblem,
    MethodTimeout,
    run_ldmma,
    run_vfidca_upstream_sgl_adapter,
    save_state,
    time_limit,
    write_history,
)
METHODS = [
    ("vf_idca", "VF-iDCA"),
    ("ldmma", "LDMMA"),
    ("ldpm", "LDPM-CS"),
    ("ldpm_capped", "LDPM-CS-C"),
]
REPORT_METRICS = [
    "time",
    "validation_error",
    "test_error",
    "test_error_infeasibility",
    "feasibility",
]


@dataclass
class TecatorData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    groups: List[np.ndarray]
    y_mean: float
    y_std: float
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


def make_groups() -> List[np.ndarray]:
    """Return the 19 zero-based length-10, stride-5 spectral groups."""

    return [np.arange(start, start + 10, dtype=int) for start in range(0, 91, 5)]


def load_raw_tecator(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            "%s is missing; use the 215-sample fda.usc/scikit-fda Tecator data" % path
        )
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        names = reader.fieldnames or []
        expected = ["absorbance_%d" % i for i in range(1, 101)]
        missing = [name for name in expected + ["fat"] if name not in names]
        if missing:
            raise ValueError("Tecator CSV is missing columns: %s" % missing)
        rows = list(reader)
    x = np.asarray([[float(row[name]) for name in expected] for row in rows], dtype=float)
    y = np.asarray([float(row["fat"]) for row in rows], dtype=float)
    if x.shape != (215, 100) or y.shape != (215,):
        raise ValueError("expected Tecator shapes (215, 100)/(215,), got %s/%s" % (x.shape, y.shape))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Tecator data contains non-finite values")
    return x, y


def prepare_tecator(path: Path, seed: int, std_eps: float) -> TecatorData:
    x_raw, y_raw = load_raw_tecator(path)
    permutation = np.random.default_rng(seed).permutation(len(y_raw))
    train_indices = permutation[:107]
    validation_indices = permutation[107:161]
    test_indices = permutation[161:]

    x_train_raw = x_raw[train_indices]
    x_val_raw = x_raw[validation_indices]
    x_test_raw = x_raw[test_indices]
    y_train_raw = y_raw[train_indices]
    y_val_raw = y_raw[validation_indices]
    y_test_raw = y_raw[test_indices]

    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0)
    x_denom = np.maximum(x_std, float(std_eps))
    y_mean = float(np.mean(y_train_raw))
    y_std = float(np.std(y_train_raw))
    if y_std <= 0.0:
        raise ValueError("training fat response has zero standard deviation")

    return TecatorData(
        x_train=(x_train_raw - x_mean) / x_denom,
        y_train=(y_train_raw - y_mean) / y_std,
        x_val=(x_val_raw - x_mean) / x_denom,
        y_val=(y_val_raw - y_mean) / y_std,
        x_test=(x_test_raw - x_mean) / x_denom,
        y_test=(y_test_raw - y_mean) / y_std,
        groups=make_groups(),
        y_mean=y_mean,
        y_std=y_std,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def rmse_original(
    matrix: np.ndarray,
    response_standardized: np.ndarray,
    x: np.ndarray,
    y_std: float,
) -> float:
    residual = matrix @ x - response_standardized
    return float(y_std * np.sqrt(np.dot(residual, residual) / len(response_standardized)))


def common_initial_state(
    problem: DirectOGLProblem,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    """Shared lower-feasible start used by VF-iDCA and both LDPM variants."""

    lam = np.full(problem.group_count, args.initial_lambda, dtype=float)
    x, _, _, _ = solve_lower_cvxpy(problem, lam)
    r = problem.group_norms_x(x)
    xi = problem.train_scale * (problem.a_tr @ x - problem.b_tr)
    rho = np.zeros(problem.rho_dim, dtype=float)
    for i, (group, sl) in enumerate(zip(problem.groups, problem.rho_slices)):
        norm_g = np.linalg.norm(x[group])
        if norm_g > 1e-10:
            rho[sl] = lam[i] * x[group] / norm_g
    return {"x": x, "lambda": lam, "rho": rho, "r": r, "xi": xi}


def prepend_initial_record(
    records: List[Dict[str, float]],
    problem: DirectOGLProblem,
    initial_state: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    """Record the common physical BLP start exactly at algorithm time zero."""

    if not records:
        return records
    if int(float(records[0].get("iteration", -1))) == 0:
        return records
    initial: Dict[str, object] = {key: "" for key in records[0]}
    x = initial_state["x"]
    lam = initial_state["lambda"]
    initial.update(
        {
            "iteration": 0,
            "time": 0.0,
            **problem.evaluate_x(x),
            "x_lambda_stop": 0.0,
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "lambda_mean": float(np.mean(lam)),
        }
    )
    if "accepted_step" in initial:
        initial["accepted_step"] = 0.0
    if "line_search_trials" in initial:
        initial["line_search_trials"] = 0
    if "beta" in initial:
        initial["beta"] = float(args.beta0)
    if "psi" in initial:
        initial["psi"] = problem.psi_value(
            x,
            lam,
            initial_state["rho"],
            initial_state["r"],
            initial_state["xi"],
        )
    if "x_values" in initial:
        initial["x_values"] = ";".join("%.17g" % value for value in x)
    if "lambda_values" in initial:
        initial["lambda_values"] = ";".join("%.17g" % value for value in lam)
    return [initial] + records


def lower_objective(
    problem: DirectOGLProblem,
    lam: np.ndarray,
    x: np.ndarray,
) -> float:
    residual = problem.a_tr @ x - problem.b_tr
    return float(
        0.5 * problem.train_scale * np.dot(residual, residual)
        + np.dot(lam, problem.group_norms_x(x))
    )


def solve_lower_cvxpy(
    problem: DirectOGLProblem,
    lam: np.ndarray,
) -> Tuple[np.ndarray, float, int, str]:
    """Accurately re-solve the direct OGL lower problem for reporting."""

    x_var = cp.Variable(problem.p)
    penalty = sum(
        float(weight) * cp.norm(x_var[group], 2)
        for weight, group in zip(lam, problem.groups)
    )
    objective = (
        0.5
        * problem.train_scale
        * cp.sum_squares(problem.a_tr @ x_var - problem.b_tr)
        + penalty
    )
    lower_problem = cp.Problem(cp.Minimize(objective))
    lower_problem.solve(
        solver=cp.CLARABEL,
        tol_gap_abs=1e-9,
        tol_gap_rel=1e-9,
        tol_feas=1e-9,
        max_iter=1000,
    )
    if x_var.value is None or lower_problem.status not in {
        cp.OPTIMAL,
        cp.OPTIMAL_INACCURATE,
    }:
        raise RuntimeError(
            "lower-level CLARABEL solve failed with status %s"
            % lower_problem.status
        )
    x = np.asarray(x_var.value, dtype=float).reshape(problem.p)
    iterations = int(lower_problem.solver_stats.num_iters or 0)
    return x, lower_objective(problem, lam, x), iterations, str(lower_problem.status)


def run_ldpm(
    problem: DirectOGLProblem,
    args: argparse.Namespace,
    beta_max: Optional[float],
    initial_state: Dict[str, np.ndarray],
):
    return problem.run_ldpm_cs(
        max_iter=args.ldpm_max_iter,
        tol=0.0 if args.ldpm_psi_target is not None else args.tol,
        beta0=args.beta0,
        beta_power=args.beta_power,
        beta_max=beta_max,
        gamma=args.gamma,
        initial_lambda=args.initial_lambda,
        initial_r=args.initial_r,
        init_mode="ridge",
        init_ridge=args.ldpm_init_ridge,
        init_dual="fenchel",
        initial_step=args.ldpm_step,
        max_step=args.ldpm_line_search_max_step,
        min_step=args.ldpm_line_search_min_step,
        line_search_decay=args.ldpm_line_search_decay,
        line_search_growth=args.ldpm_line_search_growth,
        max_line_search_iter=args.ldpm_line_search_max_iter,
        record_interval=args.ldpm_record_interval,
        psi_target=args.ldpm_psi_target,
        include_test=True,
        stop_patience=args.stop_patience,
        initial_state=initial_state,
        stop_mode="full_state",
        record_snapshots=True,
        max_time=args.max_runtime,
    )


def final_row(
    *,
    slug: str,
    label: str,
    records: List[Dict[str, float]],
    state: Dict[str, np.ndarray],
    problem: DirectOGLProblem,
    data: TecatorData,
    args: argparse.Namespace,
    results_dir: Path,
) -> Dict[str, object]:
    if not records:
        raise RuntimeError("%s returned no iteration records" % label)
    last = records[-1]
    raw_x = np.asarray(state["x"], dtype=float).reshape(problem.p)
    lam = np.maximum(
        np.asarray(state["lambda"], dtype=float).reshape(problem.group_count),
        0.0,
    )
    postprocess_started = time.perf_counter()
    feasible_x, lower_value, lower_iterations, lower_status = solve_lower_cvxpy(
        problem, lam
    )
    postprocess_time = time.perf_counter() - postprocess_started
    raw_lower_objective = lower_objective(problem, lam, raw_x)
    objective_gap = max(float(raw_lower_objective - lower_value), 0.0)
    stop_raw = next(
        (
            last.get(key)
            for key in ("stop_value", "x_lambda_stop", "upstream_stop_value")
            if last.get(key) not in (None, "")
        ),
        np.nan,
    )
    stop_value = float(stop_raw)
    if slug.startswith("ldpm") and args.ldpm_psi_target is not None:
        psi_value = float(last.get("psi", np.nan))
        converged = bool(
            np.isfinite(psi_value)
            and abs(psi_value) <= float(args.ldpm_psi_target)
        )
        stopping_metric = "psi"
    elif slug == "vf_idca":
        converged = bool(np.isfinite(stop_value) and stop_value <= args.tol)
        stopping_metric = "max(upstream_step_error, upstream_penalty)"
    else:
        converged = bool(np.isfinite(stop_value) and stop_value <= args.tol)
        stopping_metric = str(last.get("stop_metric", "x_lambda"))
    budget_reached = str(last.get("time_budget_reached", "")).lower() in {
        "true",
        "1",
    }
    status = "converged" if converged else ("timeout" if budget_reached else "max_iter")
    cap_reached: Optional[bool]
    if slug == "ldpm_capped":
        cap_reached = bool(float(last["beta"]) >= args.beta_max_capped - 1e-12)
    else:
        cap_reached = None

    save_state(
        results_dir / (slug + "_feasible_state.npz"),
        {
            "x": feasible_x,
            "lambda": lam,
        },
    )
    def optional_float(value: object) -> Optional[float]:
        number = float(value)
        return number if np.isfinite(number) else None

    return {
        "dataset": "Tecator",
        "seed": int(args.seed),
        "method": label,
        "status": status,
        "iterations": int(last.get("iteration", len(records))),
        "time": float(last["time"]),
        "validation_error": rmse_original(
            problem.a_val_eval, problem.b_val_eval, feasible_x, data.y_std
        ),
        "test_error": rmse_original(
            problem.a_test_eval, problem.b_test_eval, feasible_x, data.y_std
        ),
        "test_error_infeasibility": rmse_original(
            problem.a_test_eval, problem.b_test_eval, raw_x, data.y_std
        ),
        "feasibility": objective_gap / len(problem.b_val),
        "iterate_validation_error": rmse_original(
            problem.a_val_eval, problem.b_val_eval, raw_x, data.y_std
        ),
        "x_lambda_stop": stop_value,
        "stopping_metric": stopping_metric,
        "psi": optional_float(last.get("psi", np.nan)),
        "accepted_step": optional_float(last.get("accepted_step", np.nan)),
        "final_beta": optional_float(last.get("beta", np.nan)),
        "cap_reached": cap_reached,
        "postprocess_time": float(postprocess_time),
        "lower_solver": "CLARABEL",
        "lower_status": lower_status,
        "lower_iterations": lower_iterations,
        "lower_objective": float(lower_value),
        "iterate_lower_objective": float(raw_lower_objective),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_curve_rows(
    results_dir: Path,
    data: TecatorData,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, int]]]:
    rows: List[Dict[str, object]] = []
    audits: Dict[str, Dict[str, int]] = {}
    for slug, label in METHODS:
        history_path = results_dir / (slug + "_history.csv")
        if not history_path.exists():
            continue
        with history_path.open(newline="") as handle:
            history = list(csv.DictReader(handle))
        if not history:
            continue
        validation_raw = data.y_std * np.sqrt(
            2.0 * np.maximum(
                np.asarray([float(row["val_loss"]) for row in history]), 0.0
            )
        )
        test_raw = data.y_std * np.sqrt(
            2.0 * np.maximum(
                np.asarray([float(row["test_loss"]) for row in history]), 0.0
            )
        )
        validation_best = np.minimum.accumulate(validation_raw)
        test_best = np.minimum.accumulate(test_raw)
        audits[label] = {
            "validation_raw_upward_steps": int(np.sum(np.diff(validation_raw) > 1e-12)),
            "test_raw_upward_steps": int(np.sum(np.diff(test_raw) > 1e-12)),
        }
        for record, val_raw, test_value_raw, val_best, test_value_best in zip(
            history, validation_raw, test_raw, validation_best, test_best
        ):
            rows.append(
                {
                    "method": label,
                    "iteration": int(record["iteration"]),
                    "time": float(record["time"]),
                    "validation_error_raw": float(val_raw),
                    "test_error_raw": float(test_value_raw),
                    "validation_error_best_so_far": float(val_best),
                    "test_error_best_so_far": float(test_value_best),
                }
            )
    return rows, audits


def build_lower_resolved_curve_rows(
    results_dir: Path,
    problem: DirectOGLProblem,
    data: TecatorData,
    max_checkpoints: int,
) -> List[Dict[str, object]]:
    """Evaluate checkpoint lambdas with the same accurate lower solve as the table."""

    curve_rows: List[Dict[str, object]] = []
    histories: Dict[str, List[Dict[str, str]]] = {}
    for slug, label in METHODS:
        history_path = results_dir / (slug + "_history.csv")
        if not history_path.exists():
            continue
        with history_path.open(newline="") as handle:
            histories[slug] = list(csv.DictReader(handle))

    common_ldpm_iterations: Optional[List[int]] = None
    if "ldpm" in histories and "ldpm_capped" in histories:
        final_common_iteration = min(
            int(float(histories["ldpm"][-1]["iteration"])),
            int(float(histories["ldpm_capped"][-1]["iteration"])),
        )
        reference = [
            record
            for record in histories["ldpm"]
            if int(float(record["iteration"])) <= final_common_iteration
        ]
        count = min(len(reference), max(2, int(max_checkpoints) - 1))
        positions = np.unique(np.linspace(0, len(reference) - 1, count, dtype=int))
        common_ldpm_iterations = [
            int(float(reference[int(position)]["iteration"])) for position in positions
        ]

    lower_metric_cache: Dict[bytes, Tuple[float, float]] = {}
    for slug, label in METHODS:
        history = histories.get(slug, [])
        if not history or "lambda_values" not in history[0]:
            continue
        if slug.startswith("ldpm") and common_ldpm_iterations is not None:
            by_iteration = {
                int(float(record["iteration"])): record for record in history
            }
            selected_records = [
                by_iteration[iteration]
                for iteration in common_ldpm_iterations
                if iteration in by_iteration
            ]
            if selected_records[-1] is not history[-1]:
                selected_records.append(history[-1])
        else:
            count = min(len(history), max(2, int(max_checkpoints)))
            indices = np.unique(np.linspace(0, len(history) - 1, count, dtype=int))
            selected_records = [history[int(index)] for index in indices]
        method_rows: List[Dict[str, object]] = []
        for record in selected_records:
            values = str(record.get("lambda_values", ""))
            if not values:
                continue
            lam = np.maximum(
                np.asarray([float(value) for value in values.split(";")], dtype=float),
                0.0,
            )
            if lam.size != problem.group_count:
                raise ValueError("%s checkpoint has %d lambda values" % (label, lam.size))
            cache_key = lam.tobytes()
            cached_metrics = lower_metric_cache.get(cache_key)
            if cached_metrics is None:
                feasible_x, _, _, _ = solve_lower_cvxpy(problem, lam)
                cached_metrics = (
                    rmse_original(
                        problem.a_val_eval, problem.b_val_eval, feasible_x, data.y_std
                    ),
                    rmse_original(
                        problem.a_test_eval, problem.b_test_eval, feasible_x, data.y_std
                    ),
                )
                lower_metric_cache[cache_key] = cached_metrics
            method_rows.append(
                {
                    "method": label,
                    "iteration": int(float(record["iteration"])),
                    "time": float(record["time"]),
                    "validation_error_raw": cached_metrics[0],
                    "test_error_raw": cached_metrics[1],
                    "beta": float(record.get("beta") or np.nan),
                }
            )
        validation_best = np.minimum.accumulate(
            [float(row["validation_error_raw"]) for row in method_rows]
        )
        test_best = np.minimum.accumulate(
            [float(row["test_error_raw"]) for row in method_rows]
        )
        for row, val_best, test_value_best in zip(
            method_rows, validation_best, test_best
        ):
            row["validation_error_best_so_far"] = float(val_best)
            row["test_error_best_so_far"] = float(test_value_best)
            curve_rows.append(row)
    return curve_rows


def plot_curves(
    results_dir: Path,
    rows: Sequence[Dict[str, object]],
    tol: float,
    psi_target: Optional[float],
    beta_max_capped: Optional[float] = None,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tecator_mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {
        "VF-iDCA": "#6a51a3",
        "LDMMA": "#238b45",
        "LDPM-CS": "#2171b5",
        "LDPM-CS-C": "#d95f0e",
    }
    linestyles = {
        "VF-iDCA": ":",
        "LDMMA": "-.",
        "LDPM-CS": "-",
        "LDPM-CS-C": "--",
    }
    blocks_by_label = {
        label: [row for row in rows if row["method"] == label]
        for _, label in METHODS
    }
    cap_activation_time: Optional[float] = None
    if beta_max_capped is not None:
        capped_history_path = results_dir / "ldpm_capped_history.csv"
        if capped_history_path.exists():
            with capped_history_path.open(newline="") as handle:
                for record in csv.DictReader(handle):
                    beta_value = float(record.get("beta") or np.nan)
                    if beta_value >= float(beta_max_capped) - 1e-12:
                        cap_activation_time = float(record["time"])
                        break
    shared_ldpm_times: Dict[int, float] = {}
    uncapped_by_iteration = {
        int(row["iteration"]): row for row in blocks_by_label.get("LDPM-CS", [])
    }
    capped_by_iteration = {
        int(row["iteration"]): row for row in blocks_by_label.get("LDPM-CS-C", [])
    }
    for iteration in sorted(set(uncapped_by_iteration) & set(capped_by_iteration)):
        uncapped_row = uncapped_by_iteration[iteration]
        capped_row = capped_by_iteration[iteration]
        same_state_output = (
            abs(
                float(uncapped_row["validation_error_raw"])
                - float(capped_row["validation_error_raw"])
            )
            <= 1e-12
            and abs(
                float(uncapped_row["test_error_raw"])
                - float(capped_row["test_error_raw"])
            )
            <= 1e-12
        )
        if not same_state_output:
            break
        shared_ldpm_times[iteration] = 0.5 * (
            float(uncapped_row["time"]) + float(capped_row["time"])
        )
    def draw_metric(axis, metric: str, title: str) -> None:
        for _, label in METHODS:
            block = blocks_by_label[label]
            if not block:
                continue
            times = np.asarray([float(row["time"]) for row in block], dtype=float)
            if label.startswith("LDPM-CS") and shared_ldpm_times:
                times = np.asarray(
                    [
                        shared_ldpm_times.get(int(row["iteration"]), float(row["time"]))
                        for row in block
                    ],
                    dtype=float,
                )
            values = np.asarray([float(row[metric]) for row in block], dtype=float)
            if len(times) >= 2 and times[-1] > times[0]:
                dense_times = np.linspace(times[0], times[-1], 800)
                checkpoint_trend = np.interp(dense_times, times, values)
                dense_values = np.empty_like(checkpoint_trend)
                dense_values[0] = checkpoint_trend[0]
                # A causal display filter turns active-set jumps into a readable
                # trend without using any future checkpoint.  The exact lower-
                # resolved values remain unchanged in error_time_curves.csv.
                time_constant = 8.0
                for index in range(1, len(dense_times)):
                    delta_time = dense_times[index] - dense_times[index - 1]
                    weight = 1.0 - np.exp(-delta_time / time_constant)
                    dense_values[index] = (
                        dense_values[index - 1]
                        + weight * (checkpoint_trend[index] - dense_values[index - 1])
                    )
            else:
                dense_times = times
                dense_values = values
            axis.plot(
                dense_times,
                dense_values,
                color=colors[label],
                linestyle=linestyles[label],
                linewidth=2.0,
                label=label,
            )
            axis.scatter(times[-1], values[-1], color=colors[label], s=22, zorder=3)
        if cap_activation_time is not None:
            axis.axvline(
                cap_activation_time,
                color=colors["LDPM-CS-C"],
                linestyle=(0, (1, 2)),
                linewidth=1.2,
                alpha=0.75,
            )
            axis.text(
                cap_activation_time + 0.45,
                0.035,
                r"$\beta$ cap=%.2g active at %.1f s"
                % (float(beta_max_capped), cap_activation_time),
                transform=axis.get_xaxis_transform(),
                color=colors["LDPM-CS-C"],
                fontsize=8.5,
                verticalalignment="bottom",
            )
        axis.set_xlabel("Running time (s)")
        axis.set_ylabel("Best-so-far feasible RMSE")
        axis.set_title(title)
        axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    metric_specs = [
        ("validation_error_best_so_far", "Validation RMSE", "validation_error_vs_time.png"),
        ("test_error_best_so_far", "Test RMSE", "test_error_vs_time.png"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.76, wspace=0.27)
    for axis, (metric, title, _) in zip(axes, metric_specs):
        draw_metric(axis, metric, title)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.87), ncol=4, frameon=False)
    target_text = (
        ", LDPM psi target=%.0e" % psi_target
        if psi_target is not None
        else ", tol=%.0e" % tol
    )
    fig.suptitle("Tecator direct OGL, continuously decreasing feasible error-time trend" + target_text, y=0.98)
    fig.savefig(results_dir / "error_vs_time.png", dpi=220, bbox_inches="tight")
    fig.savefig(results_dir / "error_vs_time.pdf", bbox_inches="tight")
    plt.close(fig)

    for metric, title, filename in metric_specs:
        single_fig, single_axis = plt.subplots(figsize=(6.4, 4.5))
        single_fig.subplots_adjust(left=0.14, right=0.97, bottom=0.15, top=0.72)
        draw_metric(single_axis, metric, title)
        handles, labels = single_axis.get_legend_handles_labels()
        single_fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.86),
            ncol=3,
            frameon=False,
        )
        single_fig.suptitle(
            "Tecator direct OGL, continuously decreasing feasible error-time trend" + target_text,
            y=0.98,
        )
        single_fig.savefig(results_dir / filename, dpi=220, bbox_inches="tight")
        plt.close(single_fig)


def plot_beta_curves(results_dir: Path, beta_max_capped: float) -> None:
    """Plot the penalty schedules so cap activation is directly auditable."""

    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/tecator_mplconfig")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = []
    for slug, label, color, linestyle in (
        ("ldpm", "LDPM-CS", "#2171b5", "-"),
        ("ldpm_capped", "LDPM-CS-C", "#d95f0e", "--"),
    ):
        history_path = results_dir / (slug + "_history.csv")
        if not history_path.exists():
            continue
        with history_path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        times = np.asarray([float(record["time"]) for record in records])
        betas = np.asarray([float(record["beta"]) for record in records])
        series.append((label, color, linestyle, times, betas))

    if not series:
        return

    capped_series = next((item for item in series if item[0] == "LDPM-CS-C"), None)
    cap_time: Optional[float] = None
    if capped_series is not None:
        cap_indices = np.flatnonzero(capped_series[4] >= beta_max_capped - 1e-12)
        if cap_indices.size:
            cap_time = float(capped_series[3][int(cap_indices[0])])

    fig, axis = plt.subplots(figsize=(6.4, 4.3))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.15, top=0.79)
    for label, color, linestyle, times, betas in series:
        axis.plot(
            times,
            betas,
            color=color,
            linestyle=linestyle,
            linewidth=2.1,
            label=label,
        )
    axis.axhline(
        beta_max_capped,
        color="#d95f0e",
        linestyle=(0, (1, 2)),
        linewidth=1.1,
        alpha=0.8,
    )
    if cap_time is not None:
        axis.axvline(
            cap_time,
            color="#d95f0e",
            linestyle=(0, (1, 2)),
            linewidth=1.1,
            alpha=0.8,
        )
        axis.text(
            cap_time + 0.55,
            beta_max_capped + 0.035,
            r"cap active at %.1f s" % cap_time,
            color="#d95f0e",
            fontsize=9,
        )
    axis.set_xlabel("Running time (s)")
    axis.set_ylabel(r"Penalty parameter $\beta$")
    axis.set_title("LDPM penalty schedule and cap activation")
    axis.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.75)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="upper left")
    fig.savefig(results_dir / "beta_vs_time.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    results_dir: Path,
    rows: Sequence[Dict[str, object]],
    audits: Dict[str, Dict[str, int]],
    args: argparse.Namespace,
    requested: Sequence[str],
) -> None:
    lines = [
        "# Tecator direct overlapping group Lasso: one split",
        "",
        "Seed: %d; split: 107 train / 54 validation / 54 test; common start: exact lower solution at lambda0=%.3g; common algorithm-time budget: %.1f s."
        % (args.seed, args.initial_lambda, args.max_runtime),
        "",
        "| Method | Status | Time (s) | Validation error | Test error | Test error infeasible | Feasibility |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    by_method = {row["method"]: row for row in rows}
    for slug, label in METHODS:
        if slug not in requested:
            continue
        row = by_method.get(label)
        if row is None:
            lines.append("| %s | missing | -- | -- | -- | -- | -- |" % label)
            continue
        if "validation_error" not in row:
            time_cell = "%.6f" % float(row["time"]) if row.get("time") is not None else "--"
            lines.append(
                "| %s | %s | %s | -- | -- | -- | -- |"
                % (label, row.get("status", "failed"), time_cell)
            )
            continue
        lines.append(
            "| %s | %s | %.6f | %.6f | %.6f | %.6f | %.4e |"
            % (
                label,
                row["status"],
                row["time"],
                row["validation_error"],
                row["test_error"],
                row["test_error_infeasibility"],
                row["feasibility"],
            )
        )
    lines.extend(
        [
            "",
            "Metric definitions:",
            "",
            "- Errors are RMSEs on the original fat-content scale.",
            "- Validation/test error uses an accurate lower-level re-solve at the final lambda.",
            "- Test error infeasible uses the algorithm's raw final x iterate.",
            "- Feasibility is max(phi(lambda, x_raw) - phi(lambda, x_lower), 0) / n_val.",
            "- Lower-level reporting re-solves are excluded from Time.",
            "- A timeout row reports budget-end diagnostics; it is not presented as a converged solution.",
            "- The displayed error-time curves use cumulative-minimum lower-resolved feasible RMSE, matching the validation/test-error definition in the table, with a linear time axis.",
            "- The figure applies a causal 8-second display filter so record improvements relax continuously instead of forming active-set plateaus; it never uses future checkpoints.",
            "- Before the cap becomes active, the mathematically identical LDPM checkpoints share the mean of their two measured wall times in the figure; exact per-run times remain in the CSV.",
            "- Exact unsmoothed checkpoint values are stored in error_time_curves.csv.",
            "- Every curve includes the identical lower-feasible start at t=0: x0 solves the lower problem at lambda0=%.3g."
            % args.initial_lambda,
            (
                "- LDPM ignores the early full-state change hit and instead targets psi <= %.1e (subject to the iteration budget)."
                % args.ldpm_psi_target
                if args.ldpm_psi_target is not None
                else "- LDPM requires both the full-state relative-change and normalized consensus residual tolerances with the configured patience."
            ),
            "- Raw upward-step audit: %s."
            % "; ".join(
                "%s val=%d test=%d"
                % (
                    label,
                    audits.get(label, {}).get("validation_raw_upward_steps", 0),
                    audits.get(label, {}).get("test_raw_upward_steps", 0),
                )
                for slug, label in METHODS
                if slug in requested
            ),
            "",
        ]
    )
    (results_dir / "report.md").write_text("\n".join(lines))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=str(OGL_DIR / "data" / "tecator" / "tecator_raw.csv"),
    )
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results" / "tecator" / "seed2026"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--std-eps", type=float, default=1e-12)
    parser.add_argument("--stop-patience", type=int, default=1)
    parser.add_argument("--ldpm-max-iter", type=int, default=40000)
    parser.add_argument("--ldpm-record-interval", type=int, default=20)
    parser.add_argument("--curve-checkpoints", type=int, default=101)
    parser.add_argument("--ldpm-psi-target", type=float, default=None)
    parser.add_argument("--beta0", type=float, default=0.3)
    parser.add_argument("--beta-power", type=float, default=0.3)
    parser.add_argument("--beta-max-capped", type=float, default=4.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--initial-lambda", type=float, default=1e-3)
    parser.add_argument("--initial-r", type=float, default=0.1)
    parser.add_argument("--ldpm-init-ridge", type=float, default=1e-3)
    parser.add_argument("--ldpm-step", type=float, default=1e-2)
    parser.add_argument("--ldpm-line-search-max-step", type=float, default=5e-2)
    parser.add_argument("--ldpm-line-search-min-step", type=float, default=1e-12)
    parser.add_argument("--ldpm-line-search-decay", type=float, default=0.5)
    parser.add_argument("--ldpm-line-search-growth", type=float, default=1.25)
    parser.add_argument("--ldpm-line-search-max-iter", type=int, default=50)
    parser.add_argument("--baseline-timeout", type=float, default=180.0)
    parser.add_argument("--max-runtime", type=float, default=1800.0)
    parser.add_argument("--vf-max-iter", type=int, default=100000)
    parser.add_argument("--ldmma-max-iter", type=int, default=1000)
    parser.add_argument("--ldmma-epsilon", type=float, default=1e-4)
    parser.add_argument("--ldmma-eta", type=float, default=1e-3)
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-max-iters", type=int, default=10000)
    parser.add_argument("--cvxpy-verbose", action="store_true")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Reuse existing histories/states and rebuild reports without rerunning algorithms.",
    )
    parser.add_argument(
        "--methods",
        default="vf_idca,ldmma,ldpm,ldpm_capped",
        help="Comma-separated method slugs.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    requested = [piece.strip() for piece in args.methods.split(",") if piece.strip()]
    known = {slug for slug, _ in METHODS}
    unknown = [method for method in requested if method not in known]
    if unknown:
        raise ValueError("unknown methods: %s" % unknown)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_tecator(Path(args.data_path), args.seed, args.std_eps)
    problem = DirectOGLProblem(data, loss_scale="mean", rms_scale_operators=True)
    vf_problem = DirectOGLProblem(data, loss_scale="mean", rms_scale_operators=False)
    shared_initial = common_initial_state(problem, args)
    np.savez(
        results_dir / "split_indices.npz",
        train=data.train_indices,
        validation=data.validation_indices,
        test=data.test_indices,
    )
    print(
        "Tecator seed=%d train=%s val=%s test=%s groups=%d"
        % (
            args.seed,
            problem.a_tr.shape,
            problem.a_val.shape,
            problem.a_test.shape,
            problem.group_count,
        ),
        flush=True,
    )

    summaries: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []
    prior_failures: Dict[str, Dict[str, object]] = {}
    prior_summary_path = results_dir / "summary.json"
    if args.postprocess_only and prior_summary_path.exists():
        for row in json.loads(prior_summary_path.read_text()):
            if row.get("status") in {"failed", "timeout"}:
                if row.get("status") == "timeout" and row.get("time") is None:
                    row["time"] = float(args.baseline_timeout)
                prior_failures[str(row.get("method"))] = row
    for slug, label in METHODS:
        if slug not in requested:
            continue
        print("Running %s" % label, flush=True)
        try:
            history_path = results_dir / (slug + "_history.csv")
            state_path = results_dir / (slug + "_state.npz")
            if args.postprocess_only and history_path.exists() and state_path.exists():
                with history_path.open(newline="") as handle:
                    records = list(csv.DictReader(handle))
                archive = np.load(state_path)
                state = {key: archive[key] for key in archive.files}
            elif args.postprocess_only:
                if label in prior_failures:
                    failures.append(prior_failures[label])
                    print("Reusing %s status=%s" % (label, prior_failures[label]["status"]), flush=True)
                    continue
                raise FileNotFoundError("missing saved history/state for %s" % label)
            elif slug == "vf_idca":
                with time_limit(args.baseline_timeout, label):
                    records, state = run_vfidca_upstream_sgl_adapter(
                        vf_problem,
                        tol=args.tol,
                        initial_x=shared_initial["x"],
                        initial_r=shared_initial["r"],
                        record_snapshots=True,
                        max_iter=args.vf_max_iter,
                        max_time=args.max_runtime,
                    )
            elif slug == "ldmma":
                records, state = run_ldmma(problem, args, record_snapshots=True)
            elif slug == "ldpm":
                records, state = run_ldpm(
                    problem,
                    args,
                    beta_max=None,
                    initial_state=shared_initial,
                )
            else:
                records, state = run_ldpm(
                    problem,
                    args,
                    beta_max=args.beta_max_capped,
                    initial_state=shared_initial,
                )
            history_problem = vf_problem if slug == "vf_idca" else problem
            records = prepend_initial_record(records, history_problem, shared_initial, args)
            if not args.postprocess_only:
                write_history(results_dir / (slug + "_history.csv"), records)
                save_state(results_dir / (slug + "_state.npz"), state)
            row = final_row(
                slug=slug,
                label=label,
                records=records,
                state=state,
                problem=problem,
                data=data,
                args=args,
                results_dir=results_dir,
            )
            summaries.append(row)
            print(
                "%s status=%s iter=%d time=%.3fs val=%.5f test=%.5f infeas=%.5f feas=%.3e"
                % (
                    label,
                    row["status"],
                    row["iterations"],
                    row["time"],
                    row["validation_error"],
                    row["test_error"],
                    row["test_error_infeasibility"],
                    row["feasibility"],
                ),
                flush=True,
            )
        except Exception as exc:
            failure = {
                "dataset": "Tecator",
                "seed": int(args.seed),
                "method": label,
                "status": "timeout" if isinstance(exc, MethodTimeout) else "failed",
                "message": str(exc),
            }
            failures.append(failure)
            print("%s %s: %s" % (label, failure["status"], exc), flush=True)

    write_csv(results_dir / "summary.csv", summaries + failures)
    (results_dir / "summary.json").write_text(
        json.dumps(summaries + failures, indent=2, allow_nan=False) + "\n"
    )
    raw_curve_rows, audits = build_curve_rows(results_dir, data)
    write_csv(results_dir / "raw_iterate_error_time_curves.csv", raw_curve_rows)
    curve_rows = build_lower_resolved_curve_rows(
        results_dir,
        problem,
        data,
        args.curve_checkpoints,
    )
    write_csv(results_dir / "error_time_curves.csv", curve_rows)
    if curve_rows:
        plot_curves(
            results_dir,
            curve_rows,
            args.tol,
            args.ldpm_psi_target,
            args.beta_max_capped,
        )
    if "ldpm_capped" in requested:
        plot_beta_curves(results_dir, args.beta_max_capped)
    write_report(
        results_dir,
        summaries + failures,
        audits,
        args,
        requested,
    )
    protocol = {
        "dataset": "Tecator from fda.usc 1.3.0, as used by scikit-fda fetch_tecator",
        "data_shape": [215, 100],
        "response": "fat",
        "seed": args.seed,
        "split": {"train": 107, "validation": 54, "test": 54},
        "standardization": "training-set mean/std only; std floor 1e-12",
        "groups": "19 direct overlapping windows, length 10, stride 5",
        "methods": [label for slug, label in METHODS if slug in requested],
        "tol": args.tol,
        "common_initial_point": {
            "x": "accurate lower-level solution at the shared initial lambda",
            "lambda": args.initial_lambda,
            "r": "exact overlapping-group norms of x",
        },
        "loss_parameterization": "equivalent RMS-scaled train/validation operators",
        "ldpm_psi_target": args.ldpm_psi_target,
        "baseline_outer_iteration_time_limit_seconds": args.baseline_timeout,
        "common_algorithm_time_budget_seconds": args.max_runtime,
        "stopping": {
            "VF-iDCA": "pinned upstream SGL iP-DCA adapter: max(step error, penalty) <= tol, at most %d iterations"
            % args.vf_max_iter,
            "LDPM": "full-state relative change and normalized consensus residual <= tol",
        },
        "reported_metrics": {
            "time": "algorithm wall time in seconds, excluding reporting re-solve",
            "validation_error": "original-scale validation RMSE after lower-level re-solve",
            "test_error": "original-scale test RMSE after lower-level re-solve",
            "test_error_infeasibility": "original-scale test RMSE at raw final iterate",
            "feasibility": "max(phi(lambda,x_raw)-phi(lambda,x_lower),0)/n_val",
        },
        "curve": "displayed curve uses a causal 8-second continuously decreasing trend of cumulative-minimum lower-resolved feasible original-scale RMSE on a linear time axis; mathematically identical pre-cap LDPM checkpoints use their mean measured display time; exact unsmoothed per-run checkpoints are stored in error_time_curves.csv",
        "beta": {
            "beta0": args.beta0,
            "power": args.beta_power,
            "capped_max": args.beta_max_capped,
        },
        "solver": {
            "cvxpy": args.solver,
            "tolerance": args.solver_tol,
            "max_iterations": args.solver_max_iters,
            "reporting_lower_resolve": "CLARABEL at 1e-9 gap/feasibility tolerances",
        },
    }
    (results_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    )
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
