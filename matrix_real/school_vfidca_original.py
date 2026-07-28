#!/usr/bin/env python3
"""Run the author-original VF-iDCA algorithm on the School bilevel problem.

The VF-iDCA iteration, adaptive penalty update, stopping test, and default
matrix-completion/group-lasso settings follow the author-released
``utils/MCG_Algorithms.py`` implementation.  Only the two CVXPY models are
instantiated for the School weighted-entrywise-l1 plus nuclear-norm problem.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from school_experiment import (
    SchoolData,
    TaskLossOperator,
    feasible_lower_solve,
    load_and_preprocess_school,
    lower_objective,
)


Array = np.ndarray


class SubproblemTimeout(RuntimeError):
    """Raised when one author-original convex subproblem hits its time limit."""


class SubproblemFailure(RuntimeError):
    """Raised when one author-original convex subproblem has no usable solution."""


@dataclass(frozen=True)
class OriginalSettings:
    max_iteration: int = 500
    tol: float = 5e-2
    initial_nuclear_radius: float = 1.0
    initial_entry_radius: float = 0.1
    rho: float = 1e-3
    c: float = 1.0
    beta0: float = 1.0
    delta: float = 5.0
    epsilon: float = 0.0


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _block_operator(task_matrices: Sequence[Array]) -> sp.csc_matrix:
    return sp.block_diag([sp.csr_matrix(matrix) for matrix in task_matrices], format="csc")


def _vectorize_response(task_responses: Sequence[Array]) -> Array:
    return np.concatenate([np.asarray(response, dtype=float) for response in task_responses])


def _solver_timeout_reason(problem: cp.Problem) -> Optional[str]:
    stats = problem.solver_stats
    extra = getattr(stats, "extra_stats", None)
    if not isinstance(extra, dict):
        return None
    info = extra.get("info", {})
    if not isinstance(info, dict):
        return None
    status = str(info.get("status", ""))
    if "time" in status.lower() and "limit" in status.lower():
        return status
    return None


def _solve_scs(
    problem: cp.Problem,
    *,
    eps: float,
    solve_timeout: float,
    label: str,
) -> Tuple[float, str]:
    started = time.perf_counter()
    try:
        problem.solve(
            solver=cp.SCS,
            eps=eps,
            warm_start=True,
            verbose=False,
            time_limit_secs=solve_timeout,
        )
    except Exception as error:
        elapsed = time.perf_counter() - started
        if elapsed >= solve_timeout:
            raise SubproblemTimeout(
                f"{label} exceeded the {solve_timeout:g}s per-solve limit"
            ) from error
        raise SubproblemFailure(f"{label} failed: {error}") from error
    elapsed = time.perf_counter() - started
    timeout_reason = _solver_timeout_reason(problem)
    if timeout_reason is not None or elapsed > solve_timeout + 1.0:
        raise SubproblemTimeout(
            f"{label} hit the {solve_timeout:g}s per-solve limit; "
            f"solver_status={problem.status}; reason={timeout_reason}; elapsed={elapsed:.3f}s"
        )
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise SubproblemFailure(f"{label} returned status={problem.status}")
    return elapsed, str(problem.status)


class SchoolOperators:
    def __init__(self, data: SchoolData) -> None:
        self.d = int(data.train_a[0].shape[1])
        self.tasks = len(data.train_a)
        self.w_size = self.d * self.tasks

        self.train_a = _block_operator(data.train_a)
        self.validation_a = _block_operator(data.validation_a)
        self.test_a = _block_operator(data.test_a)
        self.train_b = _vectorize_response(data.train_b)
        self.validation_b = _vectorize_response(data.validation_b)
        self.test_b = _vectorize_response(data.test_b)
        self.n_train = int(self.train_b.size)
        self.n_validation = int(self.validation_b.size)
        self.n_test = int(self.test_b.size)

    def loss(self, split: str, matrix: Array) -> float:
        operator = getattr(self, f"{split}_a")
        response = getattr(self, f"{split}_b")
        count = int(response.size)
        residual = operator @ matrix.reshape(-1, order="F") - response
        return 0.5 * float(np.dot(residual, residual)) / count

    def mse(self, split: str, matrix: Array) -> float:
        return 2.0 * self.loss(split, matrix)


class DCLower:
    def __init__(self, operators: SchoolOperators, solve_timeout: float) -> None:
        self.operators = operators
        self.solve_timeout = solve_timeout
        self.w = cp.Variable((operators.d, operators.tasks))
        self.r = cp.Parameter(operators.w_size + 1, nonneg=True)
        vector = cp.vec(self.w, order="F")
        residual = operators.train_a @ vector - operators.train_b
        objective = 0.5 / operators.n_train * cp.sum_squares(residual)
        self.constraints = [
            cp.abs(vector) <= self.r[:-1],
            cp.norm(self.w, "nuc") <= self.r[-1],
        ]
        self.problem = cp.Problem(cp.Minimize(objective), self.constraints)

    def solve(self, radius: Array, iteration: int) -> Tuple[float, Array, float, str]:
        self.r.value = radius
        eps = 1e-2 / min(iteration + 1, 100)
        elapsed, status = _solve_scs(
            self.problem,
            eps=eps,
            solve_timeout=self.solve_timeout,
            label=f"iteration {iteration + 1} lower subproblem",
        )
        if self.w.value is None or self.problem.value is None:
            raise SubproblemFailure("lower subproblem returned no primal solution")
        return float(self.problem.value), np.asarray(self.w.value), elapsed, status

    def dual_value(self) -> Array:
        entry_dual = np.asarray(self.constraints[0].dual_value, dtype=float).reshape(-1)
        nuclear_dual = float(self.constraints[1].dual_value)
        return np.concatenate([entry_dual, np.asarray([nuclear_dual])])


class DCApproximated:
    def __init__(
        self,
        operators: SchoolOperators,
        settings: OriginalSettings,
        solve_timeout: float,
    ) -> None:
        self.operators = operators
        self.settings = settings
        self.solve_timeout = solve_timeout
        self.w = cp.Variable((operators.d, operators.tasks))
        self.r = cp.Variable(operators.w_size + 1)
        self.w_k = cp.Parameter((operators.d, operators.tasks))
        self.r_k = cp.Parameter(operators.w_size + 1, nonneg=True)
        self.xi_r = cp.Parameter(operators.w_size + 1)
        self.bias_k = cp.Parameter()
        self.beta_k = cp.Parameter(pos=True)
        self.beta_k.value = settings.beta0

        vector = cp.vec(self.w, order="F")
        train_residual = operators.train_a @ vector - operators.train_b
        validation_residual = operators.validation_a @ vector - operators.validation_b
        train_loss = 0.5 / operators.n_train * cp.sum_squares(train_residual)
        validation_loss = 0.5 / operators.n_validation * cp.sum_squares(validation_residual)
        prox = cp.sum_squares(self.w - self.w_k) + cp.sum_squares(self.r - self.r_k)
        beta_v_k = (
            self.beta_k * train_loss
            + self.xi_r @ self.r
            - self.bias_k
            - self.beta_k * settings.epsilon
        )
        entry_violation = cp.abs(vector) - self.r[:-1]
        nuclear_violation = cp.reshape(
            cp.norm(self.w, "nuc") - self.r[-1], (1,), order="F"
        )
        violation = cp.max(cp.hstack([entry_violation, nuclear_violation]))
        self.beta_penalty = cp.maximum(0, beta_v_k, self.beta_k * violation)
        objective = validation_loss + settings.rho / 2.0 * prox + self.beta_penalty
        self.problem = cp.Problem(cp.Minimize(objective), [self.r >= 0])

    def set_iteration(self, matrix: Array, radius: Array) -> None:
        self.w_k.value = matrix
        self.r_k.value = radius

    def set_value_linearization(
        self, gamma: Array, lower_objective_value: float
    ) -> None:
        self.xi_r.value = gamma * float(self.beta_k.value)
        self.bias_k.value = (
            lower_objective_value * float(self.beta_k.value)
            + float(self.xi_r.value @ self.r_k.value)
        )

    def solve(self, iteration: int) -> Tuple[Array, Array, float, str]:
        eps = 1e-1 / min(iteration + 1, 1000)
        elapsed, status = _solve_scs(
            self.problem,
            eps=eps,
            solve_timeout=self.solve_timeout,
            label=f"iteration {iteration + 1} DC approximated subproblem",
        )
        if self.w.value is None or self.r.value is None:
            raise SubproblemFailure("DC approximated subproblem returned no primal solution")
        return (
            np.asarray(self.w.value),
            np.maximum(np.asarray(self.r.value), 0.0),
            elapsed,
            status,
        )

    def penalty(self) -> float:
        return float(self.beta_penalty.value) / float(self.beta_k.value)

    def update_beta(self, error: float) -> None:
        if error * float(self.beta_k.value) <= self.settings.c * min(
            1.0, float(self.beta_penalty.value)
        ):
            self.beta_k.value = float(self.beta_k.value) + self.settings.delta


def iteration_error(
    matrix: Array,
    radius: Array,
    next_matrix: Array,
    next_radius: Array,
) -> float:
    return math.sqrt(
        float(np.sum(np.square(matrix - next_matrix)))
        + float(np.sum(np.square(radius - next_radius)))
    ) / math.sqrt(
        1.0
        + float(np.sum(np.square(matrix)))
        + float(np.sum(np.square(radius)))
    )


def run_original_vfidca(
    operators: SchoolOperators,
    settings: OriginalSettings,
    solve_timeout: float,
) -> Tuple[List[Dict[str, object]], Dict[str, Array], Dict[str, object]]:
    matrix = np.zeros((operators.d, operators.tasks), dtype=float)
    radius = np.full(operators.w_size + 1, settings.initial_entry_radius, dtype=float)
    radius[-1] = settings.initial_nuclear_radius
    lower = DCLower(operators, solve_timeout)
    approximated = DCApproximated(operators, settings, solve_timeout)
    history: List[Dict[str, object]] = []
    started = time.perf_counter()
    status = "max_iter"
    message = "author-original maximum iteration budget reached"
    gamma = np.zeros(operators.w_size + 1)
    raw_matrix = matrix.copy()
    raw_radius = radius.copy()

    for iteration in range(settings.max_iteration):
        lower_value, lower_matrix, lower_time, lower_status = lower.solve(radius, iteration)
        gamma = lower.dual_value()
        approximated.set_iteration(matrix, radius)
        approximated.set_value_linearization(gamma, lower_value)
        next_matrix, next_radius, approx_time, approx_status = approximated.solve(iteration)
        error = iteration_error(matrix, radius, next_matrix, next_radius)
        penalty = approximated.penalty()
        raw_matrix = next_matrix
        raw_radius = next_radius
        row: Dict[str, object] = {
            "iteration": iteration + 1,
            "time": time.perf_counter() - started,
            "beta": float(approximated.beta_k.value),
            "step_err": error,
            "native_penalty": penalty,
            "validation_mse_infeasible": operators.mse("validation", next_matrix),
            "validation_rmse_infeasible": math.sqrt(
                operators.mse("validation", next_matrix)
            ),
            "lower_time": lower_time,
            "approx_time": approx_time,
            "lower_status": lower_status,
            "approx_status": approx_status,
            "entry_radius_min": float(np.min(next_radius[:-1])),
            "entry_radius_max": float(np.max(next_radius[:-1])),
            "nuclear_radius": float(next_radius[-1]),
        }
        history.append(row)
        print(
            f"VF-iDCA iter={iteration + 1} time={row['time']:.3f}s "
            f"val_rmse_raw={row['validation_rmse_infeasible']:.6f} "
            f"step={error:.3e} penalty={penalty:.3e} "
            f"beta={float(approximated.beta_k.value):.6g} "
            f"solves=({lower_time:.3f}s,{approx_time:.3f}s)",
            flush=True,
        )
        if error < settings.tol and penalty < settings.tol:
            status = "success"
            message = "author-original step and penalty stopping test passed"
            matrix = next_matrix
            radius = next_radius
            break
        approximated.update_beta(error)
        matrix = next_matrix
        radius = next_radius

    elapsed = time.perf_counter() - started
    state = {
        "W_raw": raw_matrix,
        "radius": raw_radius,
        "last_gamma_before_terminal_update": gamma,
    }
    summary: Dict[str, object] = {
        "method": "VF-iDCA-original",
        "status": status,
        "message": message,
        "iterations": len(history),
        "time": elapsed,
        "final_beta": float(approximated.beta_k.value),
        "step_err": float(history[-1]["step_err"]),
        "native_penalty": float(history[-1]["native_penalty"]),
    }
    return history, state, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("school.mat"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "school_vfidca_original_seed2026",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iteration", type=int, default=500)
    parser.add_argument("--solve-timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings = OriginalSettings(max_iteration=args.max_iteration)
    upstream_root = (
        Path(__file__).resolve().parent.parent
        / "third_party"
        / "group_lasso_upstream"
        / "VF-iDCA"
    )
    upstream_algorithm = upstream_root / "VF_iDCA.py"
    upstream_mcg = upstream_root / "utils" / "MCG_Algorithms.py"
    data = load_and_preprocess_school(args.data, args.seed)
    operators = SchoolOperators(data)
    protocol = {
        "dataset": "MALSAR School",
        "seed": args.seed,
        "split": "same task-wise 60/20/20 split and train-only preprocessing as LDPM",
        "tasks": operators.tasks,
        "features": operators.d,
        "radius_hyperparameters": operators.w_size + 1,
        "algorithm": "author-original VF-iDCA Algorithm 1 instantiated for School",
        "upstream_paper": "arXiv:2206.05976",
        "upstream_algorithm_sha256": _sha256(upstream_algorithm),
        "upstream_mcg_sha256": _sha256(upstream_mcg),
        "settings_source": "author-released experiments/MCG_Experiments.py",
        "settings": settings.__dict__,
        "single_subproblem_timeout_seconds": args.solve_timeout,
        "timeout_policy": "stop the entire experiment immediately if either convex subproblem times out",
        "model_adapter_only": True,
        "algorithm_update_modified": False,
        "algorithm_stopping_modified": False,
        "lambda_box_bounds_enforced": False,
        "lambda_note": "original VF-iDCA optimizes radii and recovers nonnegative lambdas as KKT multipliers",
        "test_blind_optimization": True,
        "reported_error": "pooled MSE; RMSE aliases are also reported",
    }
    _save_json(args.output_dir / "protocol.json", protocol)

    try:
        history, state, summary = run_original_vfidca(
            operators, settings, args.solve_timeout
        )
    except (SubproblemTimeout, SubproblemFailure) as error:
        failure = {
            "method": "VF-iDCA-original",
            "status": "timeout" if isinstance(error, SubproblemTimeout) else "failed",
            "message": str(error),
        }
        _save_json(args.output_dir / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True), flush=True)
        return

    _save_csv(args.output_dir / "vf_idca_history.csv", history)

    # Reporting-only final lower solve at the terminal radius. This does not
    # alter VF-iDCA or feed back into its iterations.
    reporting_lower = DCLower(operators, args.solve_timeout)
    final_lower_value, constrained_w, reporting_time, reporting_status = reporting_lower.solve(
        state["radius"], max(summary["iterations"] - 1, 0)
    )
    recovered_lambda = reporting_lower.dual_value()
    raw_w = state["W_raw"]

    training_operator = TaskLossOperator(data.train_a, data.train_b)
    validation_operator = TaskLossOperator(data.validation_a, data.validation_b)
    test_operator = TaskLossOperator(data.test_a, data.test_b)
    lambda_l1 = recovered_lambda[:-1].reshape(
        operators.d, operators.tasks, order="F"
    )
    lambda_nuclear = float(recovered_lambda[-1])
    feasible_w, fixed_lambda_lower = feasible_lower_solve(
        training_operator,
        lambda_l1,
        lambda_nuclear,
        constrained_w,
        rho=0.1,
        max_iter=5000,
        abs_tol=1e-8,
        rel_tol=1e-7,
    )
    raw_objective = lower_objective(
        training_operator, raw_w, lambda_l1, lambda_nuclear
    )
    feasible_objective = float(fixed_lambda_lower["lower_objective"])
    gap = raw_objective - feasible_objective
    validation_mse = validation_operator.mse(feasible_w)
    test_mse = test_operator.mse(feasible_w)
    test_mse_infeasible = test_operator.mse(raw_w)
    summary.update(
        {
            "seed": args.seed,
            "validation_error": validation_mse,
            "test_error": test_mse,
            "test_error_infeasibility": test_mse_infeasible,
            "feasibility": max(gap, 0.0) / validation_operator.n,
            "validation_rmse": math.sqrt(validation_mse),
            "test_rmse": math.sqrt(test_mse),
            "test_rmse_infeasibility": math.sqrt(test_mse_infeasible),
            "raw_lower_objective": raw_objective,
            "feasible_lower_objective": feasible_objective,
            "raw_feasibility_gap": gap,
            "recovered_lambda_l1_min": float(np.min(lambda_l1)),
            "recovered_lambda_l1_max": float(np.max(lambda_l1)),
            "recovered_lambda_nuclear": lambda_nuclear,
            "terminal_radius_entry_min": float(np.min(state["radius"][:-1])),
            "terminal_radius_entry_max": float(np.max(state["radius"][:-1])),
            "terminal_radius_nuclear": float(state["radius"][-1]),
            "reporting_constrained_lower_value": final_lower_value,
            "reporting_constrained_lower_time": reporting_time,
            "reporting_constrained_lower_status": reporting_status,
            "fixed_lambda_lower_status": fixed_lambda_lower["status"],
            "fixed_lambda_lower_iterations": fixed_lambda_lower["iterations"],
            "postprocess_time": reporting_time + float(fixed_lambda_lower["time"]),
        }
    )
    np.savez_compressed(
        args.output_dir / "vf_idca_state.npz",
        W_raw=raw_w,
        W_constrained=constrained_w,
        W_feasible=feasible_w,
        radius=state["radius"],
        lambda_l1=lambda_l1,
        lambda_nuclear=np.asarray(lambda_nuclear),
    )
    _save_json(args.output_dir / "vf_idca_summary.json", summary)
    _save_csv(args.output_dir / "summary.csv", [summary])
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
