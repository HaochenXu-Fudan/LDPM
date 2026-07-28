#!/usr/bin/env python3
"""Run one AGILS-data synthetic Group/Sparse-Group Lasso experiment.

Only the data generator follows the text on page 29 of arXiv:2412.18929v5.
By default the bilevel problem is the historical five-parameter pure Group
Lasso experiment.  ``--sparse-group`` switches to the final SIAM paper's
Table-3 sparse Group Lasso model: five group-l2 penalties plus one l1 penalty.
The sample sizes and feature dimension default to the published 200/200/200
and 300 setting, but can be scaled explicitly from the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_LOCAL_DATA_HOME = Path(__file__).resolve().parent / "data" / "libsvmdata"
if _LOCAL_DATA_HOME.exists():
    os.environ.setdefault("LIBSVMDATA_HOME", str(_LOCAL_DATA_HOME))

from Data_Generator import Data, Data_with_Info
from group_lasso_baselines import METHOD_RUNNERS, _make_search_lower_solver
from ldpm_core import LeastSquaresLDPM, group_regularizers
from synthetic_group_lasso_problem import MatrixSparseGroupLassoProblem


METHOD_LABELS = {
    "grid": "Grid",
    "random": "Random",
    "tpe": "TPE",
    "vf-idca": "VF-iDCA",
    "ldmma": "LDMMA",
    "meha": "MEHA",
    "agils": "AGILS",
    "igjo": "IGJO",
    "ldpm": "LDPM-PG",
    "ldpm-capped": "LDPM-PG-C",
    "ldpm-cs": "LDPM-CS",
    "ldpm-cs-capped": "LDPM-CS-C",
}


class MethodTimeout(RuntimeError):
    pass


@contextmanager
def time_limit(seconds: float, label: str):
    seconds = float(seconds)
    if seconds <= 0 or not hasattr(signal, "setitimer"):
        yield
        return

    def handler(_signum, _frame):
        raise MethodTimeout("%s reached the %.1f-second time limit" % (label, seconds))

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def parse_methods(raw: str) -> List[str]:
    aliases = {
        "grid": "grid",
        "random": "random",
        "tpe": "tpe",
        "vf-idca": "vf-idca",
        "vfidca": "vf-idca",
        "ldmma": "ldmma",
        "meha": "meha",
        "agils": "agils",
        "igjo": "igjo",
        "ldpm": "ldpm",
        "ldpm-pg": "ldpm",
        "ldpm-capped": "ldpm-capped",
        "ldpm-pg-c": "ldpm-capped",
        "capped": "ldpm-capped",
        "ldpm-cs": "ldpm-cs",
        "ldpm-cs-capped": "ldpm-cs-capped",
        "ldpm-cs-c": "ldpm-cs-capped",
    }
    methods: List[str] = []
    for token in raw.split(","):
        key = token.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError("unknown method %r" % token)
        method = aliases[key]
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def prepare_data(
    p: int,
    seed: int,
    snr: float = 3.0,
    sparse_group: bool = False,
    rng_protocol: str = "default_rng",
    n_train: int = 200,
    n_validate: int = 200,
    n_test: int = 200,
) -> Tuple[Data_with_Info, dict]:
    """Implement the AGILS Table-3 sparse-group data generator."""

    p = int(p)
    n_train = int(n_train)
    n_validate = int(n_validate)
    n_test = int(n_test)
    if p <= 0 or p % 5:
        raise ValueError("p must be a positive multiple of five")
    if p // 5 < 10:
        raise ValueError("each group needs at least ten coordinates")
    if min(n_train, n_validate, n_test) <= 0:
        raise ValueError("train, validation, and test sizes must be positive")
    if rng_protocol == "paper_legacy":
        rng = np.random.RandomState(int(seed))
        standard_normal = lambda *shape: rng.standard_normal(shape)
    elif rng_protocol == "default_rng":
        rng = np.random.default_rng(int(seed))
        standard_normal = lambda *shape: rng.standard_normal(shape)
    else:
        raise ValueError("unknown rng protocol %r" % rng_protocol)
    sample_count = n_train + n_validate + n_test
    matrix = standard_normal(sample_count, p)
    truth = np.zeros(p, dtype=float)
    group_size = p // 5
    support_by_group = []
    for group_index in range(1, 6):
        start = (group_index - 1) * group_size
        active = 2 * group_index
        truth[start : start + active] = 2.0 * group_index
        support_by_group.append(active)
    signal_value = matrix @ truth
    noise = standard_normal(sample_count)
    sigma = float(
        np.linalg.norm(signal_value)
        / max(float(snr) * np.linalg.norm(noise), 1e-15)
    )
    response = signal_value + sigma * noise
    # The paper explicitly says that all generated observations are randomly split.
    # Keep that split for both RNG engines; ``paper_legacy`` only changes the
    # NumPy random-number implementation, not the experimental protocol.
    permutation = rng.permutation(sample_count)
    train_end = n_train
    validation_end = n_train + n_validate
    train_index = permutation[:train_end]
    validation_index = permutation[train_end:validation_end]
    test_index = permutation[validation_end:]

    data = Data()
    data.X_train = matrix[train_index]
    data.X_validate = matrix[validation_index]
    data.X_test = matrix[test_index]
    data.y_train = response[train_index]
    data.y_validate = response[validation_index]
    data.y_test = response[test_index]
    data.true_beta = truth
    data.sigma = sigma
    data.realized_snr = float(
        np.linalg.norm(signal_value) / np.linalg.norm(response - signal_value)
    )
    settings = SimpleNamespace(
        num_train=n_train,
        num_validate=n_validate,
        num_test=n_test,
        num_features=p,
        num_experiment_groups=5,
        num_true_groups=5,
        dataset="synthetic_group_lasso_p%d" % p,
    )
    problem_name = (
        "sparse Group Lasso (five group-l2 penalties plus one l1 penalty)"
        if sparse_group
        else "pure Group Lasso (no l1 regularizer)"
    )
    metadata = {
        "data_protocol": (
            "AGILS final SIAM paper DOI 10.1137/24M1721049, "
            "Table-3 model with explicit user-selected dimensions"
            if sparse_group
            else "AGILS arXiv:2412.18929v5 page 29"
        ),
        "problem": problem_name,
        "seed": int(seed),
        "p": p,
        "n_train": n_train,
        "n_validate": n_validate,
        "n_test": n_test,
        "group_count": 5,
        "group_size": group_size,
        "truth_active_coordinates_by_group": support_by_group,
        "truth_values_by_group": [2, 4, 6, 8, 10],
        "target_snr": float(snr),
        "realized_snr": data.realized_snr,
        "sigma": sigma,
        "split": (
            "random %d/%d/%d permutation from %d legacy-RNG observations"
            % (n_train, n_validate, n_test, sample_count)
            if rng_protocol == "paper_legacy"
            else "random %d/%d/%d permutation from %d generated observations"
            % (n_train, n_validate, n_test, sample_count)
        ),
        "rng_protocol": rng_protocol,
    }
    return Data_with_Info(data, settings, data_index=int(seed)), metadata


def experiment_regularizers(p: int, sparse_group: bool) -> List[dict]:
    regularizers = group_regularizers(p, 5)
    if sparse_group:
        regularizers.append({"type": "l1", "slice": slice(None)})
    return regularizers


def _parse_lambda_values(value: object) -> np.ndarray:
    normalized = str(value).replace(",", ";")
    return np.asarray(
        [float(item) for item in normalized.split(";") if item.strip()],
        dtype=float,
    )


def _make_problem(data_info, regularizers, setting=None):
    return MatrixSparseGroupLassoProblem(data_info, regularizers, setting or {})


def feasible_solution(problem, lam, iterate, args):
    if args.feasible_lower_solver == "cvxpy":
        solve = _make_search_lower_solver(
            problem,
            {
                "search_lower_solver": "cvxpy",
                "solver": args.solver,
                "solver_tol": args.solver_tol,
                "solver_max_iter": args.solver_max_iter,
            },
        )
        feasible, iterations = solve(lam, iterate)
        return feasible, iterations, {
            "solver_requested": str(args.solver),
            "solver_actual": solve.last_solver_name,
            "solver_status": solve.last_status,
            "solver_tol": float(args.solver_tol),
            "solver_max_iter": int(args.solver_max_iter),
            "solver_solve_time": solve.last_solve_time,
        }
    feasible, iterations = problem.lower_solve(
        lam,
        x0=iterate,
        max_iter=args.feasible_lower_max_iter,
        tol=args.feasible_lower_tol,
    )
    return feasible, iterations, {
        "solver_requested": "FISTA",
        "solver_actual": "FISTA",
        "solver_status": "converged_or_iteration_limit",
        "solver_tol": float(args.feasible_lower_tol),
        "solver_max_iter": int(args.feasible_lower_max_iter),
        "solver_solve_time": None,
    }


def paper29_metrics(problem, method, lam, iterate, feasible, args):
    validation_error = 2.0 * problem.validation_loss(feasible)
    test_error = 2.0 * problem.test_loss(feasible)
    if method in {"grid", "random", "tpe", "igjo"}:
        return validation_error, test_error, None, None
    test_error_infeasibility = 2.0 * problem.test_loss(iterate)
    phi_iterate = problem.lower_objective(lam, iterate)
    if method in {"meha", "agils"}:
        gamma = 1.0 / problem.p
        theta, _ = problem.proximal_lower_solve(
            lam,
            iterate,
            gamma,
            max_iter=args.feasible_lower_max_iter,
            tol=args.feasible_lower_tol,
        )
        envelope = problem.lower_objective(lam, theta) + (
            0.5 * np.linalg.norm(theta - iterate) ** 2 / gamma
        )
        raw_feasibility = phi_iterate - envelope
    else:
        raw_feasibility = phi_iterate - problem.lower_objective(lam, feasible)
    feasibility = max(float(raw_feasibility), 0.0) / problem.n_validate
    return validation_error, test_error, test_error_infeasibility, feasibility


def failure_row(
    method: str, status: str, elapsed: float, message: str, p: int, seed: int
) -> dict:
    return {
        "p": int(p),
        "seed": int(seed),
        "method": METHOD_LABELS[method],
        "method_key": method,
        "status": status,
        "time": float(elapsed),
        "iterations": None,
        "validation_error": None,
        "test_error": None,
        "test_error_infeasibility": None,
        "feasibility": None,
        "message": message,
    }


def run_method(data_info, args, method: str, run_dir: Path) -> Dict[str, object]:
    label = METHOD_LABELS[method]
    if method == "agils" and args.sparse_group:
        label = "AGILS (PGM)"
    p = int(data_info.settings.num_features)
    regularizers = experiment_regularizers(p, args.sparse_group)
    parameter_count = len(regularizers)
    common_initial_coef = np.ones(p, dtype=float)
    ldpm_methods = {"ldpm", "ldpm-capped", "ldpm-cs", "ldpm-cs-capped"}
    capped_methods = {"ldpm-capped", "ldpm-cs-capped"}
    cs_methods = {"ldpm-cs", "ldpm-cs-capped"}
    is_capped = method in capped_methods
    is_cs = method in cs_methods
    if args.ldpm_initial_lambda_vector:
        initial_lambda = _parse_lambda_values(args.ldpm_initial_lambda_vector)
        if initial_lambda.size != parameter_count:
            raise ValueError(
                "--ldpm-initial-lambda-vector needs %d values, got %d"
                % (parameter_count, initial_lambda.size)
            )
        if np.any(initial_lambda < 0.0):
            raise ValueError("--ldpm-initial-lambda-vector must be nonnegative")
    else:
        initial_lambda = np.full(parameter_count, args.initial_lambda)
    beta_max = args.beta_max_capped if is_capped else None
    if is_cs:
        step_size = (
            args.cs_capped_step_size if is_capped else args.cs_step_size
        )
    else:
        step_size = args.capped_step_size if is_capped else args.step_size
    if is_cs:
        beta0 = args.cs_beta0
        beta_power = args.cs_beta_power
    else:
        beta0 = args.capped_beta0 if is_capped else args.beta0
        beta_power = args.capped_beta_power if is_capped else args.beta_power
    ldpm_setting = {
        "step_size": step_size,
        "MAX_ITERATION": args.max_iter,
        "TOL": args.tol,
        "beta0": beta0,
        "beta_power": beta_power,
        "beta_max": beta_max,
        "initial_lambda": initial_lambda,
        "normalize_loss": True,
        "sqrt_loss_scaling": True,
        "reduced_dual": False,
        "init_max_iter": args.init_max_iter,
        "init_tol": args.init_tol,
        "init_dual": args.ldpm_init_dual,
        "stop_metric": args.ldpm_stop_metric,
        "record_interval": args.record_interval,
        "stop_patience": args.stop_patience,
        "gamma": args.cs_gamma,
        "projection_max_sweeps": args.projection_max_sweeps,
        "projection_tol": args.projection_tol,
        "line_search": True,
        "line_search_max_step": step_size,
        "line_search_min_step": 1e-12,
        "line_search_decay": 0.5,
        "line_search_growth": 1.25,
        "max_line_search_iter": 50,
    }
    algorithm_config = None
    if method in ldpm_methods:
        algorithm_config = {
            "step_size": float(step_size),
            "beta0": float(beta0),
            "beta_power": float(beta_power),
            "beta_max": None if beta_max is None else float(beta_max),
            "gamma": float(args.cs_gamma) if is_cs else None,
            "initial_coef": str(args.ldpm_initial_coef),
            "initial_lambda": np.asarray(initial_lambda, dtype=float).tolist(),
            "init_dual": str(args.ldpm_init_dual),
            "stop_metric": str(args.ldpm_stop_metric),
            "stop_patience": int(args.stop_patience),
            "tol": float(args.tol),
            "fixed_lambda_solver": str(args.solver),
            "fixed_lambda_solver_tol": float(args.solver_tol),
        }
    if args.ldpm_initial_coef == "ones":
        ldpm_setting["initial_coef"] = common_initial_coef.copy()
    started = time.perf_counter()
    try:
        with time_limit(args.time_limit, label):
            if method in ldpm_methods:
                solver = LeastSquaresLDPM(data_info, regularizers, ldpm_setting)
                history = solver.run_admm() if is_cs else solver.run_pgm()
            elif method == "igjo":
                from HC_SGL import SGL_Hillclimb

                hc_setting = {
                    "num_iters": 50,
                    "step_size_min": 1e-6,
                    "decr_enough_threshold": 5e-4,
                }
                hc_solver = SGL_Hillclimb(
                    data_info.data, data_info.settings, hc_setting
                )
                hc_solver.run(
                    [np.ones(parameter_count, dtype=float)],
                    debug=False,
                    log_file=None,
                )
                history = hc_solver.monitor.to_df()
                if len(history) == 0 or hc_solver.fmodel.best_model_params is None:
                    raise RuntimeError("IGJO returned no finite iterate")
                history["iteration"] = np.arange(len(history), dtype=int)
                history["x_lambda_stop"] = np.nan
                history.attrs["coef"] = np.concatenate(
                    [
                        np.asarray(block, dtype=float).reshape(-1)
                        for block in hc_solver.fmodel.best_model_params
                    ]
                )
                history.attrs["lambda"] = np.asarray(
                    hc_solver.fmodel.best_lambdas, dtype=float
                )
                history.attrs["termination_status"] = "complete"
            else:
                initial_coef = common_initial_coef.copy()
                initial_lambda = args.initial_lambda
                initial_radius = None
                if method == "vf-idca":
                    initial_coef = np.zeros(p)
                    initial_radius = np.ones(parameter_count)
                elif method == "ldmma":
                    initial_coef = np.zeros(p)
                    initial_lambda = 5.0
                    initial_radius = np.full(parameter_count, 0.1)
                baseline_setting = {
                    "time_origin": started,
                    "tol": args.tol,
                    "max_iter": {
                        "vf-idca": args.vfidca_max_iter,
                        "ldmma": args.ldmma_max_iter,
                        "meha": args.meha_max_iter,
                        "agils": args.agils_max_iter,
                    }.get(method, args.max_iter),
                    "record_interval": args.baseline_record_interval,
                    "initial_lambda": initial_lambda,
                    "initial_radius": initial_radius,
                    "initial_coef": initial_coef,
                    "lower_max_iter": args.lower_max_iter,
                    "lower_tol": args.lower_tol,
                    "solver": args.solver,
                    "solver_tol": args.solver_tol,
                    "solver_max_iter": args.solver_max_iter,
                    "lambda_floor": 1e-8,
                    "lambda_ceiling": np.inf if method == "agils" else 10.0,
                    "vfidca_rho": 0.01,
                    "vfidca_beta0": 5.0,
                    "vfidca_beta_delta": 0.1,
                    "vfidca_c": 1.0,
                    "vfidca_epsilon": 0.0,
                    "vfidca_violation_weight": 100.0,
                    "vfidca_solver_tol": args.vfidca_solver_tol,
                    "vfidca_solver_max_iter": args.vfidca_solver_max_iter,
                    "ldmma_epsilon": 1e-3,
                    "ldmma_eta": 0.0,
                    "ldmma_solver_tol": args.ldmma_solver_tol,
                    "ldmma_solver_max_iter": args.ldmma_solver_max_iter,
                    "moreau_gamma": 1.0 / p,
                    "meha_c0": 20.0,
                    "meha_c_power": 0.1,
                    "agils_epsilon": 1e-6,
                    "agils_penalty0": 6.0,
                    "agils_penalty_increment": 0.01,
                    "agils_cp": 1.0,
                    # The paper does not report a numerical c_y and states
                    # that feasibility correction was never triggered in any
                    # sparse-group run.  Infinity reproduces that reported
                    # execution path; c_y_tilde=50*sqrt(m) is a different
                    # inner-correction constant.
                    "agils_cy": np.inf if args.sparse_group else 50.0 * np.sqrt(p),
                    "agils_feasibility_tol": 0.1,
                    "agils_inner_max": args.agils_inner_max,
                    "agils_s0": 5.0,
                    "agils_s_power": 1.05,
                    "agils_tau0": 10.0,
                    "agils_tau_power": 0.2,
                    "agils_gamma": 1.0 / p,
                    "seed": args.seed,
                    "grid_points": 20,
                    "search_budget": 400,
                    "search_lower_solver": "cvxpy",
                    "paper29_protocol": True,
                }
                problem = _make_problem(data_info, regularizers, baseline_setting)
                agils_beta = 1.0 / (
                    problem.val_lipschitz / 6.0 + problem.train_lipschitz + 0.1
                )
                baseline_setting.update(
                    meha_alpha=(1.0 / 1.1) / args.meha_alpha_divisor,
                    meha_beta=agils_beta / args.meha_beta_divisor,
                    meha_eta=(1.0 / (problem.train_lipschitz + p))
                    / args.meha_eta_divisor,
                )
                history = METHOD_RUNNERS[method](problem, baseline_setting)
        elapsed = time.perf_counter() - started
    except MethodTimeout as exc:
        row = failure_row(
            method, "timeout", args.time_limit, str(exc), p, args.seed
        )
        row["method"] = label
        if algorithm_config is not None:
            row["algorithm_config"] = algorithm_config
        return row
    except Exception as exc:
        row = failure_row(
            method,
            "failed",
            time.perf_counter() - started,
            "%s: %s" % (type(exc).__name__, exc),
            p,
            args.seed,
        )
        row["method"] = label
        if algorithm_config is not None:
            row["algorithm_config"] = algorithm_config
        return row

    if len(history) == 0:
        row = failure_row(
            method, "failed", elapsed, "method returned no history", p, args.seed
        )
        row["method"] = label
        if algorithm_config is not None:
            row["algorithm_config"] = algorithm_config
        return row
    final_history_time = float(history.iloc[-1]["time"])
    if final_history_time > 0.0:
        history["time"] = history["time"] * (elapsed / final_history_time)
    history.to_csv(run_dir / (method.replace("-", "_") + "_history.csv"), index=False)
    iterate = np.asarray(history.attrs["coef"], dtype=float)
    native_lam = np.asarray(history.attrs["lambda"], dtype=float)
    # The CS consensus center can be O(tol) outside the nonnegative cone even
    # when every local copy is feasible.  Hyperparameters passed to the common
    # lower-level postprocessor must lie in the model domain.
    lam = np.maximum(native_lam, 0.0)
    post_started = time.perf_counter()
    metric_problem = _make_problem(
        data_info,
        regularizers,
        {"lower_max_iter": args.feasible_lower_max_iter, "lower_tol": args.feasible_lower_tol},
    )
    feasible, lower_iterations, refit_info = feasible_solution(
        metric_problem, lam, iterate, args
    )
    postprocess_time = time.perf_counter() - post_started
    metrics = paper29_metrics(metric_problem, method, lam, iterate, feasible, args)
    np.savez(
        run_dir / (method.replace("-", "_") + "_state.npz"),
        coef=feasible,
        iterate_coef=iterate,
        lambda_value=lam,
        true_beta=np.asarray(data_info.data.true_beta),
    )
    last = history.iloc[-1]
    final_beta = float(last["beta"]) if "beta" in last else None
    cap_reached = bool(
        beta_max is not None
        and final_beta is not None
        and final_beta >= float(beta_max) - 1e-12
    ) if beta_max is not None else None
    row = {
        "p": p,
        "seed": int(args.seed),
        "method": label,
        "method_key": method,
        "status": str(history.attrs.get("termination_status", "complete")),
        "time": float(elapsed),
        "iterations": int(last["iteration"]),
        "validation_error": float(metrics[0]),
        "test_error": float(metrics[1]),
        "test_error_infeasibility": None if metrics[2] is None else float(metrics[2]),
        "feasibility": None if metrics[3] is None else float(metrics[3]),
        "x_lambda_stop": (
            float(last["x_lambda_stop"]) if "x_lambda_stop" in last else None
        ),
        "postprocess_time": float(postprocess_time),
        "feasible_lower_iterations": int(lower_iterations),
        "fixed_lambda_refit": refit_info,
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "native_lambda_min": float(np.min(native_lam)),
        "final_beta": final_beta,
        "cap_reached": cap_reached,
        "accepted_step_size": float(last["accepted_step_size"])
        if "accepted_step_size" in last else None,
        "message": "",
    }
    if algorithm_config is not None:
        row["algorithm_config"] = algorithm_config
    for column in (
        "r_stat",
        "r_cons",
        "packed_relative_step",
        "full_z_relative_step",
        "stop_metric",
        "ll_duality_gap",
        "ll_feasibility",
        "paper_stop_metric",
        "native_violation",
        "native_fenchel_gap",
        "native_constraint_violation",
        "moreau_residual",
        "theta_pg_residual",
        "penalty_parameter",
    ):
        row[column] = float(last[column]) if column in last else None
    for key, value in list(row.items()):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            row[key] = None
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-validate", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--rng-protocol",
        choices=["default_rng", "paper_legacy"],
        default="default_rng",
    )
    parser.add_argument(
        "--methods",
        default=(
            "grid,random,tpe,igjo,vf-idca,ldmma,meha,agils,"
            "ldpm,ldpm-capped,ldpm-cs,ldpm-cs-capped"
        ),
    )
    parser.add_argument(
        "--sparse-group",
        action="store_true",
        help="use the final AGILS Table-3 SGL model with an additional l1 penalty",
    )
    parser.add_argument("--results-dir", default="results/group_lasso_synthetic_paper29")
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument(
        "--ldpm-stop-metric",
        choices=["full_z", "x_lambda", "tilde_z"],
        default="full_z",
        help=(
            "LDPM stationarity residual; CS always additionally requires its "
            "consensus residual"
        ),
    )
    parser.add_argument("--stop-patience", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta-power", type=float, default=0.3)
    parser.add_argument("--capped-step-size", type=float, default=0.001)
    parser.add_argument("--cs-step-size", type=float, default=0.001)
    parser.add_argument("--cs-capped-step-size", type=float, default=0.001)
    parser.add_argument("--cs-gamma", type=float, default=10.0)
    parser.add_argument("--cs-beta0", type=float, default=1.0)
    parser.add_argument("--cs-beta-power", type=float, default=0.3)
    parser.add_argument("--projection-max-sweeps", type=int, default=100)
    parser.add_argument("--projection-tol", type=float, default=1e-7)
    parser.add_argument("--capped-beta0", type=float, default=1.0)
    parser.add_argument("--capped-beta-power", type=float, default=0.3)
    parser.add_argument("--beta-max-capped", type=float, default=10.0)
    parser.add_argument("--initial-lambda", type=float, default=1.0)
    parser.add_argument(
        "--ldpm-initial-lambda-vector",
        help=(
            "optional comma- or semicolon-separated LDPM lambda0 vector; "
            "overrides --initial-lambda for LDPM methods"
        ),
    )
    parser.add_argument(
        "--ldpm-initial-coef",
        choices=["ones", "lower"],
        default="ones",
        help="initialize the LDPM coefficient at ones or at x*(lambda0)",
    )
    parser.add_argument(
        "--ldpm-init-dual",
        choices=["zero", "fenchel", "kkt"],
        default="fenchel",
    )
    parser.add_argument("--max-iter", type=int, default=100000)
    parser.add_argument("--record-interval", type=int, default=250)
    parser.add_argument("--baseline-record-interval", type=int, default=25)
    parser.add_argument("--init-max-iter", type=int, default=300)
    parser.add_argument("--init-tol", type=float, default=1e-7)
    parser.add_argument("--time-limit", type=float, default=900.0)
    parser.add_argument("--lower-max-iter", type=int, default=5000)
    parser.add_argument("--lower-tol", type=float, default=1e-10)
    parser.add_argument("--feasible-lower-max-iter", type=int, default=50000)
    parser.add_argument("--feasible-lower-tol", type=float, default=1e-10)
    parser.add_argument(
        "--feasible-lower-solver",
        choices=["fista", "cvxpy"],
        default="fista",
    )
    parser.add_argument("--solver", choices=["CLARABEL", "SCS"], default="CLARABEL")
    parser.add_argument("--solver-tol", type=float, default=1e-7)
    parser.add_argument("--solver-max-iter", type=int, default=10000)
    parser.add_argument("--vfidca-max-iter", type=int, default=100)
    parser.add_argument("--vfidca-solver-tol", type=float, default=1e-4)
    parser.add_argument("--vfidca-solver-max-iter", type=int, default=100)
    parser.add_argument("--ldmma-max-iter", type=int, default=100)
    parser.add_argument("--ldmma-solver-tol", type=float, default=1e-2)
    parser.add_argument("--ldmma-solver-max-iter", type=int, default=50)
    parser.add_argument("--meha-max-iter", type=int, default=100000)
    parser.add_argument("--meha-alpha-divisor", type=float, default=1.0)
    parser.add_argument("--meha-beta-divisor", type=float, default=8.0)
    parser.add_argument("--meha-eta-divisor", type=float, default=4.0)
    parser.add_argument("--agils-max-iter", type=int, default=100000)
    parser.add_argument("--agils-inner-max", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    data_info, metadata = prepare_data(
        args.p,
        args.seed,
        sparse_group=args.sparse_group,
        rng_protocol=args.rng_protocol,
        n_train=args.n_train,
        n_validate=args.n_validate,
        n_test=args.n_test,
    )
    run_dir = Path(args.results_dir) / ("p%d" % args.p) / ("seed%d" % args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Data: %s p=%d groups=5 train/val/test=%d/%d/%d SNR=%.12g"
        % (
            "sparse Group Lasso" if args.sparse_group else "pure Group Lasso",
            args.p,
            args.n_train,
            args.n_validate,
            args.n_test,
            metadata["realized_snr"],
        ),
        flush=True,
    )
    for method in methods:
        summary_path = run_dir / (method.replace("-", "_") + "_summary.json")
        if summary_path.exists() and not args.overwrite:
            print("Skipping existing %s" % method, flush=True)
            continue
        print("Running %s" % method, flush=True)
        row = run_method(data_info, args, method, run_dir)
        with summary_path.open("w") as handle:
            json.dump(row, handle, indent=2, sort_keys=True, allow_nan=False)
        print(json.dumps(row, sort_keys=True), flush=True)

    metadata["methods_requested"] = methods
    metadata["protocol"] = {
        "comparison_stops": "AGILS Table-3 method-specific rules; LDMMA released native stop",
        "agils_feasibility_correction": (
            "paper-reported execution path: never triggered; numerical c_y is not disclosed"
            if args.sparse_group
            else "enabled"
        ),
        "ldpm_stop_metric": args.ldpm_stop_metric,
        "ldpm_stop_patience": args.stop_patience,
        "ldpm_initial_coef": args.ldpm_initial_coef,
        "ldpm_init_dual": args.ldpm_init_dual,
        "ldpm_initial_lambda": (
            _parse_lambda_values(args.ldpm_initial_lambda_vector).tolist()
            if args.ldpm_initial_lambda_vector
            else float(args.initial_lambda)
        ),
        "ldpm_pg_stop": "%s <= %.12g" % (args.ldpm_stop_metric, args.tol),
        "ldpm_cs_stop": "max(%s,r_cons) <= %.12g"
        % (args.ldpm_stop_metric, args.tol),
        "ldpm_line_search": "backtracking with decay 0.5 and growth 1.25",
        "search": (
            "Grid 20x20 tied-group/l1 weights; Random and TPE 400 six-dimensional trials"
            if args.sparse_group
            else "Grid 20 common weights; Random and TPE 400 five-dimensional trials"
        ),
        "initialization": {
            "LDPM": {
                "coefficient": args.ldpm_initial_coef,
                "lambda": (
                    _parse_lambda_values(args.ldpm_initial_lambda_vector).tolist()
                    if args.ldpm_initial_lambda_vector
                    else [float(args.initial_lambda)]
                ),
            },
            "MEHA_AGILS_IGJO": "coefficient=ones, lambda=ones",
            "VF-iDCA": "released coefficient=zero, radius=ones",
            "LDMMA": "released coefficient=zero, radius=0.1, lambda=5",
        },
        "ldpm_pg": {
            "step_size": args.step_size,
            "beta0": args.beta0,
            "beta_power": args.beta_power,
        },
        "ldpm_pg_capped": {
            "step_size": args.capped_step_size,
            "beta0": args.capped_beta0,
            "beta_power": args.capped_beta_power,
            "beta_max": args.beta_max_capped,
        },
        "ldpm_cs": {
            "step_size": args.cs_step_size,
            "gamma": args.cs_gamma,
            "beta0": args.cs_beta0,
            "beta_power": args.cs_beta_power,
        },
        "ldpm_cs_capped": {
            "step_size": args.cs_capped_step_size,
            "gamma": args.cs_gamma,
            "beta0": args.cs_beta0,
            "beta_power": args.cs_beta_power,
            "beta_max": args.beta_max_capped,
        },
        "fixed_lambda_refit": {
            "solver_requested": args.solver,
            "solver_tol": args.solver_tol,
            "solver_max_iter": args.solver_max_iter,
        },
    }
    with (run_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    rows = []
    for summary_path in sorted(run_dir.glob("*_summary.json")):
        with summary_path.open() as handle:
            rows.append(json.load(handle))
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True, allow_nan=False)
    print("Saved results under %s" % run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
