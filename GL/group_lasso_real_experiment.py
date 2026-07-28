#!/usr/bin/env python3
"""Run Group Lasso bilevel methods on a9a or covtype.binary.

The driver keeps preprocessing outside the reported method time, freezes a
stratified split from the requested seed, and records enough metadata to
reproduce every table entry in the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

# Data_Generator imports libsvmdata at module load, so point that import at the
# repository-local immutable dataset cache before importing Data_Generator.
_LOCAL_DATA_HOME = Path(__file__).resolve().parent / "data" / "libsvmdata"
if _LOCAL_DATA_HOME.exists():
    os.environ.setdefault("LIBSVMDATA_HOME", str(_LOCAL_DATA_HOME))

from Data_Generator import Data, Data_with_Info
from group_lasso_baselines import METHOD_RUNNERS, ReducedGroupLassoProblem
from ldpm_core import LeastSquaresLDPM, group_regularizers


DATASET_KEYS = {
    "a9a": ("a9a", "a9a_test"),
    "covtype": ("covtype.binary", None),
    "covtype.binary": ("covtype.binary", None),
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
        "ldpm": "ldpm",
        "ldpm-pg": "ldpm",
        "uncapped": "ldpm",
        "ldpm-capped": "ldpm-capped",
        "ldpm-pg-c": "ldpm-capped",
        "capped": "ldpm-capped",
        "vf-idca": "vf-idca",
        "vfidca": "vf-idca",
        "vf": "vf-idca",
        "ldmma": "ldmma",
        "meha": "meha",
        "agils": "agils",
        "grid": "grid",
        "grid-search": "grid",
        "random": "random",
        "random-search": "random",
        "tpe": "tpe",
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


def fetch_dataset(name: str):
    from libsvmdata import fetch_dataset as fetch

    return fetch(name)


def binary_pm_one(y: np.ndarray, classes: Optional[np.ndarray] = None):
    y = np.asarray(y, dtype=float).reshape(-1)
    if classes is None:
        classes = np.unique(y)
    classes = np.asarray(classes, dtype=float)
    if classes.size != 2 or not np.all(np.isin(np.unique(y), classes)):
        raise ValueError("expected the same two binary classes, got %r" % np.unique(y).tolist())
    return np.where(y == classes[0], -1.0, 1.0), classes


def split_indices(
    labels: np.ndarray, seed: int, test_fraction: float, validation_fraction: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_indices = np.arange(len(labels), dtype=int)
    train_val, test = train_test_split(
        all_indices,
        test_size=test_fraction,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    train, validation = train_test_split(
        train_val,
        test_size=validation_fraction,
        random_state=seed + 1,
        shuffle=True,
        stratify=labels[train_val],
    )
    return np.sort(train), np.sort(validation), np.sort(test)


def train_rms_scale(x_train) -> np.ndarray:
    if sparse.issparse(x_train):
        squared_mean = np.asarray(x_train.power(2).mean(axis=0)).reshape(-1)
    else:
        squared_mean = np.mean(np.square(np.asarray(x_train, dtype=float)), axis=0)
    scale = np.sqrt(squared_mean)
    return np.where(scale > 1e-12, scale, 1.0)


def scaled_dense(x, scale: np.ndarray) -> np.ndarray:
    if sparse.issparse(x):
        return x.multiply(1.0 / scale).toarray().astype(float, copy=False)
    return np.asarray(x, dtype=float) / scale


def index_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.int64)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def prepare_data(args: argparse.Namespace):
    canonical = "covtype" if args.dataset == "covtype.binary" else args.dataset
    source_key, official_test_key = DATASET_KEYS[args.dataset]
    x_pool, y_pool_raw = fetch_dataset(source_key)
    y_pool, classes = binary_pm_one(y_pool_raw)

    if official_test_key is not None:
        x_test_source, y_test_raw = fetch_dataset(official_test_key)
        y_test, _ = binary_pm_one(y_test_raw, classes)
        pool_indices = np.arange(len(y_pool), dtype=int)
        train_indices, validation_indices = train_test_split(
            pool_indices,
            test_size=args.validation_fraction,
            random_state=args.seed,
            shuffle=True,
            stratify=y_pool,
        )
        train_indices = np.sort(train_indices)
        validation_indices = np.sort(validation_indices)
        test_indices = np.arange(len(y_test), dtype=int)
        x_train = x_pool[train_indices]
        x_validation = x_pool[validation_indices]
        x_test = x_test_source
        y_train = y_pool[train_indices]
        y_validation = y_pool[validation_indices]
        split_rule = "stratified 80/20 split of a9a; official a9a_test held out"
    else:
        train_indices, validation_indices, test_indices = split_indices(
            y_pool,
            args.seed,
            args.test_fraction,
            args.validation_fraction,
        )
        x_train = x_pool[train_indices]
        x_validation = x_pool[validation_indices]
        x_test = x_pool[test_indices]
        y_train = y_pool[train_indices]
        y_validation = y_pool[validation_indices]
        y_test = y_pool[test_indices]
        split_rule = "stratified 64/16/20 train/validation/test split"

    scale = train_rms_scale(x_train)
    data = Data()
    data.X_train = scaled_dense(x_train, scale)
    data.X_validate = scaled_dense(x_validation, scale)
    data.X_test = scaled_dense(x_test, scale)
    data.y_train = np.asarray(y_train, dtype=float)
    data.y_validate = np.asarray(y_validation, dtype=float)
    data.y_test = np.asarray(y_test, dtype=float)

    p = int(data.X_train.shape[1])
    num_groups = int(math.ceil(p / float(args.target_group_size)))
    settings = SimpleNamespace(
        num_train=len(data.y_train),
        num_validate=len(data.y_validate),
        num_test=len(data.y_test),
        num_features=p,
        num_experiment_groups=num_groups,
        dataset=canonical,
    )
    info = Data_with_Info(data, settings, args.seed)
    groups = group_regularizers(p, num_groups)
    group_ranges = [
        [int(reg["slice"].start), int(reg["slice"].stop)] for reg in groups
    ]
    metadata = {
        "dataset": canonical,
        "source_dataset": source_key,
        "official_test_dataset": official_test_key,
        "seed": args.seed,
        "split_rule": split_rule,
        "split_hash": index_hash(train_indices, validation_indices, test_indices),
        "data_shapes": {
            "train": list(data.X_train.shape),
            "validation": list(data.X_validate.shape),
            "test": list(data.X_test.shape),
        },
        "label_encoding": {str(classes[0]): -1, str(classes[1]): 1},
        "preprocessing": "train-only column RMS scaling without centering",
        "feature_scale_min": float(np.min(scale)),
        "feature_scale_max": float(np.max(scale)),
        "target_group_size": args.target_group_size,
        "num_groups": num_groups,
        "group_ranges_half_open": group_ranges,
        "group_sizes": [stop - start for start, stop in group_ranges],
        "intercept": False,
        "loss_normalization": "one half of per-example mean squared residual",
    }
    return info, metadata


def misclassification(x: np.ndarray, y: np.ndarray, coef: np.ndarray) -> float:
    prediction = np.where(x @ coef >= 0.0, 1.0, -1.0)
    return float(np.mean(prediction != y))


def _parse_lambda_values(value: object) -> np.ndarray:
    return np.asarray([float(item) for item in str(value).split(";") if item], dtype=float)


def feasible_history(
    data_info: Data_with_Info,
    groups,
    history,
    args: argparse.Namespace,
    method: str,
):
    """Re-solve the lower problem at selected recorded hyperparameters.

    This is the paper's feasible-error post-processing and is deliberately
    excluded from each method's reported running time.
    """

    if "lambda_values" not in history.columns:
        raise RuntimeError("history does not contain lambda snapshots")
    count = len(history)
    max_points = max(2, int(args.feasible_curve_points))
    if count <= max_points:
        selected = np.arange(count, dtype=int)
    else:
        selected = np.unique(
            np.rint(np.geomspace(1.0, float(count), max_points)).astype(int) - 1
        )
        selected = np.unique(np.concatenate(([0], selected, [count - 1]))).astype(int)
    sampled = history.iloc[selected].copy().reset_index(drop=True)
    problem = ReducedGroupLassoProblem(
        data_info,
        groups,
        {
            "lower_max_iter": args.feasible_lower_max_iter,
            "lower_tol": args.feasible_lower_tol,
        },
    )
    warm = None
    feasible_coefs = []
    val_values = []
    test_values = []
    train_values = []
    lower_iterations = []
    for value in sampled["lambda_values"]:
        lam = _parse_lambda_values(value)
        warm, iterations = problem.lower_solve(
            lam, x0=warm, max_iter=args.feasible_lower_max_iter, tol=args.feasible_lower_tol
        )
        feasible_coefs.append(warm.copy())
        train_values.append(problem.train_loss(warm))
        val_values.append(problem.validation_loss(warm))
        test_values.append(problem.test_loss(warm))
        lower_iterations.append(iterations)
    sampled["feasible_train_error"] = train_values
    sampled["feasible_validation_error"] = val_values
    sampled["feasible_test_error"] = test_values
    sampled["feasible_lower_iterations"] = lower_iterations
    return sampled, feasible_coefs[-1]


def paper29_metrics(data_info, groups, args, method, lam, iterate_coef, feasible_coef):
    """Metrics defined in the text above Table 3 of arXiv:2412.18929v5."""

    problem = ReducedGroupLassoProblem(
        data_info,
        groups,
        {"lower_max_iter": args.feasible_lower_max_iter, "lower_tol": args.feasible_lower_tol},
    )
    validation_error = 2.0 * problem.validation_loss(feasible_coef)
    test_error = 2.0 * problem.test_loss(feasible_coef)
    if method in {"grid", "random", "tpe"}:
        return validation_error, test_error, None, None

    test_error_infeasibility = 2.0 * problem.test_loss(iterate_coef)
    phi_iterate = problem.lower_objective(lam, iterate_coef)
    if method in {"meha", "agils"}:
        gamma = 1.0 / data_info.settings.num_features
        theta, _ = problem.proximal_lower_solve(
            lam,
            iterate_coef,
            gamma,
            max_iter=args.feasible_lower_max_iter,
            tol=args.feasible_lower_tol,
        )
        envelope_value = problem.lower_objective(lam, theta) + (
            0.5 * np.linalg.norm(theta - iterate_coef) ** 2 / gamma
        )
        raw_violation = phi_iterate - envelope_value
    else:
        value_function = problem.lower_objective(lam, feasible_coef)
        raw_violation = phi_iterate - value_function
    feasibility = max(float(raw_violation), 0.0) / data_info.settings.num_validate
    return validation_error, test_error, test_error_infeasibility, feasibility


def run_method(
    data_info: Data_with_Info,
    args: argparse.Namespace,
    method: str,
    run_dir: Path,
) -> Dict[str, object]:
    labels = {
        "ldpm": "LDPM-PG",
        "ldpm-capped": "LDPM-PG-C",
        "vf-idca": "VF-iDCA",
        "ldmma": "LDMMA",
        "meha": "MEHA",
        "agils": "AGILS",
        "grid": "Grid",
        "random": "Random",
        "tpe": "TPE",
    }
    label = labels[method]
    beta_max = args.beta_max_capped if method == "ldpm-capped" else None
    ldpm_step_size = (
        args.capped_step_size
        if method == "ldpm-capped" and args.capped_step_size is not None
        else args.step_size
    )
    ldpm_beta0 = (
        args.capped_beta0
        if method == "ldpm-capped" and args.capped_beta0 is not None
        else args.beta0
    )
    ldpm_beta_power = (
        args.capped_beta_power
        if method == "ldpm-capped" and args.capped_beta_power is not None
        else args.beta_power
    )
    group_count = data_info.settings.num_experiment_groups
    # LDPM and the Moreau methods use the page-29 all-ones start.  VF-iDCA and
    # LDMMA retain the auxiliary-variable starts required by their released
    # reformulations; those are set explicitly below.
    method_initial_lambda = args.initial_lambda
    common_initial_coef = np.ones(data_info.settings.num_features, dtype=float)
    setting = {
        "step_size": ldpm_step_size,
        "MAX_ITERATION": args.max_iter,
        "TOL": args.tol,
        "beta0": ldpm_beta0,
        "beta_power": ldpm_beta_power,
        "beta_max": beta_max,
        "initial_lambda": np.full(group_count, method_initial_lambda),
        "initial_coef": common_initial_coef.copy(),
        "normalize_loss": True,
        "sqrt_loss_scaling": True,
        "reduced_dual": True,
        "init_max_iter": args.init_max_iter,
        "init_tol": args.init_tol,
        "init_dual": args.init_dual,
        "stop_metric": "x_lambda",
        "record_interval": args.record_interval,
        "stop_patience": args.stop_patience,
        "line_search": True,
        "line_search_max_step": ldpm_step_size,
        "line_search_min_step": 1e-12,
        "line_search_decay": 0.5,
        "line_search_growth": 1.25,
        "max_line_search_iter": 50,
    }
    started = time.perf_counter()
    try:
        with time_limit(args.time_limit, label):
            groups = group_regularizers(data_info.settings.num_features, group_count)
            if method in ("ldpm", "ldpm-capped"):
                solver = LeastSquaresLDPM(data_info, groups, setting)
                history = solver.run_pgm()
            else:
                baseline_initial_lambda = method_initial_lambda
                baseline_initial_coef = common_initial_coef.copy()
                baseline_initial_radius = None
                if method == "vf-idca":
                    # Page 29 sets the VF upper/DC variable to one; the released
                    # VF-iDCA code initializes its regression coefficient at zero.
                    baseline_initial_coef = np.zeros_like(common_initial_coef)
                    baseline_initial_radius = np.ones(group_count)
                elif method == "ldmma":
                    # LDMMA is not part of the page-29 comparison.  Preserve its
                    # released SGL defaults instead of tuning it for these data.
                    baseline_initial_coef = np.zeros_like(common_initial_coef)
                    baseline_initial_lambda = 5.0
                    baseline_initial_radius = np.full(group_count, 0.1)
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
                    "initial_lambda": baseline_initial_lambda,
                    "initial_radius": baseline_initial_radius,
                    "lower_max_iter": args.lower_max_iter,
                    "lower_tol": args.lower_tol,
                    "solver": args.solver,
                    "solver_tol": args.solver_tol,
                    "solver_max_iter": args.solver_max_iter,
                    "lambda_floor": args.lambda_floor,
                    "lambda_ceiling": args.lambda_ceiling,
                    "vfidca_rho": args.vfidca_rho,
                    "vfidca_beta0": args.vfidca_beta0,
                    "vfidca_beta_delta": args.vfidca_beta_delta,
                    "vfidca_c": args.vfidca_c,
                    "vfidca_epsilon": args.vfidca_epsilon,
                    "vfidca_violation_weight": args.vfidca_violation_weight,
                    "vfidca_solver_tol": args.vfidca_solver_tol,
                    "vfidca_solver_max_iter": args.vfidca_solver_max_iter,
                    "ldmma_epsilon": args.ldmma_epsilon,
                    "ldmma_eta": args.ldmma_eta,
                    "ldmma_solver_tol": args.ldmma_solver_tol,
                    "ldmma_solver_max_iter": args.ldmma_solver_max_iter,
                    "moreau_gamma": args.moreau_gamma,
                    "meha_c0": args.meha_c0,
                    "meha_c_power": args.meha_c_power,
                    "meha_alpha": args.meha_alpha,
                    "meha_beta": args.meha_beta,
                    "meha_eta": args.meha_eta,
                    "moreau_initial_coef": args.moreau_initial_coef,
                    "agils_epsilon": args.agils_epsilon,
                    "agils_penalty0": args.agils_penalty0,
                    "agils_penalty_increment": args.agils_penalty_increment,
                    "agils_cp": args.agils_cp,
                    "agils_feasibility_tol": args.agils_feasibility_tol,
                    "agils_inner_max": args.agils_inner_max,
                    "agils_s0": args.agils_s0,
                    "agils_s_power": args.agils_s_power,
                    "agils_tau0": args.agils_tau0,
                    "agils_tau_power": args.agils_tau_power,
                    "agils_gamma": 1.0 / data_info.settings.num_features,
                    "initial_coef": baseline_initial_coef,
                    "seed": args.seed,
                    "grid_points": args.grid_points,
                    "search_budget": args.search_budget,
                    "paper29_protocol": args.paper29_protocol,
                }
                problem = ReducedGroupLassoProblem(data_info, groups, baseline_setting)
                if args.paper29_protocol:
                    agils_beta = 1.0 / (
                        problem.val_lipschitz / args.agils_penalty0
                        + problem.train_lipschitz
                        + 0.1
                    )
                    baseline_setting.update(
                        moreau_gamma=1.0 / problem.p,
                        meha_c0=20.0,
                        meha_c_power=0.1,
                        meha_alpha=1.0 / 1.1,
                        meha_beta=agils_beta / 8.0,
                        meha_eta=(1.0 / (problem.train_lipschitz + problem.p)) / 4.0,
                        vfidca_beta0=5.0,
                        vfidca_rho=0.01,
                        vfidca_c=1.0,
                        vfidca_beta_delta=0.1,
                    )
                history = METHOD_RUNNERS[method](problem, baseline_setting)
        elapsed = time.perf_counter() - started
    except MethodTimeout as exc:
        return {
            "method": label,
            "status": "timeout",
            "time": args.time_limit,
            "iterations": None,
            "val_loss": None,
            "test_loss": None,
            "val_misclassification": None,
            "test_misclassification": None,
            "x_lambda_stop": None,
            "message": str(exc),
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        return {
            "method": label,
            "status": "failed",
            "time": float(elapsed),
            "iterations": None,
            "val_loss": None,
            "test_loss": None,
            "val_misclassification": None,
            "test_misclassification": None,
            "x_lambda_stop": None,
            "message": "%s: %s" % (type(exc).__name__, exc),
        }

    if len(history) == 0:
        raise RuntimeError("%s returned no iterations" % label)
    final_history_time = float(history.iloc[-1]["time"])
    # Preserve the physical common start at t=0 while accounting for any
    # method-level overhead not included by the inner iteration timer.
    if final_history_time > 0.0:
        history["time"] = history["time"] * (elapsed / final_history_time)
    history_path = run_dir / (method.replace("-", "_") + "_history.csv")
    history.to_csv(history_path, index=False)
    iterate_coef = np.asarray(history.attrs["coef"], dtype=float)
    lam = np.asarray(history.attrs["lambda"], dtype=float)
    post_started = time.perf_counter()
    feasible_trace, coef = feasible_history(data_info, groups, history, args, method)
    postprocess_time = time.perf_counter() - post_started
    feasible_trace.to_csv(
        run_dir / (method.replace("-", "_") + "_feasible_history.csv"), index=False
    )
    state = {"coef": coef, "iterate_coef": iterate_coef, "lambda_value": lam}
    for attr in ("radius", "rho", "xi", "theta", "dual_anchor", "dual_multiplier"):
        if attr in history.attrs:
            state[attr] = np.asarray(history.attrs[attr], dtype=float)
    np.savez(run_dir / (method.replace("-", "_") + "_state.npz"), **state)
    last = history.iloc[-1]
    final_beta = float(last["beta"]) if "beta" in last else None
    cap_reached = (
        beta_max is not None
        and final_beta is not None
        and final_beta >= float(beta_max) - 1e-12
    )
    row = {
        "method": label,
        "status": str(history.attrs["termination_status"]),
        "time": float(elapsed),
        "iterations": int(last["iteration"]),
        "val_loss": float(last["validation_error"]),
        "test_loss": float(feasible_trace.iloc[-1]["feasible_test_error"]),
        "iterate_val_loss": float(last["validation_error"]),
        "iterate_test_loss": float(last["test_error"]),
        "postprocess_time": float(postprocess_time),
        "val_misclassification": misclassification(
            data_info.data.X_validate, data_info.data.y_validate, coef
        ),
        "test_misclassification": misclassification(
            data_info.data.X_test, data_info.data.y_test, coef
        ),
        "x_lambda_stop": float(last["x_lambda_stop"])
        if np.isfinite(float(last["x_lambda_stop"]))
        else None,
        "packed_relative_step": float(last["packed_relative_step"])
        if "packed_relative_step" in last
        else None,
        "ll_duality_gap": float(last["ll_duality_gap"])
        if "ll_duality_gap" in last
        else None,
        "ll_feasibility": float(last["ll_feasibility"])
        if "ll_feasibility" in last
        else None,
        "final_beta": final_beta,
        "cap_reached": bool(cap_reached) if beta_max is not None else None,
        "lambda_min": float(np.min(lam)),
        "lambda_max": float(np.max(lam)),
        "message": "",
    }
    row["val_loss"] = float(feasible_trace.iloc[-1]["feasible_validation_error"])
    (
        row["validation_error"],
        row["test_error"],
        row["test_error_infeasibility"],
        row["feasibility"],
    ) = paper29_metrics(data_info, groups, args, method, lam, iterate_coef, coef)
    for column in (
        "native_violation",
        "native_fenchel_gap",
        "native_constraint_violation",
        "moreau_residual",
        "theta_pg_residual",
        "agils_delta",
        "penalty_parameter",
    ):
        row[column] = float(last[column]) if column in last else None
    for key, value in list(row.items()):
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            row[key] = None
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASET_KEYS), required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tol", type=float, choices=[1e-4, 1e-5], required=True)
    parser.add_argument(
        "--methods",
        default="grid,random,tpe,vf-idca,ldmma,meha,agils,ldpm,ldpm-capped",
    )
    parser.add_argument("--paper29-protocol", action="store_true")
    parser.add_argument("--grid-points", type=int, default=20)
    parser.add_argument("--search-budget", type=int, default=400)
    parser.add_argument("--results-dir", default="results/group_lasso_real")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--target-group-size", type=int, default=8)
    parser.add_argument("--step-size", type=float, default=0.01)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta-power", type=float, default=0.3)
    parser.add_argument("--capped-step-size", type=float)
    parser.add_argument("--capped-beta0", type=float)
    parser.add_argument("--capped-beta-power", type=float)
    parser.add_argument("--beta-max-capped", type=float, default=10.0)
    parser.add_argument("--initial-lambda", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=100000)
    parser.add_argument("--record-interval", type=int, default=500)
    parser.add_argument("--baseline-record-interval", type=int, default=25)
    parser.add_argument("--stop-patience", type=int, default=1)
    parser.add_argument("--init-max-iter", type=int, default=300)
    parser.add_argument("--init-tol", type=float, default=1e-7)
    parser.add_argument("--init-dual", choices=["zero", "fenchel"], default="fenchel")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--lower-max-iter", type=int, default=5000)
    parser.add_argument("--lower-tol", type=float, default=1e-10)
    parser.add_argument("--lambda-floor", type=float, default=1e-8)
    parser.add_argument("--lambda-ceiling", type=float, default=10.0)
    parser.add_argument("--solver", choices=["CLARABEL", "SCS"], default="CLARABEL")
    parser.add_argument("--solver-tol", type=float, default=1e-7)
    parser.add_argument("--solver-max-iter", type=int, default=10000)
    parser.add_argument("--vfidca-rho", type=float, default=0.1)
    parser.add_argument("--vfidca-max-iter", type=int, default=50)
    parser.add_argument("--vfidca-beta0", type=float, default=1.0)
    parser.add_argument("--vfidca-beta-delta", type=float, default=5.0)
    parser.add_argument("--vfidca-c", type=float, default=0.01)
    parser.add_argument("--vfidca-epsilon", type=float, default=0.0)
    parser.add_argument("--vfidca-violation-weight", type=float, default=100.0)
    parser.add_argument("--vfidca-solver-tol", type=float, default=1e-4)
    parser.add_argument("--vfidca-solver-max-iter", type=int, default=100)
    parser.add_argument("--ldmma-epsilon", type=float, default=1e-3)
    parser.add_argument("--ldmma-max-iter", type=int, default=100)
    parser.add_argument("--ldmma-eta", type=float, default=0.0)
    parser.add_argument("--ldmma-solver-tol", type=float, default=1e-2)
    parser.add_argument("--ldmma-solver-max-iter", type=int, default=50)
    parser.add_argument("--moreau-gamma", type=float, default=1.0)
    parser.add_argument("--moreau-initial-coef", type=float, default=1.0)
    parser.add_argument("--meha-c0", type=float, default=1.0)
    parser.add_argument("--meha-max-iter", type=int, default=500)
    parser.add_argument("--meha-c-power", type=float, default=0.49)
    parser.add_argument("--meha-alpha", type=float, default=1e-4)
    parser.add_argument("--meha-beta", type=float, default=1e-3)
    parser.add_argument("--meha-eta", type=float, default=1e-3)
    parser.add_argument("--agils-epsilon", type=float, default=1e-6)
    parser.add_argument("--agils-max-iter", type=int, default=1000)
    parser.add_argument("--agils-penalty0", type=float, default=6.0)
    parser.add_argument("--agils-penalty-increment", type=float, default=0.01)
    parser.add_argument("--agils-cp", type=float, default=1.0)
    parser.add_argument("--agils-feasibility-tol", type=float, default=0.1)
    parser.add_argument("--agils-inner-max", type=int, default=10000)
    parser.add_argument("--agils-s0", type=float, default=5.0)
    parser.add_argument("--agils-s-power", type=float, default=1.05)
    parser.add_argument("--agils-tau0", type=float, default=10.0)
    parser.add_argument("--agils-tau-power", type=float, default=0.2)
    parser.add_argument("--feasible-curve-points", type=int, default=60)
    parser.add_argument("--feasible-lower-max-iter", type=int, default=50000)
    parser.add_argument("--feasible-lower-tol", type=float, default=1e-10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    data_info, metadata = prepare_data(args)
    tol_token = ("%.0e" % args.tol).replace("-", "m").replace("+", "")
    dataset_token = "covtype" if args.dataset == "covtype.binary" else args.dataset
    run_dir = Path(args.results_dir) / dataset_token / (
        "seed%d_tol%s" % (args.seed, tol_token)
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Data: %s train=%d val=%d test=%d features=%d groups=%d"
        % (
            dataset_token,
            data_info.settings.num_train,
            data_info.settings.num_validate,
            data_info.settings.num_test,
            data_info.settings.num_features,
            data_info.settings.num_experiment_groups,
        ),
        flush=True,
    )
    print(
        "Stopping: relative x/lambda change <= %.1e; step=%.3g; time limit=%.1fs"
        % (args.tol, args.step_size, args.time_limit),
        flush=True,
    )

    summaries: List[Dict[str, object]] = []
    for method in methods:
        print("Running %s" % method, flush=True)
        row = run_method(data_info, args, method, run_dir)
        summaries.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    metadata["tol"] = args.tol
    metadata["methods"] = methods
    metadata["solver_config"] = {
        "step_size": args.step_size,
        "beta0": args.beta0,
        "beta_power": args.beta_power,
        "capped_step_size": args.capped_step_size,
        "capped_beta0": args.capped_beta0,
        "capped_beta_power": args.capped_beta_power,
        "beta_max_capped": args.beta_max_capped,
        "initial_lambda": args.initial_lambda,
        "initialization": {
            "ldpm": "lambda=1 and coefficient=all ones",
            "meha_agils": "hyperparameter=1 and coefficient=theta=all ones (AGILS page 29)",
            "vf_idca": "DC/radius variable=1 and regression coefficient=zero (released VF-iDCA form)",
            "ldmma": "lambda=5, radius=0.1, regression coefficient=zero (released LDMMA defaults)",
            "search": "not iterative; every candidate solves the lower problem",
        },
        "max_iter": args.max_iter,
        "record_interval": args.record_interval,
        "baseline_record_interval": args.baseline_record_interval,
        "stop_patience": args.stop_patience,
        "init_max_iter": args.init_max_iter,
        "init_tol": args.init_tol,
        "init_dual": args.init_dual,
        "time_limit_seconds": args.time_limit,
        "paper29_protocol": args.paper29_protocol,
        "stop_metric": (
            "AGILS page-29 method-specific rules; LDPM uses exact relative "
            "x/lambda change sum <= tolerance"
            if args.paper29_protocol
            else "relative x/lambda change sum <= tolerance (first hit)"
        ),
        "loss_representation": "equivalent square-root sample scaling",
        "reduced_dual": "exact Gram-matrix sufficient-statistic representation",
        "error_evaluation": "lower problem re-solved at sampled lambda values; post-processing excluded from method runtime",
        "line_search": {
            "initial_and_max_step": args.step_size,
            "minimum_step": 1e-12,
            "decay": 0.5,
            "growth": 1.25,
            "maximum_trials": 50,
        },
        "baselines": {
            "grid_points": args.grid_points,
            "search_budget": args.search_budget,
            "search_log10_bounds": [-9.0, 2.0],
            "lambda_bounds": [args.lambda_floor, args.lambda_ceiling],
            "lower_max_iter": args.lower_max_iter,
            "lower_tol": args.lower_tol,
            "solver": args.solver,
            "solver_tol": args.solver_tol,
            "vfidca_solver_tol": args.vfidca_solver_tol,
            "vfidca_solver_max_iter": args.vfidca_solver_max_iter,
            "vfidca_max_iter": args.vfidca_max_iter,
            "vfidca_page29": {
                "beta0": 5.0,
                "rho": 0.01,
                "c": 1.0,
                "delta": 0.1,
                "stop": "max(relative (radius, coefficient) step, t_tilde/m)<0.1",
            }
            if args.paper29_protocol
            else None,
            "ldmma_epsilon": args.ldmma_epsilon,
            "ldmma_eta": args.ldmma_eta,
            "ldmma_solver_tol": args.ldmma_solver_tol,
            "ldmma_solver_max_iter": args.ldmma_solver_max_iter,
            "ldmma_max_iter": args.ldmma_max_iter,
            "ldmma_stop": "released relative (coefficient,radius,lambda) step<0.05",
            "moreau_gamma": (
                "1/m" if args.paper29_protocol else args.moreau_gamma
            ),
            "moreau_initial_coef": args.moreau_initial_coef,
            "meha": {
                "c0": 20.0 if args.paper29_protocol else args.meha_c0,
                "max_iter": args.meha_max_iter,
                "c_power": 0.1 if args.paper29_protocol else args.meha_c_power,
                "alpha": "AGILS alpha=1/1.1"
                if args.paper29_protocol
                else args.meha_alpha,
                "beta": "AGILS beta at p0=6 divided by 8"
                if args.paper29_protocol
                else args.meha_beta,
                "eta": "1/(L_train+m)/4"
                if args.paper29_protocol
                else args.meha_eta,
                "gamma": "1/m" if args.paper29_protocol else args.moreau_gamma,
                "stop": "relative (hyperparameter, coefficient) step < 0.005/m"
                if args.paper29_protocol
                else "requested x/lambda stop",
                "source": (
                    "released MEHA SGL update with parameters selected on AGILS page 29"
                    if args.paper29_protocol
                    else "SUSTech-Optimization/MEHAHO R/MEHA_SGL.R defaults"
                ),
            },
            "agils": {
                "epsilon": args.agils_epsilon,
                "max_iter": args.agils_max_iter,
                "penalty0": args.agils_penalty0,
                "penalty_increment": args.agils_penalty_increment,
                "cp": args.agils_cp,
                "feasibility_tol": args.agils_feasibility_tol,
                "inner_max": args.agils_inner_max,
                "s0": args.agils_s0,
                "s_power": args.agils_s_power,
                "tau0": args.agils_tau0,
                "tau_power": args.agils_tau_power,
                "gamma": 1.0 / data_info.settings.num_features,
                "alpha": "1/(Lg1+rho_g1+0.1)=1/1.1",
                "beta": "1/(L_validation/p_k+L_training+0.1)",
                "eta": "1/(L_training+m)",
                "stop": "relative (hyperparameter, coefficient) step < 0.005/m and t<0.1",
                "source": "arXiv:2412.18929v5 Algorithms 1-2 and Section 6.2; no public author repository found",
            },
            "feasible_postprocessing": {
                "curve_points": args.feasible_curve_points,
                "lower_max_iter": args.feasible_lower_max_iter,
                "lower_tol": args.feasible_lower_tol,
            },
        },
    }
    with (run_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(summaries, handle, indent=2, sort_keys=True, allow_nan=False)
    print("Saved results under %s" % run_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
