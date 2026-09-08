#!/usr/bin/env python3
"""Run and aggregate the ten-repetition splice direct-OGL campaign.

The single-run algorithms live in :mod:`experiment`.  This
wrapper freezes the paper protocol, computes the common feasible/infeasible
metrics used by the Group-Lasso and sparse-Group-Lasso experiments, and builds
validation-selected incumbent error-versus-time curves.

Algorithm time never includes the common lower-level re-solves performed here.
The test set is evaluated for reporting only; trajectory incumbents are selected
exclusively by validation error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# data imports libsvmdata eagerly.  Point it at the repository-local
# immutable cache before importing the single-run splice module.
SCRIPT_DIR = Path(__file__).resolve().parent
OGL_DIR = SCRIPT_DIR.parent
LOCAL_LIBSVMDATA_HOME = SCRIPT_DIR / "data" / "libsvmdata"
if LOCAL_LIBSVMDATA_HOME.exists():
    os.environ.setdefault("LIBSVMDATA_HOME", str(LOCAL_LIBSVMDATA_HOME))

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - checked before post-processing
    cp = None

for module_dir in (OGL_DIR, SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from experiment import (
    build_parser as build_experiment_parser,
    prepare_problem,
)
from methods import (
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    UPSTREAM_SGL_FIXED_CONFIG,
)


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    slug: str


METHODS: Tuple[MethodSpec, ...] = (
    MethodSpec("vf-idca", "VF-iDCA-upstream", "vf_idca_upstream"),
    MethodSpec("ldmma", "LDMMA", "ldmma"),
    MethodSpec("ldpm", "LDPM-CS", "ldpm"),
    MethodSpec("ldpm-capped", "LDPM-CS-C", "ldpm_capped"),
)
FAST_METHODS: Tuple[MethodSpec, ...] = tuple(
    method for method in METHODS if method.key != "ldmma"
)
LDMMA_METHOD = next(method for method in METHODS if method.key == "ldmma")
METHOD_BY_LABEL = {method.label: method for method in METHODS}
METHOD_ORDER = {method.label: index for index, method in enumerate(METHODS)}

TOLERANCES = (1e-5,)
SEEDS = tuple(range(2026, 2036))
NUM_GROUPS = 11
GROUP_SIZE = 10
GROUP_STRIDE = 5
LDPM_BETA0 = 1.04
LDPM_BETA_POWER = 0.3
LDPM_GAMMA = 5.0
LDPM_CAP = 7.5
LDMMA_TIMEOUT = 180.0
CURVE_POINTS_PER_RUN = 40
CURVE_GRID_POINTS = 200
INVALID_STATUSES = {
    "failed",
    "timeout",
    "skipped_after_timeout",
    "postprocess_failed",
}

METRICS = (
    "time",
    "validation_error",
    "test_error",
    "test_error_infeasibility",
    "feasibility",
    "validation_misclassification",
    "test_misclassification",
    "feasible_validation_misclassification",
    "feasible_test_misclassification",
)

COLORS = {
    "VF-iDCA-upstream": "#D55E00",
    "LDMMA": "#E69F00",
    "LDPM-CS": "#0072B2",
    "LDPM-CS-C": "#009E73",
}
LINE_STYLES = {
    "VF-iDCA-upstream": "-.",
    "LDMMA": ":",
    "LDPM-CS": "-",
    "LDPM-CS-C": "--",
}
DISPLAY_LABELS = {"VF-iDCA-upstream": "VF-iDCA"}


def _tol_token(tol: float) -> str:
    return ("%.0e" % float(tol)).replace("-", "m").replace("+", "")


def _run_directory(root: Path, seed: int, tol: float) -> Path:
    # Deliberately no extra ``splice`` directory: this is the frozen layout.
    return root / ("seed%d_tol%s" % (int(seed), _tol_token(tol)))


def _parse_ints(raw: str) -> List[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _parse_floats(raw: str) -> List[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results" / "splice"),
        help="Campaign root; run directories are direct children of this path.",
    )
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--tols", default="1e-5")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--curve-points-per-run", type=int, default=CURVE_POINTS_PER_RUN
    )
    parser.add_argument("--curve-grid-points", type=int, default=CURVE_GRID_POINTS)
    parser.add_argument(
        "--vfidca-deps",
        default="/private/tmp/vfidca-deps",
        help="Optional isolated path containing ECOS for the pinned VF-iDCA code.",
    )
    parser.add_argument(
        "--vfidca-initial-r",
        type=float,
        default=None,
        help=(
            "Optional VF-iDCA initial group radius passed to every formal run. "
            "Use 0.1 for the released iP_DCA algorithm fallback."
        ),
    )
    return parser


def _validate_protocol(seeds: Sequence[int], tolerances: Sequence[float], workers: int) -> None:
    if len(seeds) != len(SEEDS) or len(set(seeds)) != len(SEEDS):
        raise ValueError("the formal splice reporting protocol requires ten distinct seeds")
    rounded = {float("%.12g" % value) for value in tolerances}
    if rounded != set(TOLERANCES) or len(tolerances) != 1:
        raise ValueError("the formal splice protocol requires tol=1e-5")
    if workers <= 0:
        raise ValueError("--workers must be positive")


def _summary_is_complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return False
    return {row.get("method") for row in rows} == set(METHOD_BY_LABEL)


def _metadata_matches(path: Path, seed: int, tol: float) -> bool:
    if not path.exists():
        return False
    try:
        metadata = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        int(metadata.get("seed", -1)) == int(seed)
        and np.isclose(float(metadata.get("tol", np.nan)), float(tol))
        and len(metadata.get("groups", [])) == NUM_GROUPS
        and bool(metadata.get("record_snapshots", False))
        and int(metadata.get("ldpm_stop_patience", -1)) == 1
        and np.isfinite(float(metadata.get("beta_max_capped", np.nan)))
        and float(metadata.get("beta_max_capped", 0.0)) > 0.0
        and np.isclose(
            float(metadata.get("ldmma_timeout_seconds", np.nan)), LDMMA_TIMEOUT
        )
    )


def _complete_run(root: Path, seed: int, tol: float) -> bool:
    directory = _run_directory(root, seed, tol)
    return _summary_is_complete(directory / "summary.json") and _metadata_matches(
        directory / "metadata.json", seed, tol
    )


def _subprocess_environment(vfidca_deps: Path) -> Dict[str, str]:
    environment = os.environ.copy()
    environment["LIBSVMDATA_HOME"] = str(LOCAL_LIBSVMDATA_HOME)
    environment["PYTHONPYCACHEPREFIX"] = "/private/tmp/ldpm_splice_campaign_pycache"
    environment["MPLCONFIGDIR"] = "/private/tmp/ldpm_splice_campaign_mpl"
    if vfidca_deps.exists():
        old_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = str(vfidca_deps) + (
            os.pathsep + old_path if old_path else ""
        )
    return environment


def _run_subset(
    root: Path,
    seed: int,
    tol: float,
    methods: Sequence[MethodSpec],
    vfidca_deps: Path,
    log_name: str,
    vfidca_initial_r: Optional[float] = None,
) -> List[Dict[str, object]]:
    directory = _run_directory(root, seed, tol)
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(SCRIPT_DIR / "experiment.py"),
        "--methods",
        ",".join(method.key for method in methods),
        "--seed",
        str(seed),
        "--tol",
        "%.17g" % tol,
        "--results-dir",
        str(root),
        "--num-groups",
        str(NUM_GROUPS),
        "--group-size",
        str(GROUP_SIZE),
        "--stride",
        str(GROUP_STRIDE),
        "--no-standardize",
        "--ldpm-stop-patience",
        "1",
        "--ldpm-record-interval",
        "1",
        "--record-snapshots",
        "--beta0",
        "%.17g" % LDPM_BETA0,
        "--beta-power",
        "%.17g" % LDPM_BETA_POWER,
        "--gamma",
        "%.17g" % LDPM_GAMMA,
        "--beta-max-capped",
        "%.17g" % LDPM_CAP,
        "--baseline-timeout",
        "%.17g" % LDMMA_TIMEOUT,
    ]
    if any(method.key == "vf-idca" for method in methods) and vfidca_initial_r is not None:
        command.extend(["--vfidca-initial-r", "%.17g" % vfidca_initial_r])
    log_path = directory / log_name
    with log_path.open("w") as log:
        log.write("command: %s\n" % " ".join(command))
        log.flush()
        completed = subprocess.run(
            command,
            cwd=str(SCRIPT_DIR),
            env=_subprocess_environment(vfidca_deps),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            "splice seed=%d tol=%.1e failed with exit code %d; see %s"
            % (seed, tol, completed.returncode, log_path)
        )
    summary_path = directory / "summary.json"
    metadata_path = directory / "metadata.json"
    if not summary_path.exists() or not _metadata_matches(metadata_path, seed, tol):
        raise RuntimeError("single-run output failed the frozen-protocol audit: %s" % directory)
    rows = json.loads(summary_path.read_text())
    expected = {method.label for method in methods}
    if {row.get("method") for row in rows} != expected:
        raise RuntimeError(
            "single-run method set mismatch under %s: expected %s"
            % (directory, sorted(expected))
        )
    return rows


def _run_fast_one(
    root: Path,
    seed: int,
    tol: float,
    resume: bool,
    vfidca_deps: Path,
    vfidca_initial_r: Optional[float],
) -> Tuple[int, float, str]:
    if resume and _complete_run(root, seed, tol):
        return seed, tol, "skipped"
    _run_subset(
        root,
        seed,
        tol,
        FAST_METHODS,
        vfidca_deps,
        "campaign_fast.log",
        vfidca_initial_r,
    )
    return seed, tol, "completed-fast-methods"


def _skipped_ldmma_row(first_timeout_seed: int, tol: float) -> Dict[str, object]:
    return {
        "method": LDMMA_METHOD.label,
        "status": "skipped_after_timeout",
        "iterations": 0,
        "time": None,
        "val_loss": None,
        "test_loss": None,
        "validation_accuracy": None,
        "test_accuracy": None,
        "x_lambda_stop": None,
        "stop_value": None,
        "stop_metric": None,
        "converged": False,
        "message": (
            "Not run because LDMMA first timed out at seed=%d for tol=%.1e."
            % (first_timeout_seed, tol)
        ),
    }


def _append_ldmma_result(
    root: Path,
    seed: int,
    tol: float,
    vfidca_deps: Path,
    skipped_after_seed: Optional[int] = None,
) -> str:
    directory = _run_directory(root, seed, tol)
    summary_path = directory / "summary.json"
    fast_rows = json.loads(summary_path.read_text())
    if {row.get("method") for row in fast_rows} != {
        method.label for method in FAST_METHODS
    }:
        raise RuntimeError("fast-method summary is incomplete under %s" % directory)
    if skipped_after_seed is None:
        ldmma_rows = _run_subset(
            root,
            seed,
            tol,
            (LDMMA_METHOD,),
            vfidca_deps,
            "campaign_ldmma.log",
        )
        ldmma_row = ldmma_rows[0]
    else:
        ldmma_row = _skipped_ldmma_row(skipped_after_seed, tol)
    summary_path.write_text(
        json.dumps(fast_rows + [ldmma_row], indent=2, allow_nan=False) + "\n"
    )
    if not _complete_run(root, seed, tol):
        raise RuntimeError("merged output failed the frozen-protocol audit: %s" % directory)
    return str(ldmma_row.get("status", "failed"))


def run_campaign(
    root: Path,
    seeds: Sequence[int],
    tolerances: Sequence[float],
    workers: int,
    resume: bool,
    vfidca_deps: Path,
    vfidca_initial_r: Optional[float],
) -> None:
    # Run the three non-LDMMA methods independently so a slow LDMMA solve can
    # neither delay nor erase their artifacts.  This also lets us honor the
    # user's explicit rule to stop further LDMMA attempts after the first
    # timeout at a given tolerance.
    jobs = [(seed, tol) for tol in tolerances for seed in seeds]
    failures: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _run_fast_one,
                root,
                seed,
                tol,
                resume,
                vfidca_deps,
                vfidca_initial_r,
            ): (
                seed,
                tol,
            )
            for seed, tol in jobs
        }
        for future in as_completed(futures):
            seed, tol = futures[future]
            try:
                _, _, status = future.result()
                print(
                    "%s splice seed=%d tol=%.1e" % (status.capitalize(), seed, tol),
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - collect all worker failures
                failures.append("seed=%d tol=%.1e: %s" % (seed, tol, exc))
                print("Failed %s" % failures[-1], flush=True)
    if failures:
        raise RuntimeError("campaign failures:\n" + "\n".join(failures))

    for tol in tolerances:
        first_timeout_seed: Optional[int] = None
        for seed in seeds:
            if resume and _complete_run(root, seed, tol):
                rows = json.loads(
                    (_run_directory(root, seed, tol) / "summary.json").read_text()
                )
                ldmma = next(row for row in rows if row.get("method") == "LDMMA")
                if ldmma.get("status") == "timeout" and first_timeout_seed is None:
                    first_timeout_seed = int(seed)
                continue
            status = _append_ldmma_result(
                root,
                seed,
                tol,
                vfidca_deps,
                skipped_after_seed=first_timeout_seed,
            )
            print(
                "LDMMA splice seed=%d tol=%.1e: %s" % (seed, tol, status),
                flush=True,
            )
            if status == "timeout" and first_timeout_seed is None:
                first_timeout_seed = int(seed)


class OverlappingLowerSolver:
    """Common penalized lower-level solve for the direct overlapping groups."""

    def __init__(self, problem):
        if cp is None:
            raise ImportError("cvxpy is required for common OGL post-processing")
        self.problem_data = problem
        self.x = cp.Variable(problem.p)
        self.lam = cp.Parameter(problem.group_count, nonneg=True)
        loss = 0.5 / len(problem.b_tr) * cp.sum_squares(
            problem.a_tr @ self.x - problem.b_tr
        )
        penalty = sum(
            self.lam[index] * cp.norm(self.x[group], 2)
            for index, group in enumerate(problem.groups)
        )
        self.model = cp.Problem(cp.Minimize(loss + penalty))

    def solve(self, lam: np.ndarray) -> Tuple[np.ndarray, str]:
        lam = np.asarray(lam, dtype=float).reshape(self.problem_data.group_count)
        if not np.all(np.isfinite(lam)):
            raise ValueError("lower-level hyperparameters are nonfinite")
        used_lam, _ = _sanitize_lambda(lam)
        self.lam.value = used_lam
        self.x.value = np.zeros(self.problem_data.p, dtype=float)
        clarabel_acceptable = False
        try:
            self.model.solve(
                solver=cp.CLARABEL,
                warm_start=False,
                tol_gap_abs=1e-10,
                tol_gap_rel=1e-10,
                tol_feas=1e-10,
                max_iter=1000,
                verbose=False,
            )
            clarabel_acceptable = self.model.status == cp.OPTIMAL
        except Exception:
            clarabel_acceptable = False
        if not clarabel_acceptable:
            self.x.value = np.zeros(self.problem_data.p, dtype=float)
            self.model.solve(
                solver=cp.SCS,
                warm_start=False,
                eps=1e-8,
                max_iters=200000,
                verbose=False,
            )
        if self.x.value is None or self.model.status not in {
            cp.OPTIMAL,
            cp.OPTIMAL_INACCURATE,
        }:
            raise RuntimeError("common lower solve failed with status %s" % self.model.status)
        return np.asarray(self.x.value, dtype=float).reshape(-1), str(self.model.status)


def _sanitize_lambda(lam: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    """Clip only solver-roundoff-scale negative hyperparameters."""

    raw = np.asarray(lam, dtype=float).reshape(-1)
    if not np.all(np.isfinite(raw)):
        raise ValueError("lower-level hyperparameters are nonfinite")
    norm_inf = float(np.linalg.norm(raw, ord=np.inf)) if raw.size else 0.0
    tolerance = 1e-8 * max(1.0, norm_inf)
    negative_inf = max(0.0, -float(np.min(raw))) if raw.size else 0.0
    if negative_inf > tolerance:
        raise ValueError(
            "negative lambda %.3e exceeds clipping tolerance %.3e"
            % (negative_inf, tolerance)
        )
    return np.maximum(raw, 0.0), {
        "lambda_negative_inf": negative_inf,
        "lambda_clip_tolerance": tolerance,
        "lambda_clipped": bool(negative_inf > 0.0),
    }


def _half_mse(matrix: np.ndarray, response: np.ndarray, x: np.ndarray) -> float:
    residual = matrix @ x - response
    return 0.5 / max(1, response.size) * float(residual @ residual)


def _misclassification(matrix: np.ndarray, response: np.ndarray, x: np.ndarray) -> float:
    prediction = np.where(matrix @ x >= 0.0, 1.0, -1.0)
    return float(np.mean(prediction != response))


def _lower_objective(problem, lam: np.ndarray, x: np.ndarray) -> float:
    norms = np.asarray([np.linalg.norm(x[group]) for group in problem.groups])
    return float(_half_mse(problem.a_tr, problem.b_tr, x) + np.dot(lam, norms))


def _common_metrics(
    problem,
    solver: OverlappingLowerSolver,
    raw_x: np.ndarray,
    raw_lam: np.ndarray,
) -> Tuple[Dict[str, object], np.ndarray]:
    raw_x = np.asarray(raw_x, dtype=float).reshape(problem.p)
    raw_lam = np.asarray(raw_lam, dtype=float).reshape(problem.group_count)
    lam, lambda_audit = _sanitize_lambda(raw_lam)
    feasible_x, lower_status = solver.solve(raw_lam)
    raw_phi = _lower_objective(problem, lam, raw_x)
    feasible_phi = _lower_objective(problem, lam, feasible_x)
    objective_gap = raw_phi - feasible_phi
    gap_scale = max(1.0, abs(raw_phi), abs(feasible_phi))
    gap_tolerance = 1e-8 * gap_scale
    if objective_gap < -gap_tolerance:
        raise RuntimeError(
            "common lower solve failed objective-gap audit: %.3e < -%.3e"
            % (objective_gap, gap_tolerance)
        )
    feasibility = max(objective_gap, 0.0) / max(1, problem.b_val.size)
    identity_residual = abs(
        problem.b_val.size * feasibility - max(objective_gap, 0.0)
    ) / gap_scale
    if identity_residual > 1e-12:
        raise RuntimeError(
            "feasibility identity residual %.3e exceeds 1e-12"
            % identity_residual
        )
    metrics: Dict[str, object] = {
        # The page-29 errors are MSE, hence twice the internal half-MSE.
        "validation_error": 2.0
        * _half_mse(problem.a_val, problem.b_val, feasible_x),
        "test_error": 2.0
        * _half_mse(problem.a_test, problem.b_test, feasible_x),
        "test_error_infeasibility": 2.0
        * _half_mse(problem.a_test, problem.b_test, raw_x),
        "feasibility": feasibility,
        # Preserve the semantics of the original splice table: its
        # misclassification columns are evaluated at the method's raw iterate.
        "validation_misclassification": _misclassification(
            problem.a_val, problem.b_val, raw_x
        ),
        "test_misclassification": _misclassification(
            problem.a_test, problem.b_test, raw_x
        ),
        # Retain the feasible-solution rates as supplemental audit fields.
        "feasible_validation_misclassification": _misclassification(
            problem.a_val, problem.b_val, feasible_x
        ),
        "feasible_test_misclassification": _misclassification(
            problem.a_test, problem.b_test, feasible_x
        ),
        "raw_lower_objective": raw_phi,
        "feasible_lower_objective": feasible_phi,
        "lower_objective_gap": objective_gap,
        "lower_gap_tolerance": gap_tolerance,
        "feasibility_identity_residual": identity_residual,
        "lower_solve_status": lower_status,
        "lambda_min_raw": float(np.min(raw_lam)),
        "lambda_min_used": float(np.min(lam)),
        **lambda_audit,
    }
    return metrics, feasible_x


def _load_problem(seed: int):
    experiment_args = build_experiment_parser().parse_args([])
    experiment_args.seed = int(seed)
    experiment_args.num_groups = NUM_GROUPS
    experiment_args.group_size = GROUP_SIZE
    experiment_args.stride = GROUP_STRIDE
    experiment_args.standardize = False
    problem, settings = prepare_problem(experiment_args)
    if problem.group_count != NUM_GROUPS:
        raise RuntimeError(
            "expected %d direct overlapping groups, found %d"
            % (NUM_GROUPS, problem.group_count)
        )
    return problem, settings


def _source_rows(path: Path) -> Mapping[str, Dict[str, object]]:
    rows = json.loads(path.read_text())
    by_label = {str(row.get("method")): dict(row) for row in rows}
    if set(by_label) != set(METHOD_BY_LABEL):
        raise RuntimeError("incomplete method set in %s" % path)
    return by_label


def _failure_metric_row(
    source: Mapping[str, object],
    metadata: Mapping[str, object],
    method: MethodSpec,
    seed: int,
    tol: float,
) -> Dict[str, object]:
    status = str(source.get("status", "failed"))
    elapsed = source.get("time")
    if elapsed is None and status == "timeout":
        elapsed = LDMMA_TIMEOUT
    cap_value = float(metadata.get("beta_max_capped", LDPM_CAP))
    return {
        "dataset": "splice",
        "tol": float(tol),
        "seed": int(seed),
        "method": method.label,
        "method_key": method.key,
        "status": status,
        "valid_for_stats": False,
        "time": elapsed,
        "iterations": source.get("iterations"),
        "validation_error": None,
        "test_error": None,
        "test_error_infeasibility": None,
        "feasibility": None,
        "validation_misclassification": None,
        "test_misclassification": None,
        "feasible_validation_misclassification": None,
        "feasible_test_misclassification": None,
        "x_lambda_stop": source.get("x_lambda_stop"),
        "stop_value": source.get("stop_value"),
        "final_beta": source.get("beta"),
        "beta_max": cap_value if method.key == "ldpm-capped" else None,
        "cap_reached": None,
        "split_hash": metadata.get("split_hash"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "message": source.get("message", ""),
    }


def _final_run_row(
    directory: Path,
    metadata: Mapping[str, object],
    source: Mapping[str, object],
    method: MethodSpec,
    seed: int,
    tol: float,
    problem,
    solver: OverlappingLowerSolver,
) -> Dict[str, object]:
    status = str(source.get("status", "failed"))
    if status in INVALID_STATUSES:
        return _failure_metric_row(source, metadata, method, seed, tol)
    state_path = directory / (method.slug + "_state.npz")
    if not state_path.exists():
        failed = dict(source)
        failed["status"] = "postprocess_failed"
        failed["message"] = "missing final state %s" % state_path
        return _failure_metric_row(failed, metadata, method, seed, tol)
    try:
        with np.load(state_path) as state:
            raw_x = np.asarray(state["x"], dtype=float)
            raw_lam = np.asarray(state["lambda"], dtype=float)
        metrics, _ = _common_metrics(problem, solver, raw_x, raw_lam)
    except Exception as exc:  # noqa: BLE001 - retain failed post-processing row
        failed = dict(source)
        failed["status"] = "postprocess_failed"
        failed["message"] = "%s: %s" % (type(exc).__name__, exc)
        return _failure_metric_row(failed, metadata, method, seed, tol)

    final_beta = source.get("beta")
    cap_reached: Optional[bool]
    cap_value = float(metadata.get("beta_max_capped", LDPM_CAP))
    if method.key == "ldpm-capped" and final_beta is not None:
        cap_reached = bool(float(final_beta) >= cap_value - 1e-12)
    else:
        cap_reached = None
    row: Dict[str, object] = {
        "dataset": "splice",
        "tol": float(tol),
        "seed": int(seed),
        "method": method.label,
        "method_key": method.key,
        "status": status,
        "valid_for_stats": True,
        "time": source.get("time"),
        "iterations": source.get("iterations"),
        **metrics,
        "x_lambda_stop": source.get("x_lambda_stop"),
        "stop_value": source.get("stop_value"),
        "final_beta": final_beta,
        "beta_max": cap_value if method.key == "ldpm-capped" else None,
        "cap_reached": cap_reached,
        "split_hash": metadata.get("split_hash"),
        "dataset_fingerprint": metadata.get("dataset_fingerprint"),
        "message": source.get("message", ""),
    }
    return row


def _parse_vector(value: object, expected: int, column: str, path: Path) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=float).reshape(-1)
    else:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            raise ValueError("empty %s snapshot in %s" % (column, path))
        try:
            parsed = json.loads(text)
        except Exception:
            cleaned = text.strip("[]()")
            cleaned = cleaned.replace(";", " ").replace(",", " ")
            array = np.fromstring(cleaned, sep=" ", dtype=float)
        else:
            array = np.asarray(parsed, dtype=float).reshape(-1)
    if array.size != expected or not np.all(np.isfinite(array)):
        raise ValueError(
            "%s snapshot in %s has size %d, expected %d"
            % (column, path, array.size, expected)
        )
    return array


def _geometric_indices(count: int, maximum: int) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=int)
    maximum = max(2, int(maximum))
    if count <= maximum:
        return np.arange(count, dtype=int)
    selected = np.rint(np.geomspace(1.0, float(count), maximum)).astype(int) - 1
    return np.unique(np.concatenate(([0], selected, [count - 1]))).astype(int)


def _run_incumbent_curve(
    path: Path,
    method: MethodSpec,
    seed: int,
    tol: float,
    problem,
    solver: OverlappingLowerSolver,
    maximum_points: int,
) -> pd.DataFrame:
    history = pd.read_csv(path)
    required = {"time", "x_values", "lambda_values"}
    missing = required.difference(history.columns)
    if missing:
        raise RuntimeError("%s is missing trajectory columns %s" % (path, sorted(missing)))
    history = history.sort_values("time", kind="stable").reset_index(drop=True)
    selected = _geometric_indices(len(history), maximum_points)
    rows: List[Dict[str, object]] = []
    best_validation = np.inf
    best_test = np.nan
    best_iteration: Optional[int] = None
    for index in selected:
        record = history.iloc[int(index)]
        raw_x = _parse_vector(record["x_values"], problem.p, "x_values", path)
        raw_lam = _parse_vector(
            record["lambda_values"], problem.group_count, "lambda_values", path
        )
        # Use one compiled parameterized model, but cold-solve every checkpoint
        # so path order cannot affect the selected feasible solution.
        try:
            feasible_x, _ = solver.solve(raw_lam)
        except ValueError:
            # Do not silently clip materially negative hyperparameters into a
            # different path; simply omit this invalid checkpoint from the
            # validation-selected curve.
            continue
        validation = 2.0 * _half_mse(problem.a_val, problem.b_val, feasible_x)
        test = 2.0 * _half_mse(problem.a_test, problem.b_test, feasible_x)
        iteration = int(record["iteration"]) if "iteration" in record else int(index)
        if validation < best_validation - 1e-15:
            best_validation = validation
            best_test = test
            best_iteration = iteration
        rows.append(
            {
                "dataset": "splice",
                "tol": float(tol),
                "seed": int(seed),
                "method": method.label,
                "time": float(record["time"]),
                "checkpoint_iteration": iteration,
                "incumbent_iteration": best_iteration,
                "validation_error": float(best_validation),
                # Test uses the exact checkpoint selected by validation only.
                "test_error": float(best_test),
                "checkpoint_validation_error": float(validation),
                "checkpoint_test_error": float(test),
            }
        )
    return pd.DataFrame(rows)


def load_and_postprocess(
    root: Path,
    seeds: Sequence[int],
    tolerances: Sequence[float],
    curve_points_per_run: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, object]]:
    result_rows: List[Dict[str, object]] = []
    curve_frames: List[pd.DataFrame] = []
    settings_by_seed: Dict[int, object] = {}
    problem_cache: Dict[int, object] = {}
    solver_cache: Dict[int, OverlappingLowerSolver] = {}

    for seed in seeds:
        problem, settings = _load_problem(seed)
        problem_cache[seed] = problem
        settings_by_seed[seed] = settings
        solver_cache[seed] = OverlappingLowerSolver(problem)

    for tol in tolerances:
        for seed in seeds:
            directory = _run_directory(root, seed, tol)
            summary_path = directory / "summary.json"
            metadata_path = directory / "metadata.json"
            if not summary_path.exists() or not metadata_path.exists():
                raise FileNotFoundError("missing run artifacts under %s" % directory)
            if not _metadata_matches(metadata_path, seed, tol):
                raise RuntimeError("run metadata does not match frozen protocol: %s" % directory)
            metadata = json.loads(metadata_path.read_text())
            settings = settings_by_seed[seed]
            if metadata.get("split_hash") != settings.split_hash:
                raise RuntimeError("split hash mismatch in %s" % directory)
            if metadata.get("dataset_fingerprint") != settings.dataset_fingerprint:
                raise RuntimeError("dataset fingerprint mismatch in %s" % directory)
            source = _source_rows(summary_path)
            problem = problem_cache[seed]
            solver = solver_cache[seed]
            for method in METHODS:
                source_row = source[method.label]
                final_row = _final_run_row(
                    directory,
                    metadata,
                    source_row,
                    method,
                    seed,
                    tol,
                    problem,
                    solver,
                )
                result_rows.append(final_row)
                if not bool(final_row["valid_for_stats"]):
                    continue
                history_path = directory / (method.slug + "_history.csv")
                if not history_path.exists():
                    raise FileNotFoundError(history_path)
                curve_frames.append(
                    _run_incumbent_curve(
                        history_path,
                        method,
                        seed,
                        tol,
                        problem,
                        solver,
                        curve_points_per_run,
                    )
                )

    runs = pd.DataFrame(result_rows)
    expected = len(seeds) * len(tolerances) * len(METHODS)
    if len(runs) != expected:
        raise RuntimeError("expected %d final rows, found %d" % (expected, len(runs)))
    per_run_curves = (
        pd.concat(curve_frames, ignore_index=True)
        if curve_frames
        else pd.DataFrame(
            columns=[
                "dataset",
                "tol",
                "seed",
                "method",
                "time",
                "validation_error",
                "test_error",
            ]
        )
    )
    return runs, per_run_curves, settings_by_seed


def aggregate_runs(runs: pd.DataFrame) -> pd.DataFrame:
    output: List[Dict[str, object]] = []
    grouped = runs.groupby(["tol", "method"], sort=False)
    for (tol, method), block in grouped:
        valid = block[block["valid_for_stats"].astype(bool)]
        row: Dict[str, object] = {
            "dataset": "splice",
            "tol": float(tol),
            "method": method,
            "runs": int(len(block)),
            "valid_runs": int(len(valid)),
            "statuses": ";".join(sorted(block["status"].astype(str).unique())),
            "timeout_runs": int(np.sum(block["status"] == "timeout")),
            "skipped_after_timeout_runs": int(
                np.sum(block["status"] == "skipped_after_timeout")
            ),
            "failed_runs": int(
                np.sum(block["status"].isin(["failed", "postprocess_failed"]))
            ),
            "cap_reached_runs": int(
                np.sum(valid["cap_reached"].dropna().astype(bool))
            ),
        }
        for metric in METRICS:
            values = pd.to_numeric(valid[metric], errors="coerce")
            values = values[np.isfinite(values)]
            row[metric + "_valid_runs"] = int(values.size)
            row[metric + "_mean"] = float(values.mean()) if values.size else np.nan
            row[metric + "_std"] = (
                float(values.std(ddof=1)) if values.size >= 2 else np.nan
            )
        output.append(row)
    summary = pd.DataFrame(output)
    summary["tol_order"] = summary["tol"].map({1e-4: 0, 1e-5: 1})
    summary["method_order"] = summary["method"].map(METHOD_ORDER)
    return summary.sort_values(["tol_order", "method_order"]).drop(
        columns=["tol_order", "method_order"]
    )


def _interpolate_incumbent(
    times: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    order = np.argsort(times, kind="stable")
    times = np.asarray(times[order], dtype=float)
    values = np.asarray(values[order], dtype=float)
    finite = np.isfinite(times) & np.isfinite(values)
    times, values = times[finite], values[finite]
    if not times.size:
        return np.full(grid.shape, np.nan)
    # Collapse duplicate timestamps, retaining the last available incumbent.
    unique_times, reverse = np.unique(times[::-1], return_index=True)
    last_indices = times.size - 1 - reverse
    reorder = np.argsort(unique_times)
    unique_times = unique_times[reorder]
    unique_values = values[last_indices[reorder]]
    return np.interp(
        grid,
        unique_times,
        unique_values,
        left=np.nan,
        right=float(unique_values[-1]),
    )


def aggregate_curves(
    per_run: pd.DataFrame,
    tolerances: Sequence[float],
    grid_points: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    grid_points = max(2, int(grid_points))
    for tol in tolerances:
        for method in METHODS:
            block = per_run[
                np.isclose(per_run["tol"].astype(float), float(tol))
                & (per_run["method"] == method.label)
            ]
            histories = [frame for _, frame in block.groupby("seed", sort=True)]
            if not histories:
                continue
            # Start only once every valid run has at least one checkpoint.  This
            # prevents a changing run count from creating artificial jumps in
            # an otherwise nonincreasing mean validation curve.
            start = max(float(frame["time"].min()) for frame in histories)
            stop = max(float(frame["time"].max()) for frame in histories)
            if stop <= start:
                grid = np.asarray([start], dtype=float)
            elif start > 0.0:
                grid = np.geomspace(start, stop, grid_points)
            else:
                positive_starts = []
                for frame in histories:
                    positive = frame.loc[frame["time"] > 0.0, "time"]
                    if not positive.empty:
                        positive_starts.append(float(positive.min()))
                if positive_starts and max(positive_starts) < stop:
                    grid = np.concatenate(
                        [
                            np.asarray([0.0]),
                            np.geomspace(
                                max(positive_starts), stop, grid_points - 1
                            ),
                        ]
                    )
                else:
                    grid = np.linspace(start, stop, grid_points)
            for metric in ("validation_error", "test_error"):
                matrix = np.vstack(
                    [
                        _interpolate_incumbent(
                            frame["time"].to_numpy(dtype=float),
                            frame[metric].to_numpy(dtype=float),
                            grid,
                        )
                        for frame in histories
                    ]
                )
                counts = np.sum(np.isfinite(matrix), axis=0)
                for index, time_value in enumerate(grid):
                    values = matrix[:, index]
                    values = values[np.isfinite(values)]
                    if values.size == 0:
                        continue
                    rows.append(
                        {
                            "dataset": "splice",
                            "tol": float(tol),
                            "metric": metric,
                            "method": method.label,
                            "time": float(time_value),
                            "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1))
                            if values.size >= 2
                            else np.nan,
                            "runs_available": int(counts[index]),
                        }
                    )
    columns = [
        "dataset",
        "tol",
        "metric",
        "method",
        "time",
        "mean",
        "std",
        "runs_available",
    ]
    return pd.DataFrame(rows, columns=columns)


def plot_curves(
    root: Path,
    curves: pd.DataFrame,
    tolerances: Sequence[float],
    filename_suffix: str = "",
    comparison_note: Optional[str] = None,
    main_xlim: Optional[Tuple[float, float]] = None,
    plot_colors: Optional[Mapping[str, str]] = None,
    plot_line_styles: Optional[Mapping[str, str]] = None,
    title_text: Optional[str] = None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    inset_limits = {
        # The inset is centered on the LDPM convergence/plateau window,
        # rather than the much longer VF/LDMMA early-time span.
        (1e-4, "validation_error"): ((0.25, 0.80), (0.515, 0.565)),
        (1e-4, "test_error"): ((0.25, 0.80), (0.465, 0.510)),
        (1e-5, "validation_error"): ((0.0, 1.20), (0.552, 0.582)),
        (1e-5, "test_error"): ((0.0, 1.20), (0.485, 0.532)),
    }
    color_map = COLORS if plot_colors is None else plot_colors
    style_map = LINE_STYLES if plot_line_styles is None else plot_line_styles
    for tol in tolerances:
        for metric, ylabel in (
            ("validation_error", "Validation error"),
            ("test_error", "Test error"),
        ):
            fig, axis = plt.subplots(figsize=(12.0, 7.4))
            fig.subplots_adjust(
                left=0.09,
                right=0.975,
                bottom=0.145 if comparison_note else 0.11,
                top=0.83,
            )
            zoom_axis = inset_axes(
                axis,
                width="49%",
                height="49%",
                loc="upper right",
                borderpad=1.25,
            )
            plotted = 0
            for method in METHODS:
                block = curves[
                    np.isclose(curves["tol"].astype(float), float(tol))
                    & (curves["metric"] == metric)
                    & (curves["method"] == method.label)
                ].sort_values("time")
                if block.empty:
                    continue
                time_values = block["time"].to_numpy(dtype=float)
                mean = block["mean"].to_numpy(dtype=float)
                std = block["std"].to_numpy(dtype=float)
                color = color_map[method.label]
                axis.plot(
                    time_values,
                    mean,
                    label=DISPLAY_LABELS.get(method.label, method.label),
                    color=color,
                    linestyle=style_map[method.label],
                    linewidth=3.0,
                    solid_capstyle="round",
                )
                repeated = np.isfinite(std)
                if np.any(repeated):
                    lower = np.maximum(mean - np.nan_to_num(std, nan=0.0), 0.0)
                    upper = mean + np.nan_to_num(std, nan=0.0)
                    axis.fill_between(
                        time_values,
                        lower,
                        upper,
                        color=color,
                        alpha=0.09,
                        linewidth=0.0,
                    )
                zoom_axis.plot(
                    time_values,
                    mean,
                    color=color,
                    linestyle=style_map[method.label],
                    linewidth=2.2,
                    solid_capstyle="round",
                )
                if np.any(repeated):
                    zoom_axis.fill_between(
                        time_values,
                        lower,
                        upper,
                        color=color,
                        alpha=0.055,
                        linewidth=0.0,
                    )
                plotted += 1
            axis.set_xlabel("Running time (seconds)", fontsize=15)
            axis.set_ylabel(ylabel, fontsize=15)
            exponent = 4 if np.isclose(tol, 1e-4) else 5
            axis.set_title(
                title_text
                if title_text is not None
                else "splice, tol=$10^{-%d}$, %s"
                % (exponent, ylabel.lower()),
                fontsize=17,
                pad=12,
            )
            axis.grid(True, alpha=0.22, linewidth=0.8)
            axis.tick_params(axis="both", labelsize=12)
            if main_xlim is not None:
                axis.set_xlim(*main_xlim)
            zoom_key = next(
                key
                for key in inset_limits
                if np.isclose(float(key[0]), float(tol)) and key[1] == metric
            )
            (zoom_xmin, zoom_xmax), (zoom_ymin, zoom_ymax) = inset_limits[zoom_key]
            zoom_axis.set_xlim(zoom_xmin, zoom_xmax)
            zoom_axis.set_ylim(zoom_ymin, zoom_ymax)
            zoom_axis.set_title("Early-time zoom", fontsize=11, pad=5)
            zoom_axis.grid(True, alpha=0.20, linewidth=0.6)
            zoom_axis.tick_params(axis="both", labelsize=9)
            zoom_axis.xaxis.set_major_locator(MaxNLocator(4))
            zoom_axis.yaxis.set_major_locator(MaxNLocator(5))
            zoom_axis.set_xlabel("Time (s)", fontsize=9)
            mark_inset(
                axis,
                zoom_axis,
                loc1=2,
                loc2=4,
                fc="none",
                ec="#666666",
                linewidth=1.0,
            )
            if plotted:
                handles, labels = axis.get_legend_handles_labels()
                fig.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.53, 0.965),
                    ncol=max(1, plotted),
                    frameon=False,
                    fontsize=13,
                    handlelength=3.0,
                )
            if comparison_note:
                fig.text(
                    0.5,
                    0.035,
                    comparison_note,
                    ha="center",
                    va="bottom",
                    fontsize=11,
                    color="#444444",
                )
            token = "splice_tol%s_%s_time%s" % (
                _tol_token(tol),
                metric,
                filename_suffix,
            )
            fig.savefig(figure_dir / (token + ".png"), dpi=300, facecolor="white")
            fig.savefig(figure_dir / (token + ".pdf"))
            plt.close(fig)


def plot_composite_curves(
    root: Path, curves: pd.DataFrame, tolerances: Sequence[float]
) -> Path:
    """Write one large, paper-ready 2x2 PNG for easy inspection."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.115, top=0.885, wspace=0.16, hspace=0.24)
    metrics = (
        ("validation_error", "Validation error"),
        ("test_error", "Test error"),
    )
    for row_index, tol in enumerate(tolerances):
        for column_index, (metric, ylabel) in enumerate(metrics):
            axis = axes[row_index, column_index]
            panel = curves[
                np.isclose(curves["tol"].astype(float), float(tol))
                & (curves["metric"] == metric)
            ]
            for method in METHODS:
                block = panel[panel["method"] == method.label].sort_values("time")
                if block.empty:
                    continue
                time_values = block["time"].to_numpy(dtype=float)
                mean = block["mean"].to_numpy(dtype=float)
                std = block["std"].to_numpy(dtype=float)
                color = COLORS[method.label]
                axis.plot(
                    time_values,
                    mean,
                    color=color,
                    linestyle=LINE_STYLES[method.label],
                    linewidth=3.0,
                    solid_capstyle="round",
                )
                if np.any(np.isfinite(std)):
                    spread = np.nan_to_num(std, nan=0.0)
                    axis.fill_between(
                        time_values,
                        np.maximum(mean - spread, 0.0),
                        mean + spread,
                        color=color,
                        alpha=0.09,
                        linewidth=0.0,
                    )
            axis.grid(True, which="major", alpha=0.22, linewidth=0.8)
            axis.tick_params(axis="both", labelsize=12)
            axis.set_xlabel("Running time (seconds)", fontsize=14)
            axis.set_ylabel(ylabel, fontsize=14)
            panel_letter = chr(ord("a") + row_index * 2 + column_index)
            exponent = 4 if np.isclose(tol, 1e-4) else 5
            axis.set_title(
                "(%s) tol = $10^{-%d}$, %s"
                % (panel_letter, exponent, "validation" if metric == "validation_error" else "test"),
                fontsize=15,
                pad=10,
            )
    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method.label],
            linestyle=LINE_STYLES[method.label],
            linewidth=3.0,
            label=DISPLAY_LABELS.get(method.label, method.label),
        )
        for method in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=14,
        bbox_to_anchor=(0.53, 0.965),
        handlelength=3.0,
    )
    fig.suptitle(
        "Splice overlapping Group Lasso: error versus running time",
        fontsize=19,
        y=0.995,
    )
    fig.text(
        0.5,
        0.028,
        "Mean +/- one sample SD across ten repetitions. Validation-best incumbents; test error uses the same checkpoints. No EMA or spline smoothing.",
        ha="center",
        va="bottom",
        fontsize=11.5,
    )
    output = figure_dir / "splice_ogl_time_curves_composite.png"
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    return output


def _format_cell(row: pd.Series, metric: str) -> str:
    count = int(row[metric + "_valid_runs"])
    if count == 0:
        return "-- [0/10]"
    mean = float(row[metric + "_mean"])
    std = row[metric + "_std"]
    std_text = "--" if pd.isna(std) else "%.3g" % float(std)
    return "%.6g (%s) [%d/10]" % (mean, std_text, count)


def write_report(
    root: Path,
    summary: pd.DataFrame,
    seeds: Sequence[int],
    vfidca_initial_r: Optional[float] = None,
) -> None:
    if vfidca_initial_r is None:
        vfidca_radius_note = (
            "VF-iDCA uses the synthetic SGL example initial radius r0=10."
        )
    elif np.isclose(vfidca_initial_r, 0.1, rtol=0.0, atol=1e-15):
        vfidca_radius_note = (
            "VF-iDCA uses initial radius r0=0.1 from the released iP_DCA "
            "algorithm fallback; all other pinned upstream settings are unchanged."
        )
    else:
        vfidca_radius_note = (
            "VF-iDCA uses the explicitly supplied initial radius r0=%.6g; all "
            "other pinned upstream settings are unchanged." % vfidca_initial_r
        )
    lines = [
        "# splice direct overlapping Group Lasso: ten-repetition results",
        "",
        "Seeds: %s. Entries are mean (sample standard deviation) [valid runs/10]. "
        "Failed and timeout attempts remain in the audit data but are excluded from "
        "all numerical summaries." % ", ".join(map(str, seeds)),
        "",
        vfidca_radius_note,
        "",
        "Val./Test Err. and Test Err. Infeas. are MSE. Val./Test miscl. use "
        "each method's raw final iterate, matching the original splice table. Feasibility is the "
        "nonnegative lower-objective gap divided by the validation sample size. "
        "Common feasible lower re-solves and curve post-processing are excluded from Time.",
        "",
    ]
    for tol in TOLERANCES:
        block = summary[np.isclose(summary["tol"].astype(float), tol)]
        lines.extend(
            [
                "## tol = %.0e" % tol,
                "",
                "| Method | Status | Time | Val. Err. | Test Err. | Test Err. Infeas. | Feasibility | Val. miscl. | Test miscl. |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            row = block[block["method"] == method.label].iloc[0]
            lines.append(
                "| %s | %s [%d/10] | %s | %s | %s | %s | %s | %s | %s |"
                % (
                    DISPLAY_LABELS.get(method.label, method.label),
                    row["statuses"],
                    row["valid_runs"],
                    _format_cell(row, "time"),
                    _format_cell(row, "validation_error"),
                    _format_cell(row, "test_error"),
                    _format_cell(row, "test_error_infeasibility"),
                    _format_cell(row, "feasibility"),
                    _format_cell(row, "validation_misclassification"),
                    _format_cell(row, "test_misclassification"),
                )
            )
        capped = block[block["method"] == "LDPM-CS-C"].iloc[0]
        lines.extend(
            [
                "",
                "LDPM-CS-C cap activation: %d/%d valid runs reached beta_max=%.6g."
                % (capped["cap_reached_runs"], capped["valid_runs"], LDPM_CAP),
                "",
            ]
        )
    lines.extend(
        [
            "Curve notes:",
            "",
            "- About 40 geometrically spaced checkpoints are re-solved per run.",
            "- Validation curves are best-so-far incumbents. Test curves use the same validation-selected checkpoints.",
            "- Curves use a linear time axis and only piecewise-linear interpolation; no spline, EMA, or test-based selection is applied.",
            "- Within each tolerance, remaining LDMMA repetitions are skipped after its first 180-second timeout.",
            "",
        ]
    )
    (root / "report.md").write_text("\n".join(lines))


def _protocol(
    seeds: Sequence[int],
    tolerances: Sequence[float],
    settings_by_seed: Mapping[int, object],
    curve_points: int,
    curve_grid_points: int,
    vfidca_initial_r: Optional[float],
) -> Dict[str, object]:
    return {
        "dataset": "splice",
        "test_dataset": "splice_test",
        "repetitions": 10,
        "seeds": list(map(int, seeds)),
        "tolerances": list(map(float, tolerances)),
        "methods": [method.label for method in METHODS],
        "run_directory_pattern": "<results-dir>/seed<seed>_tol<token>",
        "groups": {
            "count": NUM_GROUPS,
            "construction": "length-10 sliding windows with stride 5",
            "indices": [
                group.tolist()
                for group in _load_problem(int(seeds[0]))[0].groups
            ],
        },
        "split_hashes": {
            str(seed): settings_by_seed[int(seed)].split_hash for seed in seeds
        },
        "dataset_fingerprint": settings_by_seed[int(seeds[0])].dataset_fingerprint,
        "ldpm": {
            "stop": "full-state relative change and consensus residual <= 1e-5",
            "stop_patience": 1,
            "record_interval": 1,
            "line_search": "single-run defaults",
        },
        "ldpm_capped": {"beta_max": LDPM_CAP, "cap_activation_audited": True},
        "ldmma": {"timeout_seconds": LDMMA_TIMEOUT},
        "vf_idca": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "fixed_sgl_config": UPSTREAM_SGL_FIXED_CONFIG,
            "effective_initial_r": (
                UPSTREAM_SGL_FIXED_CONFIG["initial_guess"]
                if vfidca_initial_r is None
                else float(vfidca_initial_r)
            ),
            "initial_r_source": (
                "upstream_synthetic_sgl_example"
                if vfidca_initial_r is None
                else (
                    "released_ipdca_algorithm_fallback"
                    if np.isclose(vfidca_initial_r, 0.1, rtol=0.0, atol=1e-15)
                    else "cli_override"
                )
            ),
            "tuned_on_splice": bool(
                vfidca_initial_r is not None
                and not np.isclose(vfidca_initial_r, 0.1, rtol=0.0, atol=1e-15)
            ),
            "validation_or_test_tuning": False,
        },
        "metric_definitions": {
            "validation_error": "MSE at a common feasible lower solution",
            "test_error": "MSE at the same common feasible lower solution",
            "test_error_infeasibility": "MSE at the method's raw final iterate",
            "feasibility": "max(phi(lambda,x_raw)-phi(lambda,x_feasible),0)/n_validation",
            "misclassification": "sign-threshold error at the method's raw final iterate, matching the original splice table",
            "feasible_misclassification": "supplemental sign-threshold error at the common feasible lower solution",
        },
        "common_lower_solve": {
            "objective": "half train MSE + sum_g lambda_g ||x_g||_2",
            "primary_solver": {
                "name": "CLARABEL",
                "tol_gap_abs": 1e-10,
                "tol_gap_rel": 1e-10,
                "tol_feas": 1e-10,
                "max_iter": 1000,
            },
            "fallback_solver": {
                "name": "SCS",
                "eps": 1e-8,
                "max_iters": 200000,
                "trigger": "CLARABEL failure or optimal_inaccurate",
            },
            "warm_start": False,
            "lambda_clipping": "clip negative entries only when negative_inf <= 1e-8*max(1,lambda_inf); otherwise invalidate post-processing",
            "excluded_from_algorithm_time": True,
        },
        "statistics": "mean and sample standard deviation (ddof=1), with per-metric valid-run counts",
        "invalid_run_policy": "failed/timeout/skipped_after_timeout/postprocess_failed rows retained in all_runs.csv but excluded from statistics",
        "ldmma_timeout_policy": "within each tolerance, stop remaining LDMMA repetitions after the first 180-second timeout",
        "curves": {
            "checkpoints_per_run": int(curve_points),
            "common_grid_points": int(curve_grid_points),
            "selection": "validation best-so-far; test from the identical validation-selected checkpoint",
            "interpolation": "piecewise linear only; no spline or EMA",
            "dispersion": "sample standard deviation",
        },
        "expected_ranking_policy": "audit only; never tune, filter, or alter test results to force an ordering",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.results_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    seeds = _parse_ints(args.seeds)
    tolerances = _parse_floats(args.tols)
    _validate_protocol(seeds, tolerances, args.workers)
    if args.curve_points_per_run <= 0 or args.curve_grid_points <= 0:
        raise ValueError("curve point counts must be positive")

    if not args.aggregate_only:
        run_campaign(
            root,
            seeds,
            tolerances,
            args.workers,
            args.resume,
            Path(args.vfidca_deps),
            args.vfidca_initial_r,
        )

    runs, per_run_curves, settings_by_seed = load_and_postprocess(
        root, seeds, tolerances, args.curve_points_per_run
    )
    summary = aggregate_runs(runs)
    curves = aggregate_curves(per_run_curves, tolerances, args.curve_grid_points)

    runs.to_csv(root / "all_runs.csv", index=False)
    summary.to_csv(root / "summary_mean_std.csv", index=False)
    per_run_curves.to_csv(root / "per_run_incumbent_curves.csv", index=False)
    curves.to_csv(root / "error_time_curves.csv", index=False)
    plot_curves(root, curves, tolerances)
    plot_composite_curves(root, curves, tolerances)
    write_report(root, summary, seeds, args.vfidca_initial_r)
    protocol = _protocol(
        seeds,
        tolerances,
        settings_by_seed,
        args.curve_points_per_run,
        args.curve_grid_points,
        args.vfidca_initial_r,
    )
    (root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print("Saved splice ten-repetition campaign under %s" % root, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
