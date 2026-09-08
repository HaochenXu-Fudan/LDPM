#!/usr/bin/env python3
"""Run the manuscript LDPM scalability experiment.

The default contract follows ``INFORMS-IJOC-Template.tex``: one paired
LDPM-PG/LDPM-CS run at each p in {300, 600, 1200, 2400, 3600, 4800, 6000},
contiguous groups of size 30, fixed train/validation/test sizes 150/25/25,
SNR 2, and a 1e-5 complete-state stopping tolerance. Legacy campaign
selectors remain available explicitly for auditing older artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


# The experiment contract requires single-threaded numerical kernels.  These
# variables must be fixed before NumPy (and therefore BLAS) is imported.
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _thread_name, _thread_value in THREAD_ENVIRONMENT.items():
    os.environ[_thread_name] = _thread_value
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ldpm_scalability_matplotlib_cache")

import numpy as np
import pandas as pd

from data import Data, Data_with_Info
from methods import LeastSquaresLDPM, group_regularizers
from problem import MatrixSparseGroupLassoProblem


HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = HERE / "results" / "ldpm_scalability"
FIXED_M5_RESULTS_DIR = HERE / "results" / "ldpm_scalability_fixed_m5"
FIXED_M5_FIXED_N_RESULTS_DIR = (
    HERE / "results" / "ldpm_scalability_fixed_m5_fixed_n"
)
FIXED_M5_FIXED_N_PG_FULL_Z_RESULTS_DIR = (
    HERE / "results" / "ldpm_scalability_fixed_m5_fixed_n_pg_full_z"
)
FIXED_M5_FIXED_N_PG_FULL_Z_TOL_1E5_RESULTS_DIR = (
    HERE
    / "results"
    / "ldpm_scalability_fixed_m5_fixed_n_pg_full_z_tol_1e-5"
)
FIXED_M5_FIXED_N_PG_BLOCKWISE_TOL_1E5_RESULTS_DIR = (
    HERE
    / "results"
    / "ldpm_scalability_fixed_m5_fixed_n_pg_blockwise_tol_1e-5"
)
FIXED_GROUP_SIZE_30_FIXED_N_PG_BLOCKWISE_TOL_1E5_RESULTS_DIR = (
    HERE
    / "results"
    / "ldpm_scalability_fixed_group_size_30_fixed_n_pg_blockwise_tol_1e-5"
)
FULL_Z_BACKUP_SUFFIX = "_full_z_backup"

DEFAULT_P_LIST: Tuple[int, ...] = (300, 600, 1200, 2400, 3600, 4800, 6000)
FIXED_GROUP_SIZE_30_P_LIST: Tuple[int, ...] = DEFAULT_P_LIST
P_LIST: Tuple[int, ...] = DEFAULT_P_LIST
BASELINE_GROUP_SIZE = 60
FIXED_M5_GROUP_COUNT = 5
FIXED_GROUP_SIZE_30 = 30
USE_FIXED_M5 = False
USE_FIXED_GROUP_SIZE_30 = False
USE_FIXED_N = False
USE_PG_FULL_Z = False
USE_PG_TIGHT_TOL = False
USE_PG_BLOCKWISE = False
FIXED_SAMPLE_SIZES: Tuple[int, int, int] = (150, 25, 25)
BLOCKWISE_PG_PLOT_EXCLUDED_P: Tuple[int, ...] = ()
PLOT_OVERRIDES_FILENAME = "plot_overrides.csv"
N_REPEATS = 1
BASE_SEED = 20260726
TOL = 1e-4
PG_TIGHT_TOL = 1e-5
PG_MAX_OUTER_ITER = 10000
FIXED_GROUP_SIZE_30_PG_MAX_OUTER_ITER = 20000
CS_MAX_OUTER_ITER = 5000
TIME_LIMIT_SEC = 600.0
RECORD_INTERVAL = 250
LL_RESOLVE_MAX_ITER = 50000
LL_RESOLVE_TOL = 1e-10
TARGET_SNR = 2.0

METHODS: Tuple[str, ...] = ("LDPM-PG", "LDPM-CS")
METHOD_ORDER = {method: index for index, method in enumerate(METHODS)}
STATUS_VALUES = {
    "converged",
    "time_limit",
    "max_outer_iter",
    "numerical_failure",
    "exception",
}

RAW_COLUMNS: Tuple[str, ...] = (
    "method",
    "model",
    "p",
    "M",
    "group_size",
    "num_hyperparameters",
    "n_train",
    "n_validation",
    "n_test",
    "rep",
    "seed",
    "dataset_path",
    "dataset_hash",
    "actual_snr",
    "status",
    "algorithm_status",
    "stop_metric",
    "termination_message",
    "outer_iterations",
    "algorithm_runtime_sec",
    "postprocess_runtime_sec",
    "total_runtime_sec",
    "seconds_per_outer_iteration",
    "total_backtracking_trials",
    "mean_backtracking_trials_per_outer_iter",
    "final_r_step",
    "final_blockwise_stop",
    "final_x_lambda_stop",
    "final_stop_metric_value",
    "final_r_cons",
    "final_psi",
    "final_ll_feasibility",
    "final_beta",
    "final_step_size",
    "min_accepted_step_size",
    "max_accepted_step_size",
    "total_local_projections",
    "val_error_feasible",
    "test_error_feasible",
    "test_error_infeasible",
    "ll_objective_iterate",
    "ll_objective_resolved",
    "ll_gap_relative",
    "ll_gap_relative_raw",
    "ll_resolve_status",
    "ll_resolve_iterations",
    "ll_resolve_relative_step",
    "postprocess_failure",
    "native_lambda_min",
    "resolved_lambda_min",
    "nan_or_inf_detected",
    "history_path",
    "config_json",
    "started_at",
    "finished_at",
    "git_commit",
)
ADDITIVE_RAW_COLUMNS = {
    "stop_metric",
    "final_blockwise_stop",
    "final_x_lambda_stop",
    "final_stop_metric_value",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_for(p: int, rep: int) -> int:
    return int(BASE_SEED + 10000 * int(rep) + int(p))


def group_count_for(p: int) -> int:
    p = int(p)
    if USE_FIXED_M5:
        return FIXED_M5_GROUP_COUNT
    if USE_FIXED_GROUP_SIZE_30:
        if p % FIXED_GROUP_SIZE_30:
            raise ValueError(
                "p=%d is not divisible by fixed group size %d"
                % (p, FIXED_GROUP_SIZE_30)
            )
        return p // FIXED_GROUP_SIZE_30
    if p % BASELINE_GROUP_SIZE:
        raise ValueError(
            "p=%d is not divisible by baseline group size %d"
            % (p, BASELINE_GROUP_SIZE)
        )
    return p // BASELINE_GROUP_SIZE


def group_size_for(p: int) -> int:
    p = int(p)
    group_count = group_count_for(p)
    if p % group_count:
        raise ValueError(
            "p=%d is not divisible by group count %d" % (p, group_count)
        )
    return p // group_count


def sample_sizes(p: int) -> Tuple[int, int, int]:
    p = int(p)
    if p not in P_LIST:
        raise ValueError("p=%d is outside the prescribed P_LIST" % p)
    if USE_FIXED_N:
        return FIXED_SAMPLE_SIZES
    if p % 12:
        raise ValueError("p=%d does not give integral p/12 sample sizes" % p)
    return p // 4, p // 12, p // 12


def pg_uses_full_z() -> bool:
    return USE_PG_FULL_Z or USE_PG_TIGHT_TOL


def pg_stop_metric() -> str:
    if USE_PG_BLOCKWISE:
        return "blockwise"
    return "full_z" if pg_uses_full_z() else "x_lambda"


def pg_stop_tolerance() -> float:
    return PG_TIGHT_TOL if (USE_PG_TIGHT_TOL or USE_PG_BLOCKWISE) else TOL


def pg_max_outer_iter() -> int:
    return (
        FIXED_GROUP_SIZE_30_PG_MAX_OUTER_ITER
        if USE_FIXED_GROUP_SIZE_30
        else PG_MAX_OUTER_ITER
    )


def plot_only_excluded_pg_dimensions() -> Tuple[int, ...]:
    return BLOCKWISE_PG_PLOT_EXCLUDED_P if USE_PG_BLOCKWISE else ()


def plot_overrides_path(output_dir: Path) -> Optional[Path]:
    if USE_FIXED_GROUP_SIZE_30:
        return None
    path = output_dir / PLOT_OVERRIDES_FILENAME
    return path if USE_PG_BLOCKWISE and path.exists() else None


def format_tolerance(value: float) -> str:
    return ("%.0e" % float(value)).replace("e-0", "e-").replace("e+0", "e+")


def default_results_dir() -> Path:
    if USE_FIXED_GROUP_SIZE_30:
        return FIXED_GROUP_SIZE_30_FIXED_N_PG_BLOCKWISE_TOL_1E5_RESULTS_DIR
    if USE_PG_BLOCKWISE:
        return FIXED_M5_FIXED_N_PG_BLOCKWISE_TOL_1E5_RESULTS_DIR
    if USE_PG_TIGHT_TOL:
        return FIXED_M5_FIXED_N_PG_FULL_Z_TOL_1E5_RESULTS_DIR
    if pg_uses_full_z():
        return FIXED_M5_FIXED_N_PG_FULL_Z_RESULTS_DIR
    if USE_FIXED_N:
        return FIXED_M5_FIXED_N_RESULTS_DIR
    if USE_FIXED_M5:
        return FIXED_M5_RESULTS_DIR
    return DEFAULT_RESULTS_DIR


def dataset_format_version() -> int:
    if USE_FIXED_GROUP_SIZE_30:
        return 4
    if USE_FIXED_N:
        return 3
    if USE_FIXED_M5:
        return 2
    return 1


def dataset_group_structure_mode() -> str:
    if USE_FIXED_GROUP_SIZE_30:
        return "fixed_group_size_30"
    if USE_FIXED_M5:
        return "fixed_M_5"
    return "fixed_group_size_60"


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value: object) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(handle) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        _fsync_file(handle)
    os.replace(str(temporary), str(path))


def atomic_write_json(path: Path, payload: object) -> None:
    text = json.dumps(
        _json_ready(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(path, text + "\n")


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False)
        _fsync_file(handle)
    os.replace(str(temporary), str(path))


def atomic_write_npz(path: Path, arrays: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        _fsync_file(handle)
    os.replace(str(temporary), str(path))


def git_commit() -> Optional[str]:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(HERE),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for package in ("numpy", "pandas", "scipy", "matplotlib", "cvxpy", "clarabel"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_contract() -> None:
    expected_p_list = (
        FIXED_GROUP_SIZE_30_P_LIST
        if USE_FIXED_GROUP_SIZE_30
        else DEFAULT_P_LIST
    )
    if tuple(P_LIST) != expected_p_list:
        raise AssertionError("P_LIST changed from the prescribed dimensions")
    if N_REPEATS != 1:
        raise AssertionError("the explicit user override requires exactly rep=0")
    for p in P_LIST:
        n_train, n_validation, n_test = sample_sizes(p)
        m = group_count_for(p)
        group_size = group_size_for(p)
        expected_m = (
            FIXED_M5_GROUP_COUNT
            if USE_FIXED_M5
            else (
                p // FIXED_GROUP_SIZE_30
                if USE_FIXED_GROUP_SIZE_30
                else p // BASELINE_GROUP_SIZE
            )
        )
        if m != expected_m:
            raise AssertionError("invalid group structure at p=%d" % p)
        if m < 5 or m * group_size != p:
            raise AssertionError("groups do not partition p=%d" % p)
        if group_size < 10:
            raise AssertionError(
                "group size %d is too small for the fifth active block at p=%d"
                % (group_size, p)
            )
        expected_sizes = FIXED_SAMPLE_SIZES if USE_FIXED_N else (
            p // 4,
            p // 12,
            p // 12,
        )
        if (n_train, n_validation, n_test) != expected_sizes:
            raise AssertionError("invalid sample sizes at p=%d" % p)


def _dataset_file(output_dir: Path, p: int, rep: int) -> Path:
    return output_dir / "datasets" / ("dataset_p%d_rep%d.npz" % (p, rep))


def _active_coefficients_by_group(
    beta: np.ndarray, m: int, group_size: int
) -> Tuple[int, ...]:
    return tuple(
        int(
            np.count_nonzero(
                beta[
                    group_index * group_size : (group_index + 1) * group_size
                ]
            )
        )
        for group_index in range(m)
    )


def _generate_dataset_arrays(p: int, rep: int) -> Dict[str, np.ndarray]:
    n_train, n_validation, n_test = sample_sizes(p)
    m = group_count_for(p)
    group_size = group_size_for(p)
    seed = seed_for(p, rep)
    rng = np.random.default_rng(seed)

    a_train = np.asarray(rng.standard_normal((n_train, p)), dtype=np.float64)
    a_validation = np.asarray(
        rng.standard_normal((n_validation, p)), dtype=np.float64
    )
    a_test = np.asarray(rng.standard_normal((n_test, p)), dtype=np.float64)

    beta_star = np.zeros(p, dtype=np.float64)
    for group_index in range(1, 6):
        start = (group_index - 1) * group_size
        active = 2 * group_index
        beta_star[start : start + active] = float(2 * group_index)
    if int(np.count_nonzero(beta_star)) != 30:
        raise AssertionError("beta_star must have exactly 30 nonzero entries")
    if USE_FIXED_M5 or USE_FIXED_GROUP_SIZE_30:
        expected_active_counts = (2, 4, 6, 8, 10) + (0,) * (m - 5)
        actual_active_counts = _active_coefficients_by_group(
            beta_star, m, group_size
        )
        if actual_active_counts != expected_active_counts:
            raise AssertionError(
                "beta_star active counts by group are %s, expected %s"
                % (actual_active_counts, expected_active_counts)
            )

    epsilon_train = np.asarray(rng.standard_normal(n_train), dtype=np.float64)
    epsilon_validation = np.asarray(
        rng.standard_normal(n_validation), dtype=np.float64
    )
    epsilon_test = np.asarray(rng.standard_normal(n_test), dtype=np.float64)

    signal_train = a_train @ beta_star
    signal_validation = a_validation @ beta_star
    signal_test = a_test @ beta_star
    signal_all = np.concatenate((signal_train, signal_validation, signal_test))
    noise_all = np.concatenate(
        (epsilon_train, epsilon_validation, epsilon_test)
    )
    sigma = float(
        np.linalg.norm(signal_all)
        / (TARGET_SNR * max(float(np.linalg.norm(noise_all)), 1e-300))
    )
    b_train = signal_train + sigma * epsilon_train
    b_validation = signal_validation + sigma * epsilon_validation
    b_test = signal_test + sigma * epsilon_test
    response_all = np.concatenate((b_train, b_validation, b_test))
    actual_snr = float(
        np.linalg.norm(signal_all)
        / max(float(np.linalg.norm(response_all - signal_all)), 1e-300)
    )
    relative_snr_error = abs(actual_snr - TARGET_SNR) / TARGET_SNR
    if relative_snr_error >= 1e-10:
        raise RuntimeError(
            "actual SNR %.17g misses target %.17g" % (actual_snr, TARGET_SNR)
        )

    group_starts = np.arange(m, dtype=np.int64) * group_size
    group_ends = group_starts + group_size
    return {
        "A_tr": a_train,
        "b_tr": np.asarray(b_train, dtype=np.float64),
        "A_val": a_validation,
        "b_val": np.asarray(b_validation, dtype=np.float64),
        "A_test": a_test,
        "b_test": np.asarray(b_test, dtype=np.float64),
        "beta_star": beta_star,
        "group_starts": group_starts,
        "group_ends": group_ends,
        "p": np.asarray(p, dtype=np.int64),
        "M": np.asarray(m, dtype=np.int64),
        "group_size": np.asarray(group_size, dtype=np.int64),
        "n_train": np.asarray(n_train, dtype=np.int64),
        "n_validation": np.asarray(n_validation, dtype=np.int64),
        "n_test": np.asarray(n_test, dtype=np.int64),
        "rep": np.asarray(rep, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "target_snr": np.asarray(TARGET_SNR, dtype=np.float64),
        "actual_snr": np.asarray(actual_snr, dtype=np.float64),
        "sigma": np.asarray(sigma, dtype=np.float64),
        "format_version": np.asarray(dataset_format_version(), dtype=np.int64),
        "sample_size_mode": np.asarray(
            "fixed_75_25_25" if USE_FIXED_N else "proportional_to_p"
        ),
        "group_structure_mode": np.asarray(dataset_group_structure_mode()),
    }


def _load_npz_arrays(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _validate_dataset_arrays(
    arrays: Mapping[str, np.ndarray], p: int, rep: int
) -> Dict[str, object]:
    required = {
        "A_tr",
        "b_tr",
        "A_val",
        "b_val",
        "A_test",
        "b_test",
        "beta_star",
        "group_starts",
        "group_ends",
        "p",
        "M",
        "group_size",
        "n_train",
        "n_validation",
        "n_test",
        "rep",
        "seed",
        "actual_snr",
        "format_version",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError("dataset cache is missing: %s" % ", ".join(missing))
    if USE_FIXED_N:
        if "sample_size_mode" not in arrays:
            raise ValueError("fixed-n dataset cache is missing sample_size_mode")
        sample_size_mode = str(np.asarray(arrays["sample_size_mode"]).item())
        if sample_size_mode != "fixed_75_25_25":
            raise ValueError(
                "fixed-n dataset cache has sample_size_mode=%r"
                % sample_size_mode
            )
    if USE_FIXED_GROUP_SIZE_30:
        if "group_structure_mode" not in arrays:
            raise ValueError(
                "fixed-group-size-30 dataset cache is missing "
                "group_structure_mode"
            )
        group_structure_mode = str(
            np.asarray(arrays["group_structure_mode"]).item()
        )
        if group_structure_mode != "fixed_group_size_30":
            raise ValueError(
                "fixed-group-size-30 dataset cache has group_structure_mode=%r"
                % group_structure_mode
            )
    n_train, n_validation, n_test = sample_sizes(p)
    m = group_count_for(p)
    group_size = group_size_for(p)
    expected_scalars = {
        "p": p,
        "M": m,
        "group_size": group_size,
        "n_train": n_train,
        "n_validation": n_validation,
        "n_test": n_test,
        "rep": rep,
        "seed": seed_for(p, rep),
        "format_version": dataset_format_version(),
    }
    for name, expected in expected_scalars.items():
        actual = int(np.asarray(arrays[name]).item())
        if actual != expected:
            raise ValueError(
                "dataset %s=%d, expected %d" % (name, actual, expected)
            )

    expected_shapes = {
        "A_tr": (n_train, p),
        "b_tr": (n_train,),
        "A_val": (n_validation, p),
        "b_val": (n_validation,),
        "A_test": (n_test, p),
        "b_test": (n_test,),
        "beta_star": (p,),
        "group_starts": (m,),
        "group_ends": (m,),
    }
    for name, shape in expected_shapes.items():
        if tuple(np.asarray(arrays[name]).shape) != shape:
            raise ValueError(
                "dataset %s shape=%s, expected %s"
                % (name, np.asarray(arrays[name]).shape, shape)
            )
    for name in ("A_tr", "b_tr", "A_val", "b_val", "A_test", "b_test", "beta_star"):
        if np.asarray(arrays[name]).dtype != np.float64:
            raise ValueError("dataset %s must be float64" % name)
    if int(np.count_nonzero(arrays["beta_star"])) != 30:
        raise ValueError("cached beta_star does not have 30 nonzeros")
    if USE_FIXED_M5 or USE_FIXED_GROUP_SIZE_30:
        expected_active_counts = (2, 4, 6, 8, 10) + (0,) * (m - 5)
        actual_active_counts = _active_coefficients_by_group(
            np.asarray(arrays["beta_star"]), m, group_size
        )
        if actual_active_counts != expected_active_counts:
            raise ValueError(
                "cached beta_star active counts by group are %s, expected %s"
                % (actual_active_counts, expected_active_counts)
            )
    expected_starts = np.arange(m, dtype=np.int64) * group_size
    if not np.array_equal(arrays["group_starts"], expected_starts):
        raise ValueError("cached group starts do not match the current partition")
    if not np.array_equal(arrays["group_ends"], expected_starts + group_size):
        raise ValueError("cached group ends do not match the current partition")

    signal = np.concatenate(
        (
            arrays["A_tr"] @ arrays["beta_star"],
            arrays["A_val"] @ arrays["beta_star"],
            arrays["A_test"] @ arrays["beta_star"],
        )
    )
    response = np.concatenate((arrays["b_tr"], arrays["b_val"], arrays["b_test"]))
    actual_snr = float(
        np.linalg.norm(signal)
        / max(float(np.linalg.norm(response - signal)), 1e-300)
    )
    if abs(actual_snr - TARGET_SNR) / TARGET_SNR >= 1e-10:
        raise ValueError("cached dataset actual SNR is %.17g" % actual_snr)
    recorded_snr = float(np.asarray(arrays["actual_snr"]).item())
    if abs(recorded_snr - actual_snr) > 1e-12 * max(1.0, abs(actual_snr)):
        raise ValueError("cached actual_snr metadata does not match its arrays")

    return {
        **expected_scalars,
        "actual_snr": actual_snr,
    }


def load_or_create_dataset(
    output_dir: Path, p: int, rep: int
) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    path = _dataset_file(output_dir, p, rep)
    if path.exists():
        arrays = _load_npz_arrays(path)
    else:
        arrays = _generate_dataset_arrays(p, rep)
        atomic_write_npz(path, arrays)
    metadata = _validate_dataset_arrays(arrays, p, rep)
    dataset_hash = sha256_file(path)
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if hash_path.exists():
        recorded_hash = hash_path.read_text(encoding="utf-8").strip()
        if recorded_hash != dataset_hash:
            raise ValueError("dataset hash sidecar does not match %s" % path)
    else:
        atomic_write_text(hash_path, dataset_hash + "\n")
    metadata.update(
        dataset_path=str(path.resolve()),
        dataset_hash=dataset_hash,
    )
    return arrays, metadata


def prepare_variant_datasets(output_dir: Path) -> None:
    """Copy byte-identical datasets from the directly preceding campaign."""

    if USE_FIXED_GROUP_SIZE_30:
        # The changed partition also changes beta_star locations and responses;
        # this campaign must never inherit a fixed-M5 dataset cache.
        return
    if USE_PG_BLOCKWISE:
        source_results_dir = FIXED_M5_FIXED_N_PG_FULL_Z_TOL_1E5_RESULTS_DIR
    elif USE_PG_TIGHT_TOL:
        source_results_dir = FIXED_M5_FIXED_N_PG_FULL_Z_RESULTS_DIR
    else:
        return
    if output_dir.resolve() == source_results_dir.resolve():
        raise RuntimeError(
            "the PG stopping-criterion variant cannot overwrite its source "
            "campaign directory"
        )
    destination_directory = output_dir / "datasets"
    for p in P_LIST:
        source = _dataset_file(
            source_results_dir,
            p,
            0,
        )
        source_hash_path = source.with_suffix(source.suffix + ".sha256")
        if not source.exists() or not source_hash_path.exists():
            raise FileNotFoundError(
                "variant source dataset or hash is missing: %s"
                % source
            )
        expected_hash = source_hash_path.read_text(encoding="utf-8").strip()
        if sha256_file(source) != expected_hash:
            raise ValueError(
                "variant source dataset hash sidecar is stale: %s"
                % source
            )

        destination = destination_directory / source.name
        destination_hash_path = destination.with_suffix(
            destination.suffix + ".sha256"
        )
        if destination.exists():
            if sha256_file(destination) != expected_hash:
                raise ValueError(
                    "existing variant dataset differs from source: %s"
                    % destination
                )
        else:
            shutil.copy2(source, destination)
        if destination_hash_path.exists():
            recorded_hash = destination_hash_path.read_text(
                encoding="utf-8"
            ).strip()
            if recorded_hash != expected_hash:
                raise ValueError(
                    "existing variant hash sidecar differs from source: "
                    "%s" % destination_hash_path
                )
        else:
            shutil.copy2(source_hash_path, destination_hash_path)


def data_info_from_arrays(
    arrays: Mapping[str, np.ndarray], metadata: Mapping[str, object]
) -> Data_with_Info:
    data = Data()
    data.X_train = arrays["A_tr"]
    data.y_train = arrays["b_tr"]
    data.X_validate = arrays["A_val"]
    data.y_validate = arrays["b_val"]
    data.X_test = arrays["A_test"]
    data.y_test = arrays["b_test"]
    data.true_beta = arrays["beta_star"]
    data.sigma = float(np.asarray(arrays["sigma"]).item())
    data.realized_snr = float(metadata["actual_snr"])
    settings = SimpleNamespace(
        num_train=int(metadata["n_train"]),
        num_validate=int(metadata["n_validation"]),
        num_test=int(metadata["n_test"]),
        num_features=int(metadata["p"]),
        num_experiment_groups=int(metadata["M"]),
        num_true_groups=5,
        dataset="ldpm_scalability_p%d_rep%d"
        % (int(metadata["p"]), int(metadata["rep"])),
    )
    return Data_with_Info(data, settings, data_index=int(metadata["seed"]))


def regularizers_for(method: str, p: int, m: int) -> List[dict]:
    regularizers = group_regularizers(p, m)
    if method == "LDPM-CS":
        regularizers.append({"type": "l1", "slice": slice(None)})
    elif method != "LDPM-PG":
        raise ValueError("unsupported method %r" % method)
    return regularizers


def algorithm_contract(method: str, p: int, m: int) -> Dict[str, object]:
    common: Dict[str, object] = {
        "method": method,
        "uncapped_continuation": True,
        "beta_max": None,
        "TOL": TOL,
        "time_limit_sec": TIME_LIMIT_SEC,
        "stop_metric": "full_z",
        "stop_patience": 1,
        "normalize_loss": True,
        "sqrt_loss_scaling": True,
        "reduced_dual": False,
        "record_interval": RECORD_INTERVAL,
        "line_search": True,
        "line_search_min_step": 1e-12,
        "line_search_decay": 0.5,
        "line_search_growth": 1.25,
        "max_line_search_iter": 50,
        "projection_max_sweeps": 100,
        "projection_tol": 1e-7,
        # The prepared state is created once for one solver call.  Reusing it
        # directly keeps initialization/copying outside algorithm timing and
        # avoids duplicating the large LDPM-CS local-copy state at high p.
        "copy_prepared_state": False,
        "save_local_copies": False,
    }
    if method == "LDPM-PG":
        selected_pg_stop_metric = pg_stop_metric()
        if USE_PG_BLOCKWISE:
            pg_parameter_source = (
                "INFORMS-IJOC-Template.tex fixes the manuscript contract and the initial "
                "trial step to 0.01 and pins beta0=1,q=0.3; all other "
                "initialization/backtracking settings follow "
                "synthetic.py; the latest explicit "
                "follow-up changes only the PG stop to the maximum separately "
                "normalized change of x, r, lambda, rho and xi at 1e-5"
            )
        elif pg_uses_full_z():
            pg_parameter_source = (
                "INFORMS-IJOC-Template.tex fixes the manuscript contract and the initial "
                "trial step to 0.01 and pins beta0=1,q=0.3; all other "
                "initialization/backtracking settings follow "
                "synthetic.py; the latest explicit "
                "user instruction restores the PG stop metric from x_lambda "
                "to the relative change of the complete state full_z"
            )
        else:
            pg_parameter_source = (
                "INFORMS-IJOC-Template.tex fixes the manuscript contract and the initial "
                "trial step to 0.01 and pins beta0=1,q=0.3; all other "
                "initialization/backtracking settings follow "
                "synthetic.py; the latest explicit "
                "user instruction changes only the PG stop metric from "
                "full_z to x_lambda"
            )
        if USE_PG_TIGHT_TOL:
            pg_parameter_source += (
                "; the latest explicit follow-up tightens only the LDPM-PG "
                "full_z tolerance from 1e-4 to 1e-5"
            )
        common.update(
            {
                "model": "Group Lasso",
                "MAX_ITERATION": pg_max_outer_iter(),
                "TOL": pg_stop_tolerance(),
                "stop_metric": selected_pg_stop_metric,
                "parameter_source": pg_parameter_source,
                "step_size": 0.01,
                "line_search_max_step": 0.01,
                "beta0": 1.0,
                "beta_power": 0.3,
                "gamma": None,
                "initial_coef": "ones",
                "initial_lambda": [1.0] * m,
                "init_dual": "fenchel",
                "init_max_iter": 300,
                "init_tol": 1e-7,
                "num_hyperparameters": m,
            }
        )
    elif method == "LDPM-CS":
        common.update(
            {
                "model": "Sparse Group Lasso",
                "MAX_ITERATION": CS_MAX_OUTER_ITER,
                "parameter_source": (
                    (
                        "SGL/experiment.py formal configuration with five group "
                        "penalties; the common scalability stopping/budget "
                        "contract overrides its stop"
                    )
                    if USE_FIXED_M5
                    else (
                        "SGL/experiment.py formal configuration, generalized from five group "
                        "penalties to M group penalties; the common scalability "
                        "stopping/budget contract overrides its stop"
                    )
                ),
                "step_size": 0.1,
                "line_search_max_step": 0.1,
                "beta0": 0.03,
                "beta_power": 1.2,
                "gamma": 1.0,
                "initial_coef": "lower",
                "initial_lambda": [0.01] * m + [0.75],
                "init_dual": "kkt",
                "init_max_iter": 5000,
                "init_tol": 1e-10,
                "num_hyperparameters": m + 1,
            }
        )
    else:
        raise ValueError("unsupported method %r" % method)
    return common


def solver_setting(contract: Mapping[str, object], p: int) -> Dict[str, object]:
    setting = {
        key: value
        for key, value in contract.items()
        if key
        not in {
            "method",
            "model",
            "parameter_source",
            "uncapped_continuation",
            "num_hyperparameters",
            "initial_coef",
        }
    }
    setting["initial_lambda"] = np.asarray(contract["initial_lambda"], dtype=float)
    if contract["initial_coef"] == "ones":
        setting["initial_coef"] = np.ones(p, dtype=float)
    elif contract["initial_coef"] != "lower":
        raise ValueError("unknown initial coefficient rule")
    return setting


def _loss(matrix: np.ndarray, response: np.ndarray, coefficient: np.ndarray) -> float:
    residual = matrix @ coefficient - response
    return float(0.5 * np.dot(residual, residual) / response.size)


def postprocess_lower_level(
    data_info: Data_with_Info,
    regularizers: Sequence[dict],
    native_lambda: np.ndarray,
    iterate: np.ndarray,
    max_iter: int = LL_RESOLVE_MAX_ITER,
    tol: float = LL_RESOLVE_TOL,
) -> Dict[str, object]:
    started = time.perf_counter()
    resolved_lambda = np.maximum(np.asarray(native_lambda, dtype=float), 0.0)
    problem = MatrixSparseGroupLassoProblem(
        data_info,
        regularizers,
        {"lower_max_iter": int(max_iter), "lower_tol": float(tol)},
    )
    resolved, iterations = problem.lower_solve(
        resolved_lambda,
        x0=iterate,
        max_iter=int(max_iter),
        tol=float(tol),
    )
    step = 1.0 / problem.train_lipschitz
    prox_point = problem.prox(
        resolved - step * problem.train_grad(resolved),
        step,
        resolved_lambda,
    )
    relative_step = float(
        np.linalg.norm(prox_point - resolved)
        / max(1.0, float(np.linalg.norm(resolved)))
    )
    if iterations < int(max_iter):
        resolve_status = "converged"
    elif np.isfinite(relative_step) and relative_step <= float(tol):
        resolve_status = "converged_at_iteration_limit"
    else:
        resolve_status = "max_iter"

    objective_iterate = problem.lower_objective(resolved_lambda, iterate)
    objective_resolved = problem.lower_objective(resolved_lambda, resolved)
    gap_raw = float(
        (objective_iterate - objective_resolved)
        / max(1.0, abs(float(objective_resolved)))
    )
    gap_reported = max(gap_raw, 0.0) if gap_raw > -1e-10 else gap_raw
    finite = bool(
        np.all(np.isfinite(resolved))
        and np.all(np.isfinite(resolved_lambda))
        and np.isfinite(relative_step)
        and np.isfinite(objective_iterate)
        and np.isfinite(objective_resolved)
        and np.isfinite(gap_raw)
    )
    postprocess_failure = (not finite) or gap_raw < -1e-10
    elapsed = time.perf_counter() - started
    return {
        "resolved_coef": np.asarray(resolved, dtype=float),
        "resolved_lambda": resolved_lambda,
        "postprocess_runtime_sec": float(elapsed),
        "ll_objective_iterate": float(objective_iterate),
        "ll_objective_resolved": float(objective_resolved),
        "ll_gap_relative": float(gap_reported),
        "ll_gap_relative_raw": float(gap_raw),
        "ll_resolve_status": resolve_status,
        "ll_resolve_iterations": int(iterations),
        "ll_resolve_relative_step": float(relative_step),
        "postprocess_failure": bool(postprocess_failure),
        "val_error_feasible": problem.validation_loss(resolved),
        "test_error_feasible": problem.test_loss(resolved),
        "test_error_infeasible": problem.test_loss(iterate),
    }


def _history_column(
    history: pd.DataFrame, name: str, default: float = np.nan
) -> np.ndarray:
    if name not in history:
        return np.full(len(history), default, dtype=float)
    return pd.to_numeric(history[name], errors="coerce").to_numpy(dtype=float)


def _history_payload(
    history: pd.DataFrame,
    method: str,
    dataset_hash: str,
    contract_json: str,
    iterate: np.ndarray,
    native_lambda: np.ndarray,
    postprocess: Optional[Mapping[str, object]],
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "runtime_cumulative_sec": _history_column(history, "time"),
        "outer_iteration": _history_column(history, "iteration").astype(np.int64),
        "r_step": _history_column(history, "full_z_relative_step"),
        "blockwise_stop": _history_column(history, "blockwise_stop"),
        "x_lambda_stop": _history_column(history, "x_lambda_stop"),
        "selected_stop_metric": _history_column(history, "stop_metric"),
        "r_cons": (
            _history_column(history, "r_cons")
            if method == "LDPM-CS"
            else np.full(len(history), np.nan, dtype=float)
        ),
        "psi": _history_column(history, "ll_duality_gap"),
        "beta_k": _history_column(history, "beta"),
        "accepted_step_size": _history_column(history, "accepted_step_size"),
        "backtracking_trials": _history_column(
            history, "backtracking_trials", default=0.0
        ).astype(np.int64),
        "validation_error": _history_column(history, "validation_error"),
        "test_error": _history_column(history, "test_error"),
        "ll_feasibility": _history_column(history, "ll_feasibility"),
        "all_accepted_step_sizes": np.asarray(
            history.attrs.get("accepted_step_sizes", []), dtype=float
        ),
        "final_iterate": np.asarray(iterate, dtype=float),
        "final_lambda_native": np.asarray(native_lambda, dtype=float),
        "dataset_hash": np.asarray(dataset_hash),
        "config_json": np.asarray(contract_json),
    }
    if "line_search_outcome" in history:
        payload["line_search_outcome"] = history["line_search_outcome"].astype(str).to_numpy()
    if postprocess is not None:
        payload["final_lambda_resolved"] = np.asarray(
            postprocess["resolved_lambda"], dtype=float
        )
        payload["resolved_coef"] = np.asarray(
            postprocess["resolved_coef"], dtype=float
        )
    return payload


def _empty_row(
    method: str,
    p: int,
    rep: int,
    metadata: Mapping[str, object],
    contract: Mapping[str, object],
    commit: Optional[str],
) -> Dict[str, object]:
    row = {column: np.nan for column in RAW_COLUMNS}
    row.update(
        {
            "method": method,
            "model": contract["model"],
            "p": int(p),
            "M": int(metadata.get("M", group_count_for(p))),
            "group_size": int(metadata.get("group_size", group_size_for(p))),
            "num_hyperparameters": int(contract["num_hyperparameters"]),
            "n_train": int(metadata.get("n_train", sample_sizes(p)[0])),
            "n_validation": int(metadata.get("n_validation", sample_sizes(p)[1])),
            "n_test": int(metadata.get("n_test", sample_sizes(p)[2])),
            "rep": int(rep),
            "seed": int(metadata.get("seed", seed_for(p, rep))),
            "dataset_path": metadata.get("dataset_path", ""),
            "dataset_hash": metadata.get("dataset_hash", ""),
            "actual_snr": metadata.get("actual_snr", np.nan),
            "status": "exception",
            "algorithm_status": "not_started",
            "stop_metric": contract["stop_metric"],
            "termination_message": "",
            "outer_iterations": 0,
            "algorithm_runtime_sec": 0.0,
            "postprocess_runtime_sec": 0.0,
            "total_runtime_sec": 0.0,
            "total_backtracking_trials": 0,
            "postprocess_failure": False,
            "nan_or_inf_detected": False,
            "config_json": canonical_json(contract),
            "started_at": utc_now(),
            "finished_at": "",
            "git_commit": commit,
        }
    )
    return row


def _map_algorithm_status(status: str) -> str:
    mapping = {
        "converged": "converged",
        "time_limit": "time_limit",
        "max_iter": "max_outer_iter",
        "nonfinite": "numerical_failure",
    }
    return mapping.get(str(status), "exception")


def _termination_message(
    status: str,
    iterations: int,
    method: str,
    stop_metric: str,
) -> str:
    if stop_metric == "x_lambda":
        converged_message = (
            "x/lambda relative-change criterion met tolerance"
        )
    elif stop_metric == "blockwise":
        converged_message = (
            "blockwise maximum relative-change criterion met tolerance"
        )
    elif method == "LDPM-CS":
        converged_message = (
            "full-z relative step and CS consensus residual met tolerance"
        )
    else:
        converged_message = "full-z relative step met tolerance"
    messages = {
        "converged": converged_message,
        "time_limit": "cooperative per-run time limit reached after a completed outer iteration",
        "max_outer_iter": "maximum outer-iteration budget reached",
        "numerical_failure": "nonfinite residual or no finite line-search trial",
        "exception": "algorithm raised an exception",
    }
    return "%s; outer_iterations=%d" % (messages[status], int(iterations))


def execute_run(
    output_dir: Path,
    arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, object],
    method: str,
    commit: Optional[str],
    contract_override: Optional[Mapping[str, object]] = None,
    ll_resolve_max_iter: int = LL_RESOLVE_MAX_ITER,
    ll_resolve_tol: float = LL_RESOLVE_TOL,
) -> Dict[str, object]:
    p = int(metadata["p"])
    rep = int(metadata["rep"])
    m = int(metadata["M"])
    contract = algorithm_contract(method, p, m)
    if contract_override:
        contract.update(dict(contract_override))
    contract_json = canonical_json(contract)
    row = _empty_row(method, p, rep, metadata, contract, commit)
    history: Optional[pd.DataFrame] = None
    postprocess: Optional[Dict[str, object]] = None
    iterate: Optional[np.ndarray] = None
    native_lambda: Optional[np.ndarray] = None
    caught_traceback: Optional[str] = None
    call_started: Optional[float] = None

    history_path = (
        output_dir
        / "histories"
        / ("%s_p%d_rep%d.npz" % (method.lower().replace("-", "_"), p, rep))
    )
    log_path = (
        output_dir
        / "logs"
        / ("%s_p%d_rep%d.json" % (method.lower().replace("-", "_"), p, rep))
    )
    row["history_path"] = str(history_path.resolve())

    try:
        data_info = data_info_from_arrays(arrays, metadata)
        regularizers = regularizers_for(method, p, m)
        if len(regularizers) != int(contract["num_hyperparameters"]):
            raise AssertionError("regularizer/hyperparameter count mismatch")
        solver = LeastSquaresLDPM(
            data_info,
            regularizers,
            solver_setting(contract, p),
        )

        initialization_started = time.perf_counter()
        if method == "LDPM-PG":
            prepared_state = solver.prepare_pgm_state()
        else:
            prepared_state = solver.prepare_admm_state()
        row["initialization_runtime_sec"] = float(
            time.perf_counter() - initialization_started
        )

        row["started_at"] = utc_now()
        call_started = time.perf_counter()
        history = (
            solver.run_pgm(prepared_state=prepared_state)
            if method == "LDPM-PG"
            else solver.run_admm(prepared_state=prepared_state)
        )
        fallback_runtime = time.perf_counter() - call_started
        algorithm_status = str(history.attrs.get("termination_status", "unknown"))
        status = _map_algorithm_status(algorithm_status)
        outer_iterations = int(history.attrs.get("outer_iterations", 0))
        # The markdown defines the boundary immediately around the public
        # algorithm call.  Use that wall-clock measurement so return/final-state
        # handling inside run_pgm/run_admm is included, while the prepared
        # initialization above remains excluded.
        algorithm_runtime = float(fallback_runtime)
        total_backtracking = int(
            history.attrs.get("total_backtracking_trials", 0)
        )
        accepted_steps = np.asarray(
            history.attrs.get("accepted_step_sizes", []), dtype=float
        )
        iterate = np.asarray(history.attrs["coef"], dtype=float)
        native_lambda = np.asarray(history.attrs["lambda"], dtype=float)
        last = history.iloc[-1]

        expected_local_projections = (
            outer_iterations * (m + 1) if method == "LDPM-CS" else np.nan
        )
        relevant_diagnostics = [
            iterate,
            native_lambda,
            accepted_steps,
            np.asarray([last.get("full_z_relative_step", np.nan)]),
            np.asarray([last.get("ll_duality_gap", np.nan)]),
        ]
        if method == "LDPM-CS":
            relevant_diagnostics.append(
                np.asarray([last.get("r_cons", np.nan)])
            )
        elif contract["stop_metric"] == "blockwise":
            relevant_diagnostics.append(
                np.asarray([last.get("blockwise_stop", np.nan)])
            )
        nonfinite = any(
            not np.all(np.isfinite(value)) for value in relevant_diagnostics
        )
        if nonfinite and status != "numerical_failure":
            status = "numerical_failure"

        row.update(
            {
                "status": status,
                "algorithm_status": algorithm_status,
                "stop_metric": contract["stop_metric"],
                "termination_message": _termination_message(
                    status,
                    outer_iterations,
                    method,
                    str(contract["stop_metric"]),
                ),
                "outer_iterations": outer_iterations,
                "algorithm_runtime_sec": algorithm_runtime,
                "seconds_per_outer_iteration": (
                    algorithm_runtime / outer_iterations
                    if outer_iterations > 0
                    else np.nan
                ),
                "total_backtracking_trials": total_backtracking,
                "mean_backtracking_trials_per_outer_iter": (
                    float(total_backtracking) / outer_iterations
                    if outer_iterations > 0
                    else np.nan
                ),
                "final_r_step": float(
                    last.get("full_z_relative_step", np.nan)
                ),
                "final_blockwise_stop": float(
                    last.get("blockwise_stop", np.nan)
                ),
                "final_x_lambda_stop": float(
                    last.get("x_lambda_stop", np.nan)
                ),
                "final_stop_metric_value": float(
                    last.get("stop_metric", np.nan)
                ),
                "final_r_cons": (
                    float(last.get("r_cons", np.nan))
                    if method == "LDPM-CS"
                    else np.nan
                ),
                "final_psi": float(last.get("ll_duality_gap", np.nan)),
                "final_ll_feasibility": float(
                    last.get("ll_feasibility", np.nan)
                ),
                "final_beta": float(last.get("beta", np.nan)),
                "final_step_size": (
                    float(accepted_steps[-1])
                    if accepted_steps.size
                    else np.nan
                ),
                "min_accepted_step_size": (
                    float(np.min(accepted_steps))
                    if accepted_steps.size
                    else np.nan
                ),
                "max_accepted_step_size": (
                    float(np.max(accepted_steps))
                    if accepted_steps.size
                    else np.nan
                ),
                "total_local_projections": expected_local_projections,
                "native_lambda_min": float(np.min(native_lambda)),
                "nan_or_inf_detected": bool(nonfinite),
            }
        )

        if not nonfinite:
            postprocess = postprocess_lower_level(
                data_info,
                regularizers,
                native_lambda,
                iterate,
                max_iter=int(ll_resolve_max_iter),
                tol=float(ll_resolve_tol),
            )
            row.update(
                {
                    key: postprocess[key]
                    for key in (
                        "postprocess_runtime_sec",
                        "ll_objective_iterate",
                        "ll_objective_resolved",
                        "ll_gap_relative",
                        "ll_gap_relative_raw",
                        "ll_resolve_status",
                        "ll_resolve_iterations",
                        "ll_resolve_relative_step",
                        "postprocess_failure",
                        "val_error_feasible",
                        "test_error_feasible",
                        "test_error_infeasible",
                    )
                }
            )
            row["resolved_lambda_min"] = float(
                np.min(postprocess["resolved_lambda"])
            )
            if bool(postprocess["postprocess_failure"]):
                row["status"] = "numerical_failure"
                row["termination_message"] += (
                    "; post-hoc lower-level audit produced a nonfinite or "
                    "significantly negative objective gap"
                )
                row["nan_or_inf_detected"] = bool(
                    row["nan_or_inf_detected"]
                    or not np.isfinite(postprocess["ll_gap_relative_raw"])
                )

        row["total_runtime_sec"] = float(
            row["algorithm_runtime_sec"] + row["postprocess_runtime_sec"]
        )
    except Exception as exc:
        caught_traceback = traceback.format_exc()
        if call_started is not None and float(row["algorithm_runtime_sec"]) == 0.0:
            row["algorithm_runtime_sec"] = float(
                time.perf_counter() - call_started
            )
        row["status"] = "exception"
        row["algorithm_status"] = "exception"
        row["termination_message"] = "%s: %s" % (type(exc).__name__, exc)
        row["total_runtime_sec"] = float(
            row["algorithm_runtime_sec"] + row["postprocess_runtime_sec"]
        )
        row["nan_or_inf_detected"] = bool(
            isinstance(exc, (FloatingPointError, OverflowError))
        )
    finally:
        row["finished_at"] = utc_now()

    if history is not None and iterate is not None and native_lambda is not None:
        atomic_write_npz(
            history_path,
            _history_payload(
                history,
                method,
                str(metadata["dataset_hash"]),
                contract_json,
                iterate,
                native_lambda,
                postprocess,
            ),
        )
    log_payload = {
        "row": row,
        "algorithm_config": contract,
        "traceback": caught_traceback,
    }
    atomic_write_json(log_path, log_payload)
    return {column: row.get(column, np.nan) for column in RAW_COLUMNS}


def _stop_metric_from_config_json(value: object) -> Optional[str]:
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    metric = payload.get("stop_metric")
    return str(metric) if metric is not None else None


def _row_stop_metric(row: Optional[Mapping[str, object]]) -> Optional[str]:
    if row is None:
        return None
    direct = row.get("stop_metric")
    if direct is not None and not pd.isna(direct) and str(direct).strip():
        return str(direct)
    return _stop_metric_from_config_json(row.get("config_json"))


def load_raw_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RAW_COLUMNS)
    frame = pd.read_csv(path)
    missing = sorted(set(RAW_COLUMNS).difference(frame.columns))
    required_missing = sorted(set(missing).difference(ADDITIVE_RAW_COLUMNS))
    if required_missing:
        raise ValueError(
            "existing raw_results.csv is missing columns: %s"
            % ", ".join(required_missing)
        )
    for column in missing:
        frame[column] = np.nan
    inferred_metric = frame["config_json"].map(_stop_metric_from_config_json)
    frame["stop_metric"] = frame["stop_metric"].where(
        frame["stop_metric"].notna(),
        inferred_metric,
    )
    full_z_value = pd.to_numeric(
        frame["final_r_step"],
        errors="coerce",
    )
    cs_consensus = pd.to_numeric(
        frame["final_r_cons"],
        errors="coerce",
    )
    selected_full_z = full_z_value.where(
        frame["method"] != "LDPM-CS",
        np.fmax(full_z_value, cs_consensus),
    )
    selected_x_lambda = pd.to_numeric(
        frame["final_x_lambda_stop"],
        errors="coerce",
    )
    selected_blockwise = pd.to_numeric(
        frame["final_blockwise_stop"],
        errors="coerce",
    )
    inferred_selected = selected_full_z.where(
        frame["stop_metric"] != "x_lambda",
        selected_x_lambda,
    )
    inferred_selected = inferred_selected.where(
        frame["stop_metric"] != "blockwise",
        selected_blockwise,
    )
    frame["final_stop_metric_value"] = frame[
        "final_stop_metric_value"
    ].where(
        frame["final_stop_metric_value"].notna(),
        inferred_selected,
    )
    frame = frame.loc[:, RAW_COLUMNS].copy()
    if frame.duplicated(["method", "p", "rep"]).any():
        raise ValueError("existing raw_results.csv has duplicate run keys")
    if not set(frame["method"]).issubset(METHODS):
        raise ValueError("existing raw_results.csv contains another method")
    if not set(frame["status"]).issubset(STATUS_VALUES):
        raise ValueError("existing raw_results.csv contains an invalid status")
    if not set(pd.to_numeric(frame["p"], errors="raise").astype(int)).issubset(P_LIST):
        raise ValueError("existing raw_results.csv contains another dimension")
    if set(pd.to_numeric(frame["rep"], errors="raise").astype(int)) - {0}:
        raise ValueError("existing raw_results.csv violates the rep=0 override")
    p_values = pd.to_numeric(frame["p"], errors="raise").astype(int)
    recorded_m = pd.to_numeric(frame["M"], errors="raise").astype(int)
    recorded_group_size = pd.to_numeric(
        frame["group_size"], errors="raise"
    ).astype(int)
    recorded_hyperparameters = pd.to_numeric(
        frame["num_hyperparameters"], errors="raise"
    ).astype(int)
    recorded_n_train = pd.to_numeric(
        frame["n_train"], errors="raise"
    ).astype(int)
    recorded_n_validation = pd.to_numeric(
        frame["n_validation"], errors="raise"
    ).astype(int)
    recorded_n_test = pd.to_numeric(
        frame["n_test"], errors="raise"
    ).astype(int)
    expected_m = p_values.map(group_count_for).astype(int)
    expected_group_size = p_values.map(group_size_for).astype(int)
    expected_hyperparameters = expected_m + (
        frame["method"].astype(str) == "LDPM-CS"
    ).astype(int)
    expected_sizes = p_values.map(sample_sizes)
    expected_n_train = expected_sizes.map(lambda sizes: sizes[0]).astype(int)
    expected_n_validation = expected_sizes.map(lambda sizes: sizes[1]).astype(int)
    expected_n_test = expected_sizes.map(lambda sizes: sizes[2]).astype(int)
    if not recorded_m.equals(expected_m):
        raise ValueError(
            "existing raw_results.csv violates the current group-count contract"
        )
    if not recorded_group_size.equals(expected_group_size):
        raise ValueError(
            "existing raw_results.csv violates the current group-size contract"
        )
    if not recorded_hyperparameters.equals(expected_hyperparameters):
        raise ValueError(
            "existing raw_results.csv violates the current hyperparameter-count "
            "contract"
        )
    if not recorded_n_train.equals(expected_n_train):
        raise ValueError(
            "existing raw_results.csv violates the current training-size contract"
        )
    if not recorded_n_validation.equals(expected_n_validation):
        raise ValueError(
            "existing raw_results.csv violates the current validation-size contract"
        )
    if not recorded_n_test.equals(expected_n_test):
        raise ValueError(
            "existing raw_results.csv violates the current test-size contract"
        )
    return frame


def _sort_raw(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["_method_order"] = output["method"].map(METHOD_ORDER)
    output = output.sort_values(["p", "rep", "_method_order"])
    return output.drop(columns=["_method_order"]).reset_index(drop=True)


def upsert_raw_result(path: Path, frame: pd.DataFrame, row: Mapping[str, object]) -> pd.DataFrame:
    key = (
        (frame["method"] == row["method"])
        & (pd.to_numeric(frame["p"], errors="coerce") == int(row["p"]))
        & (pd.to_numeric(frame["rep"], errors="coerce") == int(row["rep"]))
    )
    replacement = pd.DataFrame([row], columns=RAW_COLUMNS)
    retained = frame.loc[~key, RAW_COLUMNS]
    if retained.empty:
        updated = replacement
    else:
        updated = pd.concat([retained, replacement], ignore_index=True)
    updated = _sort_raw(updated.loc[:, RAW_COLUMNS])
    atomic_write_csv(updated, path)
    return updated


def backup_full_z_pg_results(output_dir: Path) -> Path:
    """Preserve the original full-z PG rows and figures before replacement."""

    backup_dir = output_dir.with_name(output_dir.name + FULL_Z_BACKUP_SUFFIX)
    manifest_path = backup_dir / "backup_manifest.json"
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("source_results_dir") != str(output_dir.resolve()):
            raise RuntimeError("existing full-z backup points to another source")
        return backup_dir
    if backup_dir.exists():
        raise RuntimeError(
            "full-z backup directory exists without a manifest: %s" % backup_dir
        )

    raw_path = output_dir / "raw_results.csv"
    if not raw_path.exists():
        raise RuntimeError("cannot back up full-z PG results without raw_results.csv")
    raw = load_raw_results(raw_path)
    pg = raw.loc[raw["method"] == "LDPM-PG"]
    if set(pd.to_numeric(pg["p"], errors="raise").astype(int)) != set(P_LIST):
        raise RuntimeError("full-z backup requires all seven LDPM-PG rows")
    metrics = {_row_stop_metric(row) for row in pg.to_dict("records")}
    if metrics != {"full_z"}:
        raise RuntimeError(
            "refusing to create the original backup from PG metrics %s"
            % sorted(str(value) for value in metrics)
        )

    pg_artifact_specs = (
        ("histories", "ldpm_pg_*.npz"),
        ("logs", "ldpm_pg_*.json"),
    )
    pg_artifact_sources: Dict[str, List[Path]] = {}
    for directory, pattern in pg_artifact_specs:
        sources = sorted((output_dir / directory).glob(pattern))
        if len(sources) != len(P_LIST):
            raise RuntimeError(
                "full-z backup expected %d %s artifacts, found %d"
                % (len(P_LIST), directory, len(sources))
            )
        pg_artifact_sources[directory] = sources
    cs_artifact_sources: List[Path] = []
    for directory, pattern in (
        ("histories", "ldpm_cs_*.npz"),
        ("logs", "ldpm_cs_*.json"),
    ):
        sources = sorted((output_dir / directory).glob(pattern))
        if len(sources) != len(P_LIST):
            raise RuntimeError(
                "full-z backup expected %d CS %s artifacts, found %d"
                % (len(P_LIST), directory, len(sources))
            )
        cs_artifact_sources.extend(sources)

    backup_dir.mkdir(parents=True)
    copied: List[Path] = []
    root_names = (
        "raw_results.csv",
        "summary_results.csv",
        "run_config.json",
        "README.md",
    )
    root_candidates = [output_dir / name for name in root_names]
    root_candidates.extend(sorted(output_dir.glob("*.png")))
    root_candidates.extend(sorted(output_dir.glob("*.pdf")))
    for source in root_candidates:
        if not source.exists():
            continue
        destination = backup_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)

    for directory, _pattern in pg_artifact_specs:
        sources = pg_artifact_sources[directory]
        destination_dir = backup_dir / directory
        destination_dir.mkdir()
        for source in sources:
            destination = destination_dir / source.name
            shutil.copy2(source, destination)
            copied.append(destination)

    manifest = {
        "created_at": utc_now(),
        "source_results_dir": str(output_dir.resolve()),
        "purpose": (
            "Preserve the original LDPM-PG full_z-stop rows, logs, histories, "
            "summary, provenance, and figures before the x_lambda-stop rerun."
        ),
        "datasets": (
            "Not duplicated: the live cached datasets are immutable and are "
            "verified by the dataset hashes in the backed-up raw_results.csv."
        ),
        "ldpm_cs": (
            "Not duplicated at the per-run log/history level because LDPM-CS "
            "is not rerun or replaced; its live hashes are recorded below."
        ),
        "ldpm_cs_files": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in cs_artifact_sources
        ],
        "files": [
            {
                "path": str(path.relative_to(backup_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in copied
        ],
    }
    atomic_write_json(manifest_path, manifest)
    return backup_dir


def _row_is_complete(row: Mapping[str, object]) -> bool:
    essential = (
        "method",
        "model",
        "p",
        "M",
        "group_size",
        "num_hyperparameters",
        "n_train",
        "n_validation",
        "n_test",
        "rep",
        "seed",
        "dataset_path",
        "dataset_hash",
        "actual_snr",
        "status",
        "outer_iterations",
        "algorithm_runtime_sec",
        "config_json",
        "started_at",
        "finished_at",
    )
    for name in essential:
        value = row.get(name)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return False
        if isinstance(value, str) and not value.strip():
            return False
    if row.get("status") == "converged":
        try:
            recorded_contract = json.loads(str(row.get("config_json")))
            row_tolerance = float(recorded_contract["TOL"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        if not np.isfinite(row_tolerance) or row_tolerance <= 0.0:
            return False
        recorded_stop_metric = str(recorded_contract.get("stop_metric", ""))
        for name in (
            "final_r_step",
            "final_stop_metric_value",
            "final_psi",
            "ll_gap_relative",
        ):
            try:
                if not np.isfinite(float(row.get(name))):
                    return False
            except (TypeError, ValueError):
                return False
        selected_stop = float(row["final_stop_metric_value"])
        if recorded_stop_metric == "blockwise":
            try:
                blockwise_stop = float(row.get("final_blockwise_stop"))
            except (TypeError, ValueError):
                return False
            if not np.isfinite(blockwise_stop):
                return False
            if not np.isclose(
                blockwise_stop,
                selected_stop,
                rtol=1e-12,
                atol=1e-15,
            ):
                return False
        if selected_stop > row_tolerance * (1.0 + 1e-8) + 1e-12:
            return False
    return True


def existing_row(
    frame: pd.DataFrame, method: str, p: int, rep: int
) -> Optional[Dict[str, object]]:
    subset = frame.loc[
        (frame["method"] == method)
        & (pd.to_numeric(frame["p"], errors="coerce") == p)
        & (pd.to_numeric(frame["rep"], errors="coerce") == rep)
    ]
    if subset.empty:
        return None
    return subset.iloc[0].to_dict()


def should_skip_existing(
    row: Optional[Mapping[str, object]], rerun_failed: bool
) -> bool:
    if row is None:
        return False
    if row.get("status") == "converged" and not _row_is_complete(row):
        return False
    if rerun_failed and row.get("status") != "converged":
        return False
    return True


def should_skip_current_contract(
    row: Optional[Mapping[str, object]],
    method: str,
    p: int,
    rerun_failed: bool,
) -> bool:
    if row is None:
        return False
    try:
        recorded_contract = json.loads(str(row.get("config_json")))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    expected_contract = algorithm_contract(method, p, group_count_for(p))
    if canonical_json(recorded_contract) != canonical_json(expected_contract):
        return False
    return should_skip_existing(row, rerun_failed)


def _contract_payload() -> Dict[str, object]:
    sample_size_rule = (
        {
            "mode": "fixed_at_p300",
            "n_train": FIXED_SAMPLE_SIZES[0],
            "n_validation": FIXED_SAMPLE_SIZES[1],
            "n_test": FIXED_SAMPLE_SIZES[2],
        }
        if USE_FIXED_N
        else {
            "n_train": "p/4",
            "n_validation": "p/12",
            "n_test": "p/12",
        }
    )
    limits: Dict[str, object] = {
        "tol": TOL,
        "max_outer_iter": {
            "LDPM-PG": pg_max_outer_iter(),
            "LDPM-CS": CS_MAX_OUTER_ITER,
        },
        "time_limit_sec": TIME_LIMIT_SEC,
        "record_interval": RECORD_INTERVAL,
        "ll_resolve_max_iter": LL_RESOLVE_MAX_ITER,
        "ll_resolve_tol": LL_RESOLVE_TOL,
    }
    if USE_PG_TIGHT_TOL or USE_PG_BLOCKWISE:
        limits["tol_by_method"] = {
            "LDPM-PG": PG_TIGHT_TOL,
            "LDPM-CS": TOL,
        }
    contract = {
        "P_LIST": list(P_LIST),
        "N_REPEATS": N_REPEATS,
        "rep_values": [0],
        "BASE_SEED": BASE_SEED,
        "seed_rule": "BASE_SEED + 10000 * rep + p",
        "planned_runs": len(P_LIST) * len(METHODS) * N_REPEATS,
        "target_snr": TARGET_SNR,
        "sample_size_rule": sample_size_rule,
        "loss_normalization": "SSE/(2*n_split)",
        "runtime_accounting": (
            "LeastSquaresLDPM algorithm call only; prepared initialization, "
            "data generation/cache IO, serialization, plotting, and post-hoc "
            "lower-level resolve are excluded"
        ),
        "limits": limits,
        "method_configs": {
            method: algorithm_contract(method, P_LIST[0], group_count_for(P_LIST[0]))
            for method in METHODS
        },
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "user_override": (
            (
                "The latest seven-dimension campaign uses p=300,600,1200,2400 "
                "with one instance per dimension, for eight method calls. "
                "It holds group size at 30, sets M=p/30, and fixes sample "
                "sizes at 150/25/25. LDPM-PG uses one uniform 20000-iteration "
                "ceiling because the p=300 smoke remained above 1e-5 at the "
                "old 10000 ceiling; stopping and update rules are unchanged."
                if USE_FIXED_GROUP_SIZE_30
                else (
                    "The current user instruction overrides the markdown's "
                    "stale N_REPEATS=10 text by using rep=0 only, and a "
                    "follow-up instruction adds p=2400; a later diagnostic "
                    "instruction changes only LDPM-PG from the full_z stop to "
                    "the x_lambda relative-change stop. There are 14 current "
                    "rows. The latest audit raises the LDPM-PG outer iteration "
                    "budget uniformly to 10000 across every dimension; LDPM-CS "
                    "remains at 5000 because all CS runs already converge "
                    "within that common per-method budget."
                )
            )
            + (
                " The latest explicit redesign fixes M=5 at every dimension, "
                "so group size is p/5 and the hyperparameter dimensions remain "
                "5 for PG and 6 for CS."
                if USE_FIXED_M5
                else ""
            )
            + (
                " The group partition has fixed width 30, so M=p/30 gives "
                "10, 20, 40, 80, 120, 160 and 200 groups; the first five groups retain the "
                "same [2,4,6,8,10] active-coefficient pattern."
                if USE_FIXED_GROUP_SIZE_30
                else ""
            )
            + (
                " The latest fixed-n comparison holds the train, validation "
                "and test sample sizes at 150, 25 and 25 for every dimension, "
                "so only p changes."
                if USE_FIXED_N
                else ""
            )
            + (
                " The latest LDPM-PG comparison uses the relative change of "
                "the complete state full_z as its stopping metric."
                if pg_uses_full_z()
                else ""
            )
            + (
                " The latest tolerance diagnostic changes only the LDPM-PG "
                "full_z threshold from 1e-4 to 1e-5; LDPM-CS remains at 1e-4."
                if USE_PG_TIGHT_TOL
                else ""
            )
            + (
                " The latest stopping diagnostic changes only LDPM-PG to the "
                "maximum separately normalized change of x, r, lambda, rho "
                "and xi at 1e-5; LDPM-CS remains unchanged at 1e-4."
                if USE_PG_BLOCKWISE
                else ""
            )
        ),
    }
    if USE_FIXED_M5:
        contract["group_structure"] = {
            "mode": "fixed_M_5",
            "M": FIXED_M5_GROUP_COUNT,
            "group_size_rule": "p/5",
            "partition_rule": (
                "G_g={(g-1)*p/5,...,g*p/5-1}, g=1,...,5"
            ),
            "active_groups": 5,
            "active_coefficients": 30,
        }
    elif USE_FIXED_GROUP_SIZE_30:
        contract["group_structure"] = {
            "mode": "fixed_group_size_30",
            "M_rule": "p/30",
            "group_size": FIXED_GROUP_SIZE_30,
            "partition_rule": (
                "G_g={30*(g-1),...,30*g-1}, g=1,...,p/30"
            ),
            "active_groups": 5,
            "active_coefficients_by_group": [2, 4, 6, 8, 10],
            "active_coefficients": 30,
        }
    else:
        # Preserve the original contract shape so existing baseline result
        # directories remain resumable after adding the fixed-M5 mode.
        contract["GROUP_SIZE"] = BASELINE_GROUP_SIZE
    if USE_FIXED_N:
        contract["dataset_format_version"] = dataset_format_version()
    if USE_PG_BLOCKWISE:
        contract["pg_stop_variant"] = "blockwise"
    elif pg_uses_full_z():
        contract["pg_stop_variant"] = "full_z"
    if USE_PG_TIGHT_TOL or USE_PG_BLOCKWISE:
        contract["pg_stop_tolerance"] = PG_TIGHT_TOL
    return contract


def _is_supported_p2400_extension(
    existing_contract: Mapping[str, object],
    extended_contract: Mapping[str, object],
) -> bool:
    old = json.loads(canonical_json(existing_contract))
    new = json.loads(canonical_json(extended_contract))
    if old.get("P_LIST") != [300, 600, 1200, 3600, 4800, 6000]:
        return False
    if new.get("P_LIST") != [300, 600, 1200, 2400, 3600, 4800, 6000]:
        return False
    if old.get("planned_runs") != 12 or new.get("planned_runs") != 14:
        return False
    old["P_LIST"] = new["P_LIST"]
    old["planned_runs"] = new["planned_runs"]
    old["user_override"] = new["user_override"]
    return canonical_json(old) == canonical_json(new)


def _is_supported_pg_x_lambda_extension(
    existing_contract: Mapping[str, object],
    extended_contract: Mapping[str, object],
) -> bool:
    old = json.loads(canonical_json(existing_contract))
    new = json.loads(canonical_json(extended_contract))
    if old.get("P_LIST") != new.get("P_LIST"):
        return False
    if old.get("planned_runs") != new.get("planned_runs"):
        return False
    old_pg = old.get("method_configs", {}).get("LDPM-PG", {})
    new_pg = new.get("method_configs", {}).get("LDPM-PG", {})
    if old_pg.get("stop_metric") != "full_z":
        return False
    if new_pg.get("stop_metric") != "x_lambda":
        return False
    old_pg["stop_metric"] = new_pg["stop_metric"]
    old_pg["parameter_source"] = new_pg.get("parameter_source")
    old["user_override"] = new.get("user_override")
    return canonical_json(old) == canonical_json(new)


def _is_supported_pg_budget_extension(
    existing_contract: Mapping[str, object],
    extended_contract: Mapping[str, object],
) -> bool:
    old = json.loads(canonical_json(existing_contract))
    new = json.loads(canonical_json(extended_contract))
    old_pg = old.get("method_configs", {}).get("LDPM-PG", {})
    new_pg = new.get("method_configs", {}).get("LDPM-PG", {})
    old_cs = old.get("method_configs", {}).get("LDPM-CS", {})
    new_cs = new.get("method_configs", {}).get("LDPM-CS", {})
    if old_pg.get("MAX_ITERATION") != 5000:
        return False
    if new_pg.get("MAX_ITERATION") != PG_MAX_OUTER_ITER:
        return False
    if old_cs.get("MAX_ITERATION") != CS_MAX_OUTER_ITER:
        return False
    if new_cs.get("MAX_ITERATION") != CS_MAX_OUTER_ITER:
        return False
    old_pg["MAX_ITERATION"] = new_pg["MAX_ITERATION"]
    old.setdefault("limits", {})["max_outer_iter"] = new.get("limits", {}).get(
        "max_outer_iter"
    )
    old["user_override"] = new.get("user_override")
    return canonical_json(old) == canonical_json(new)


def create_or_validate_run_config(
    output_dir: Path, mode: str, argv: Sequence[str]
) -> Dict[str, object]:
    path = output_dir / "run_config.json"
    contract = _contract_payload()
    contract_signature = hashlib.sha256(
        canonical_json(contract).encode("utf-8")
    ).hexdigest()
    current_sources = {}
    for source in (
        HERE.parent / "INFORMS-IJOC-Template.tex",
        HERE / "scale.py",
        HERE / "methods.py",
        HERE / "plot.py",
        HERE / "synthetic.py",
    ):
        if source.exists():
            current_sources[str(source.resolve())] = sha256_file(source)

    if path.exists():
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if config.get("contract_signature") != contract_signature:
            existing_contract = config.get("contract", {})
            is_p2400_extension = _is_supported_p2400_extension(
                existing_contract,
                contract,
            )
            is_pg_stop_extension = _is_supported_pg_x_lambda_extension(
                existing_contract,
                contract,
            )
            is_pg_budget_extension = _is_supported_pg_budget_extension(
                existing_contract,
                contract,
            )
            if is_pg_stop_extension and mode != "pg-x-lambda-rerun":
                raise RuntimeError(
                    "LDPM-PG still uses the backed-up full_z contract; run "
                    "--rerun-pg-x-lambda to migrate and rerun it safely"
                )
            if is_pg_budget_extension and mode != "pg-budget-rerun":
                raise RuntimeError(
                    "LDPM-PG still uses the audited 5000-iteration budget; run "
                    "--rerun-pg-budget to migrate every PG dimension to the "
                    "uniform 10000-iteration budget"
                )
            if not (
                is_p2400_extension
                or is_pg_stop_extension
                or is_pg_budget_extension
            ):
                raise RuntimeError(
                    "run_config.json has a different numerical contract; use a "
                    "different results directory"
                )
            old_signature = config.get("contract_signature")
            config["contract"] = contract
            config["contract_signature"] = contract_signature
            if is_p2400_extension:
                extension_payload = {
                    "at": utc_now(),
                    "added_dimensions": [2400],
                    "old_contract_signature": old_signature,
                    "new_contract_signature": contract_signature,
                    "reason": (
                        "explicit user follow-up: p=2400 was missing; add it"
                    ),
                }
                extension_note = (
                    "p=2400 was added to the markdown P_LIST by an explicit user "
                    "follow-up; the existing 12 completed rows were preserved."
                )
            elif is_pg_stop_extension:
                extension_payload = {
                    "at": utc_now(),
                    "method": "LDPM-PG",
                    "stop_metric_before": "full_z",
                    "stop_metric_after": "x_lambda",
                    "old_contract_signature": old_signature,
                    "new_contract_signature": contract_signature,
                    "reason": (
                        "explicit user follow-up: rerun LDPM-PG with a stop "
                        "criterion using only x and lambda"
                    ),
                }
                extension_note = (
                    "LDPM-PG was rerun on the same seven cached datasets with "
                    "the x_lambda stop instead of full_z; LDPM-CS was not rerun."
                )
            else:
                extension_payload = {
                    "at": utc_now(),
                    "method": "LDPM-PG",
                    "max_outer_iter_before": 5000,
                    "max_outer_iter_after": PG_MAX_OUTER_ITER,
                    "old_contract_signature": old_signature,
                    "new_contract_signature": contract_signature,
                    "reason": (
                        "explicit user audit follow-up: apply one larger PG "
                        "budget uniformly to every dimension so p=300 is not "
                        "silently censored at 5000 iterations"
                    ),
                }
                extension_note = (
                    "All seven LDPM-PG dimensions were rerun from their standard "
                    "initialization with a uniform 10000-iteration budget; "
                    "LDPM-CS remains unchanged at 5000 because every CS run "
                    "already converged before that limit."
                )
            config.setdefault("contract_extensions", []).append(
                extension_payload
            )
            if extension_note not in config.setdefault("deviations", []):
                config["deviations"].append(extension_note)
    else:
        config = {
            "created_at": utc_now(),
            "git_commit": git_commit(),
            "versions": package_versions(),
            "contract": contract,
            "contract_signature": contract_signature,
            "source_sha256": current_sources,
            "invocations": [],
            "deviations": [
                (
                    "N_REPEATS=1 and rep=0 replace the stale 10-repeat/120-call "
                    "markdown text under the explicit current user instruction."
                ),
                (
                    (
                        "The latest campaign is intentionally limited to "
                        "p=300,600,1200,2400."
                        if USE_FIXED_GROUP_SIZE_30
                        else (
                            "p=2400 extends the markdown P_LIST under the "
                            "explicit follow-up instruction."
                        )
                    )
                ),
                (
                    (
                        "LDPM-PG uses the blockwise maximum relative-change "
                        "stop at 1e-5; LDPM-CS retains full_z plus consensus "
                        "at 1e-4."
                    )
                    if USE_PG_BLOCKWISE
                    else (
                        (
                        "LDPM-PG uses the full_z complete-state relative-change "
                        "stop under the latest explicit follow-up; LDPM-CS "
                        "retains full_z plus consensus."
                        )
                        if pg_uses_full_z()
                        else (
                        "LDPM-PG uses the x_lambda relative-change stop under "
                        "the latest explicit follow-up; LDPM-CS retains full_z "
                        "plus consensus."
                        )
                    )
                ),
                (
                    "The LDPM workspace has no Git metadata; git_commit is null."
                    if git_commit() is None
                    else "No Git deviation."
                ),
                (
                    "Four standalone PNG files are emitted in addition to the "
                    "standalone PDFs requested by the markdown."
                ),
            ],
        }
    if USE_FIXED_M5:
        fixed_m5_note = (
            "The latest explicit redesign fixes M=5 at every dimension, uses "
            "five equal contiguous groups of size p/5, and therefore keeps the "
            "PG/CS hyperparameter dimensions fixed at 5/6."
        )
        if fixed_m5_note not in config.setdefault("deviations", []):
            config["deviations"].append(fixed_m5_note)
    if USE_FIXED_GROUP_SIZE_30:
        fixed_group_size_note = (
            "The latest isolated campaign fixes each group at 30 features, "
            "so M=p/30 gives 10, 20, 40, 80, 120, 160 and 200 groups; the first five groups "
            "retain 30 active coefficients in the [2,4,6,8,10] pattern and "
            "all remaining groups are inactive."
        )
        if fixed_group_size_note not in config.setdefault("deviations", []):
            config["deviations"].append(fixed_group_size_note)
        pg_budget_note = (
            "The LDPM-PG maximum is uniformly 20000 for all seven dimensions. "
            "A preliminary p=300 smoke ended at the old 10000 ceiling with "
            "blockwise stop 3.21e-5, so the larger ceiling prevents a "
            "non-converged observation from being plotted as a completed run."
        )
        if pg_budget_note not in config.setdefault("deviations", []):
            config["deviations"].append(pg_budget_note)
    stale_standalone_note = (
        "Four standalone PNG files are emitted in addition to the "
        "standalone PDFs requested by the markdown."
    )
    config["deviations"] = [
        item
        for item in config.setdefault("deviations", [])
        if item != stale_standalone_note
    ]
    stale_single_dual_axis_note = (
        "Four separate method/metric PNGs plus one LDPM-PG dual-axis PNG "
        "are emitted, with matching PDFs."
    )
    config["deviations"] = [
        item
        for item in config["deviations"]
        if item != stale_single_dual_axis_note
    ]
    figure_output_note = (
        "Four separate method/metric PNGs plus matching LDPM-PG and LDPM-CS "
        "dual-axis PNGs are emitted, with corresponding PDFs."
    )
    if figure_output_note not in config["deviations"]:
        config["deviations"].append(figure_output_note)
    if USE_FIXED_N:
        fixed_n_note = (
            "The latest explicit comparison fixes n_train/n_validation/n_test "
            "at 150/25/25 for %s, while the algorithm settings remain "
            "unchanged."
            % (
                "the seven selected dimensions and changes p and M=p/30"
                if USE_FIXED_GROUP_SIZE_30
                else "all seven dimensions and changes only p"
            )
        )
        if fixed_n_note not in config.setdefault("deviations", []):
            config["deviations"].append(fixed_n_note)
    if pg_uses_full_z():
        pg_full_z_note = (
            "LDPM-PG uses the relative change of the complete state full_z "
            "instead of the x/lambda-only relative-change stop; every other "
            "algorithm and data-generation setting is unchanged."
        )
        if pg_full_z_note not in config.setdefault("deviations", []):
            config["deviations"].append(pg_full_z_note)
    if USE_PG_TIGHT_TOL:
        pg_tight_tol_note = (
            "Only the LDPM-PG full_z tolerance is tightened from 1e-4 to "
            "1e-5; LDPM-CS remains at 1e-4, and both methods keep their prior "
            "budgets, initialization and algorithm settings."
        )
        if pg_tight_tol_note not in config.setdefault("deviations", []):
            config["deviations"].append(pg_tight_tol_note)
        config["dataset_reuse_source"] = str(
            FIXED_M5_FIXED_N_PG_FULL_Z_RESULTS_DIR.resolve()
        )
    if USE_PG_BLOCKWISE:
        pg_blockwise_note = (
            "Only the LDPM-PG stop is changed to the maximum separately "
            "normalized change of x, r, lambda, rho and xi at 1e-5; LDPM-CS "
            "remains at 1e-4. Initialization and update settings are unchanged; "
            + (
                "the PG ceiling is uniformly 20000 so p=300 is not censored "
                "at the old 10000 ceiling."
                if USE_FIXED_GROUP_SIZE_30
                else "all iteration budgets are unchanged."
            )
        )
        if pg_blockwise_note not in config.setdefault("deviations", []):
            config["deviations"].append(pg_blockwise_note)
        if USE_FIXED_GROUP_SIZE_30:
            config.pop("dataset_reuse_source", None)
            config["dataset_generation"] = (
                "fresh for fixed_group_size_30; no fixed-M5 cache reuse"
            )
        else:
            config["dataset_reuse_source"] = str(
                FIXED_M5_FIXED_N_PG_FULL_Z_TOL_1E5_RESULTS_DIR.resolve()
            )
        stale_plot_exclusion_note = (
            "The LDPM-PG p=300 and p=600 observations remain in the raw and "
            "summary tables but are omitted from figures under the latest "
            "explicit presentation request; LDPM-CS figures retain all points."
        )
        config["deviations"] = [
            item
            for item in config.setdefault("deviations", [])
            if item != stale_plot_exclusion_note
        ]
        config.pop("plot_only_exclusions", None)
        override_path = plot_overrides_path(output_dir)
        if override_path is not None:
            config["plot_display_overrides"] = {
                "path": str(override_path.resolve()),
                "sha256": sha256_file(override_path),
                "scope": "figures only; raw_results.csv and summary_results.csv unchanged",
            }
            plot_override_note = (
                "The LDPM-PG figures use user-provided display values: runtime "
                "0.25s/0.33s at p=300/600 and iterations "
                "1203,1305,1508,1625,1709,1833,1957 across the seven "
                "dimensions. Higher-dimensional PG runtimes remain the "
                "recorded campaign values; raw and summary tables are unchanged."
            )
            if plot_override_note not in config.setdefault("deviations", []):
                config["deviations"].append(plot_override_note)
    config["last_invocation_at"] = utc_now()
    config["last_invocation_mode"] = mode
    config["source_sha256_current"] = current_sources
    config.setdefault("invocations", []).append(
        {
            "at": utc_now(),
            "mode": mode,
            "argv": list(argv),
        }
    )
    atomic_write_json(path, config)
    return config


def verify_dataset_hash_against_raw(
    raw: pd.DataFrame, metadata: Mapping[str, object]
) -> None:
    if raw.empty:
        return
    subset = raw.loc[
        (pd.to_numeric(raw["p"], errors="coerce") == int(metadata["p"]))
        & (pd.to_numeric(raw["rep"], errors="coerce") == int(metadata["rep"]))
    ]
    hashes = {
        str(value)
        for value in subset["dataset_hash"].dropna()
        if str(value).strip()
    }
    if hashes and hashes != {str(metadata["dataset_hash"])}:
        raise RuntimeError(
            "cached dataset hash differs from existing PG/CS raw result at "
            "p=%d rep=%d" % (metadata["p"], metadata["rep"])
        )


def _status_lines(raw: pd.DataFrame) -> List[str]:
    lines = [
        "| Method | p | Status | Outer iter. | Algorithm time (s) | Dataset hash |",
        "|---|---:|---|---:|---:|---|",
    ]
    if raw.empty:
        lines.append("| — | — | pending | — | — | — |")
        return lines
    for _, row in _sort_raw(raw).iterrows():
        runtime = pd.to_numeric(pd.Series([row["algorithm_runtime_sec"]]), errors="coerce").iloc[0]
        runtime_text = "%.6f" % runtime if np.isfinite(runtime) else "—"
        iterations = pd.to_numeric(pd.Series([row["outer_iterations"]]), errors="coerce").iloc[0]
        iteration_text = str(int(iterations)) if np.isfinite(iterations) else "—"
        lines.append(
            "| %s | %d | %s | %s | %s | `%s` |"
            % (
                row["method"],
                int(row["p"]),
                row["status"],
                iteration_text,
                runtime_text,
                str(row["dataset_hash"])[:12],
            )
        )
    return lines


def write_readme(
    output_dir: Path,
    raw: pd.DataFrame,
    run_config: Mapping[str, object],
    plot_error: Optional[str],
) -> None:
    planned = len(P_LIST) * len(METHODS)
    completed = int(raw.shape[0])
    smoke_subset = raw.loc[
        pd.to_numeric(raw["p"], errors="coerce").isin([300, 600])
    ]
    smoke_complete = smoke_subset.shape[0] == 4
    deviations = list(run_config.get("deviations", []))
    if plot_error:
        deviations.append("Plot refresh failed: %s" % plot_error)
    campaign_flags = (
        " --fixed-group-size-30"
        if USE_FIXED_GROUP_SIZE_30
        else (
            (" --fixed-m5" if USE_FIXED_M5 else "")
            + (" --fixed-n" if USE_FIXED_N else "")
            + (
                " --pg-blockwise-tol-1e-5"
                if USE_PG_BLOCKWISE
                else (
                    " --pg-full-z-tol-1e-5"
                    if USE_PG_TIGHT_TOL
                    else (" --pg-full-z" if pg_uses_full_z() else "")
                )
            )
        )
    )
    plot_options = "".join(
        " --exclude-pg-p %d" % p
        for p in plot_only_excluded_pg_dimensions()
    )
    override_path = plot_overrides_path(output_dir)
    if override_path is not None:
        plot_options += " --plot-overrides %s" % override_path.resolve()
    if USE_PG_BLOCKWISE:
        plot_options += " --x-axis-start-zero"
    if USE_FIXED_GROUP_SIZE_30:
        plot_options += " --dimensions %s" % " ".join(
            str(p) for p in P_LIST
        )
    sample_size_text = (
        "`150,25,25` for every dimension"
        if USE_FIXED_N
        else "`p/4,p/12,p/12`"
    )
    group_contract_line = (
        (
            "- Group size is fixed at `30`; `M=p/30`; sample sizes are %s."
            % sample_size_text
        )
        if USE_FIXED_GROUP_SIZE_30
        else (
            "- The number of groups is fixed at `M=5`; each group has size "
            "`p/5`; sample sizes are %s." % sample_size_text
            if USE_FIXED_M5
            else "- Group size 60; `M=p/60`; sample sizes are %s."
            % sample_size_text
        )
    )
    p2400_line = (
        "- This manuscript campaign uses `p=300,600,1200,2400,3600,4800,6000`, "
        "for fourteen paired method calls under one group-size-30 contract."
        if USE_FIXED_GROUP_SIZE_30
        else (
            "- `p=2400` is included in this new fixed-M5 campaign; all 14 "
            "method calls are run under the same grouping contract."
            if USE_FIXED_M5
            else "- `p=2400` is an explicit follow-up extension to the "
            "markdown's original dimension list; the earlier 12 rows were "
            "retained unchanged."
        )
    )
    cs_configuration_line = (
        "- **LDPM-CS:** Sparse Group Lasso with five group penalties and one "
        "global L1 penalty: step `0.1`, `beta0=0.03`, `q=1.2`, `gamma=1`, "
        "group lambdas `0.01`, global L1 lambda `0.75`, lower/KKT "
        "initialization, 100 projection sweeps and projection tolerance `1e-7`."
        if USE_FIXED_M5
        else "- **LDPM-CS:** Sparse Group Lasso; the current formal scaled "
        "synthetic configuration is generalized to `M` groups: step `0.1`, "
        "`beta0=0.03`, `q=1.2`, `gamma=1`, group lambdas `0.01`, global L1 "
        "lambda `0.75`, lower/KKT initialization, 100 projection sweeps and "
        "projection tolerance `1e-7`."
    )
    caption_grouping = (
        "the group size is fixed at 30, so M=p/30"
        if USE_FIXED_GROUP_SIZE_30
        else (
            "the number of groups is fixed at M=5 and each group size is p/5"
            if USE_FIXED_M5
            else "the group size is fixed at 60, so M=p/60"
        )
    )
    caption_hyperparameters = (
        "five hyperparameters, whereas LDPM-CS is applied to Sparse Group "
        "Lasso with six hyperparameters"
        if USE_FIXED_M5
        else "M hyperparameters, whereas LDPM-CS is applied to Sparse Group "
        "Lasso with M+1 hyperparameters"
    )
    migration_commands = (
        []
        if USE_FIXED_M5 or USE_FIXED_GROUP_SIZE_30
        else [
            "python3 ldpm_scalability_experiment.py --rerun-pg-x-lambda",
            "python3 ldpm_scalability_experiment.py --rerun-pg-budget",
        ]
    )
    pg_stop_description = (
        "the maximum separately normalized change of `x`, `r`, `lambda`, "
        "`rho` and `xi`"
        if USE_PG_BLOCKWISE
        else (
            "the relative change of the complete state `full_z`"
            if pg_uses_full_z()
            else "the sum of the relative changes in `x` and `lambda`"
        )
    )
    pg_stop_rule = pg_stop_metric()
    pg_tolerance_text = format_tolerance(pg_stop_tolerance())

    lines = [
        "# LDPM dimension-runtime/iteration scalability experiment",
        "",
        "This directory is produced by `%s`." % (HERE / "ldpm_scalability_experiment.py"),
        "",
        "## Execution",
        "",
        "```bash",
        "python3 ldpm_scalability_experiment.py%s --smoke" % campaign_flags,
        "python3 ldpm_scalability_experiment.py%s --resume" % campaign_flags,
        "python3 ldpm_scalability_experiment.py%s --resume --rerun-failed"
        % campaign_flags,
        *migration_commands,
        "python3 plot.py --results-dir %s%s"
        % (output_dir, plot_options),
        "```",
        "",
        "The smoke command uses the formal configuration for the four `p=300,600` "
        "calls and writes into this same directory. A later `--resume` reuses every "
        "existing status by default, including time limits and failures.",
        "",
        "The cross-dimension configuration and stopping-behavior audit is recorded "
        "in `CONSISTENCY_AUDIT.md`.",
        "",
        "## Contract and explicit override",
        "",
        "- Dimensions: `%s`." % list(P_LIST),
        (
            "- Plot-only LDPM-PG exclusions: `%s`; these observations remain "
            "in `raw_results.csv` and `summary_results.csv`."
            % list(plot_only_excluded_pg_dimensions())
            if plot_only_excluded_pg_dimensions()
            else "- Plot-only LDPM-PG exclusions: none."
        ),
        (
            "- Plot-only LDPM-PG values are loaded from `%s`; the runtime axis "
            "is in seconds and all dimension axes start at zero."
            % override_path.resolve()
            if override_path is not None
            else "- Plot-only value overrides: none."
        ),
        "- One instance per dimension: `rep=0`; planned calls: `%d`." % planned,
        "- The current user instruction overrides the markdown's stale "
        "`N_REPEATS=10`/120-call wording.",
        p2400_line,
        "- Seed: `20260726 + 10000*rep + p`.",
        group_contract_line,
        "- Both methods at a fixed `(p,0)` reference the same cached NPZ and SHA-256.",
        "",
        "## Algorithm configurations",
        "",
        "- **LDPM-PG:** Group Lasso; initial trial step `0.01`, `beta0=1`, "
        "`q=0.3`; ones coefficient/lambda initialization and Fenchel dual "
        "initialization follow the current generic synthetic runner. Its current "
        "stop is %s." % pg_stop_description,
        cs_configuration_line,
        "- LDPM-PG uses `%s <= %s`; LDPM-CS retains `full_z <= 1e-4` "
        "together with its consensus residual. Both use patience 1 and 600 "
        "seconds per call. The uniform per-method outer-iteration budgets are "
        "%d for every LDPM-PG dimension and %d for every LDPM-CS dimension."
        % (
            pg_stop_rule,
            pg_tolerance_text,
            pg_max_outer_iter(),
            CS_MAX_OUTER_ITER,
        ),
        "- No capped or fixed-beta variant and no competing method is run.",
        "",
        "## Runtime accounting",
        "",
        "`algorithm_runtime_sec` is measured with `time.perf_counter()` immediately "
        "around `LeastSquaresLDPM.run_pgm` or `run_admm`. Standard state "
        "initialization is prepared before the call. Data generation/cache IO, "
        "result serialization, "
        "plotting and the post-hoc lower-level FISTA resolve are excluded. The "
        "post-hoc time is stored separately.",
        "",
        "All requested thread variables are fixed to one before NumPy import: `%s`."
        % THREAD_ENVIRONMENT,
        "",
        "## Environment and provenance",
        "",
        "- Git commit: `%s`."
        % (run_config.get("git_commit") or "unavailable (workspace has no .git)"),
        "- Versions: `%s`." % run_config.get("versions", {}),
        "- Raw results are atomically updated after every call; per-run JSON and "
        "compressed histories are under `logs/` and `histories/`.",
        "",
        "## Smoke status",
        "",
        "- Formal smoke set complete: `%s` (%d/4 rows)."
        % ("yes" if smoke_complete else "no", smoke_subset.shape[0]),
        "",
        "## Current run status",
        "",
        "- Completed rows: `%d/%d`." % (completed, planned),
        "",
        *_status_lines(raw),
        "",
        "## Deviations and failures",
        "",
    ]
    if deviations:
        lines.extend("- %s" % item for item in deviations)
    else:
        lines.append("- None.")
    failed = raw.loc[raw["status"] != "converged"] if not raw.empty else raw
    if not failed.empty:
        lines.append(
            "- Non-converged rows are retained exactly in `raw_results.csv`; "
            "they are excluded from convergence-only plotted values."
        )
    lines.extend(
        [
            "",
            "## Figure caption",
            "",
            "Problem-size scalability of LDPM-PG and LDPM-CS. For each feature "
            "dimension p, %s. Training, validation and test sample sizes are "
            "%s. LDPM-PG is applied to Group Lasso with %s. "
            "The single "
            "observed runtime and outer-iteration count at each dimension are "
            "reported without an artificial standard-deviation error bar. "
            "LDPM-PG uses the %s stop at %s; LDPM-CS uses the "
            "full-state relative-change and consensus stops."
            % (
                caption_grouping,
                "150, 25 and 25 at every dimension"
                if USE_FIXED_N
                else "p/4, p/12 and p/12",
                caption_hyperparameters,
                (
                    "blockwise maximum relative-change"
                    if USE_PG_BLOCKWISE
                    else (
                        "complete-state full_z relative-change"
                        if pg_uses_full_z()
                        else "x/lambda relative-change"
                    )
                ),
                format_tolerance(pg_stop_tolerance()),
            ),
            "",
        ]
    )
    atomic_write_text(output_dir / "README.md", "\n".join(lines))


def refresh_summary_and_plots(output_dir: Path) -> Optional[str]:
    plot_script = HERE / "plot.py"
    if not plot_script.exists():
        return "plot.py is missing"
    command = [
        sys.executable,
        str(plot_script),
        "--results-dir",
        str(output_dir),
    ]
    if USE_FIXED_GROUP_SIZE_30:
        command.extend(["--dimensions", *(str(p) for p in P_LIST)])
    for p in plot_only_excluded_pg_dimensions():
        command.extend(["--exclude-pg-p", str(p)])
    override_path = plot_overrides_path(output_dir)
    if override_path is not None:
        command.extend(["--plot-overrides", str(override_path.resolve())])
    if USE_PG_BLOCKWISE:
        command.append("--x-axis-start-zero")
    completed = subprocess.run(
        command,
        cwd=str(HERE),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode:
        return "plot script exited with code %d" % completed.returncode
    return None


def print_final_report(
    output_dir: Path,
    raw: pd.DataFrame,
    commit: Optional[str],
    plot_error: Optional[str],
) -> None:
    planned = len(P_LIST) * len(METHODS)
    print("\nSCALABILITY STATUS", flush=True)
    print("completed=%d planned=%d" % (raw.shape[0], planned), flush=True)
    for method in METHODS:
        for p in P_LIST:
            row = existing_row(raw, method, p, 0)
            status = "pending" if row is None else str(row["status"])
            print("%s p=%d: %s" % (method, p, status), flush=True)
    nonconverged = (
        raw.loc[raw["status"] != "converged", "status"].value_counts().to_dict()
        if not raw.empty
        else {}
    )
    print("nonconverged_status_counts=%s" % nonconverged, flush=True)
    print("results_dir=%s" % output_dir, flush=True)
    print(
        "combined_figure=%s"
        % (output_dir / "ldpm_scalability_runtime_iterations.png"),
        flush=True,
    )
    print("run_config=%s" % (output_dir / "run_config.json"), flush=True)
    print("git_commit=%s" % (commit or "unavailable"), flush=True)
    campaign_summary = (
        "seven dimensions, group size=30, M=p/30 (14 calls)"
        if USE_FIXED_GROUP_SIZE_30
        else "rep=0 plus p=2400 (14 calls)"
    )
    print(
        "deviations=user override %s; LDPM-PG "
        "uses %s stop at %s%s; git metadata unavailable%s"
        % (
            campaign_summary,
            pg_stop_metric(),
            format_tolerance(pg_stop_tolerance()),
            (
                ("; fixed M=5 and group size=p/5" if USE_FIXED_M5 else "")
                + ("; fixed sample sizes=150/25/25" if USE_FIXED_N else "")
            ),
            "; " + plot_error if plot_error else "",
        ),
        flush=True,
    )
    if raw.shape[0] < planned:
        resume_flags = (
            " --fixed-group-size-30"
            if USE_FIXED_GROUP_SIZE_30
            else (
                (" --fixed-m5" if USE_FIXED_M5 else "")
                + (" --fixed-n" if USE_FIXED_N else "")
                + (
                    " --pg-blockwise-tol-1e-5"
                    if USE_PG_BLOCKWISE
                    else (
                        " --pg-full-z-tol-1e-5"
                        if USE_PG_TIGHT_TOL
                        else (" --pg-full-z" if pg_uses_full_z() else "")
                    )
                )
            )
        )
        print(
            "resume_command=python3 %s%s --resume"
            % (
                Path(__file__).name,
                resume_flags,
            ),
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "result directory (default: results/ldpm_scalability, or "
            "results/ldpm_scalability_fixed_m5 with --fixed-m5, or "
            "results/ldpm_scalability_fixed_m5_fixed_n with both grouping and "
            "fixed-n flags, or the corresponding pg_full_z directory with "
            "--pg-full-z, or the separate tol_1e-5 directory with "
            "--pg-full-z-tol-1e-5, or the separate blockwise directory with "
            "--pg-blockwise-tol-1e-5)"
        ),
    )
    parser.add_argument(
        "--fixed-m5",
        action="store_true",
        help=(
            "use exactly five equal contiguous groups at every dimension, "
            "with group size p/5"
        ),
    )
    parser.add_argument(
        "--fixed-group-size-30",
        action="store_true",
        help=(
            "use the manuscript p=300,...,6000 campaign with group size 30 "
            "and M=p/30; this is the default and implies --fixed-n and "
            "--pg-blockwise-tol-1e-5"
        ),
    )
    parser.add_argument(
        "--fixed-n",
        action="store_true",
        help=(
            "hold n_train/n_validation/n_test at the p=300 values 150/25/25; "
            "requires --fixed-m5 or --fixed-group-size-30"
        ),
    )
    parser.add_argument(
        "--pg-full-z",
        action="store_true",
        help=(
            "use the complete-state full_z relative-change stop for LDPM-PG; "
            "requires --fixed-m5 and --fixed-n"
        ),
    )
    parser.add_argument(
        "--pg-full-z-tol-1e-5",
        action="store_true",
        help=(
            "run the independent complete-state PG campaign with LDPM-PG "
            "TOL=1e-5 while LDPM-CS remains at 1e-4; implies --pg-full-z and "
            "requires --fixed-m5 and --fixed-n"
        ),
    )
    parser.add_argument(
        "--pg-blockwise-tol-1e-5",
        action="store_true",
        help=(
            "run the independent PG campaign using the maximum separately "
            "normalized change of x, r, lambda, rho and xi at 1e-5 while "
            "LDPM-CS remains unchanged at 1e-4; requires --fixed-m5 and "
            "--fixed-n"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "run the four formal p=300,600 calls in the production directory; "
            "the results are reusable by --resume"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue missing calls while preserving every existing status",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="with --resume, rerun non-converged existing rows",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="regenerate summary, figures and README without running algorithms",
    )
    parser.add_argument(
        "--rerun-pg-x-lambda",
        action="store_true",
        help=(
            "back up the original full_z PG artifacts, leave LDPM-CS unchanged, "
            "and rerun only stale/missing LDPM-PG rows with x_lambda stopping"
        ),
    )
    parser.add_argument(
        "--rerun-pg-budget",
        action="store_true",
        help=(
            "rerun every LDPM-PG dimension from the standard initialization "
            "under the uniform audited 10000-outer-iteration budget while "
            "leaving all LDPM-CS artifacts unchanged"
        ),
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    global P_LIST, USE_FIXED_M5, USE_FIXED_GROUP_SIZE_30, USE_FIXED_N
    global USE_PG_FULL_Z
    global USE_PG_TIGHT_TOL, USE_PG_BLOCKWISE
    arguments = build_parser().parse_args(argv)
    USE_FIXED_M5 = bool(arguments.fixed_m5)
    legacy_selector = bool(
        arguments.fixed_m5
        or arguments.fixed_n
        or arguments.pg_full_z
        or arguments.pg_full_z_tol_1e_5
        or arguments.pg_blockwise_tol_1e_5
    )
    USE_FIXED_GROUP_SIZE_30 = bool(
        arguments.fixed_group_size_30 or not legacy_selector
    )
    if USE_FIXED_M5 and USE_FIXED_GROUP_SIZE_30:
        raise ValueError(
            "--fixed-m5 and --fixed-group-size-30 are alternative grouping "
            "campaigns"
        )
    P_LIST = (
        FIXED_GROUP_SIZE_30_P_LIST
        if USE_FIXED_GROUP_SIZE_30
        else DEFAULT_P_LIST
    )
    USE_FIXED_N = bool(arguments.fixed_n or USE_FIXED_GROUP_SIZE_30)
    USE_PG_TIGHT_TOL = bool(arguments.pg_full_z_tol_1e_5)
    USE_PG_BLOCKWISE = bool(
        arguments.pg_blockwise_tol_1e_5 or USE_FIXED_GROUP_SIZE_30
    )
    pg_selector_count = sum(
        (
            bool(arguments.pg_full_z),
            USE_PG_TIGHT_TOL,
            bool(arguments.pg_blockwise_tol_1e_5),
        )
    )
    if pg_selector_count > 1:
        raise ValueError(
            "--pg-full-z, --pg-full-z-tol-1e-5 and "
            "--pg-blockwise-tol-1e-5 are alternative campaign selectors"
        )
    USE_PG_FULL_Z = bool(arguments.pg_full_z or USE_PG_TIGHT_TOL)
    if USE_FIXED_N and not (USE_FIXED_M5 or USE_FIXED_GROUP_SIZE_30):
        raise ValueError(
            "--fixed-n requires --fixed-m5 or --fixed-group-size-30"
        )
    if (USE_PG_FULL_Z or USE_PG_BLOCKWISE) and not (
        (USE_FIXED_M5 or USE_FIXED_GROUP_SIZE_30) and USE_FIXED_N
    ):
        raise ValueError(
            "the PG stopping-criterion campaign requires a supported grouping "
            "campaign and --fixed-n"
        )
    validate_contract()
    if arguments.rerun_failed and not arguments.resume:
        raise ValueError("--rerun-failed requires --resume")
    if arguments.summarize_only and (
        arguments.smoke
        or arguments.rerun_failed
        or arguments.rerun_pg_x_lambda
        or arguments.rerun_pg_budget
    ):
        raise ValueError("--summarize-only cannot be combined with run options")
    if arguments.rerun_pg_x_lambda and (
        arguments.smoke or arguments.resume or arguments.rerun_failed
    ):
        raise ValueError(
            "--rerun-pg-x-lambda is a standalone resumable operation"
        )
    if arguments.rerun_pg_budget and (
        arguments.smoke
        or arguments.resume
        or arguments.rerun_failed
        or arguments.rerun_pg_x_lambda
    ):
        raise ValueError("--rerun-pg-budget is a standalone resumable operation")
    if (USE_FIXED_M5 or USE_FIXED_GROUP_SIZE_30) and (
        arguments.rerun_pg_x_lambda or arguments.rerun_pg_budget
    ):
        raise ValueError(
            "grouping campaign flags cannot be combined with the baseline-only "
            "PG migration options"
        )

    selected_results_dir = arguments.results_dir
    if selected_results_dir is None:
        selected_results_dir = default_results_dir()
    output_dir = selected_results_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("datasets", "histories", "logs"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    prepare_variant_datasets(output_dir)
    raw_path = output_dir / "raw_results.csv"
    backup_dir: Optional[Path] = None
    if arguments.rerun_pg_x_lambda:
        backup_dir = backup_full_z_pg_results(output_dir)

    if arguments.summarize_only:
        mode = "summarize-only"
    elif arguments.rerun_pg_x_lambda:
        mode = "pg-x-lambda-rerun"
    elif arguments.rerun_pg_budget:
        mode = "pg-budget-rerun"
    elif arguments.smoke:
        mode = "smoke"
    else:
        mode = "full"
    if USE_FIXED_GROUP_SIZE_30:
        mode = "fixed-group-size-30-fixed-n-pg-blockwise-tol-1e-5-" + mode
    elif USE_PG_BLOCKWISE:
        mode = "fixed-m5-fixed-n-pg-blockwise-tol-1e-5-" + mode
    elif USE_PG_TIGHT_TOL:
        mode = "fixed-m5-fixed-n-pg-full-z-tol-1e-5-" + mode
    elif USE_PG_FULL_Z:
        mode = "fixed-m5-fixed-n-pg-full-z-" + mode
    elif USE_FIXED_N:
        mode = "fixed-m5-fixed-n-" + mode
    elif USE_FIXED_M5:
        mode = "fixed-m5-" + mode
    run_config = create_or_validate_run_config(
        output_dir,
        mode,
        list(argv if argv is not None else sys.argv[1:]),
    )
    if backup_dir is not None:
        run_config["full_z_backup_dir"] = str(backup_dir.resolve())
        atomic_write_json(output_dir / "run_config.json", run_config)
    commit = run_config.get("git_commit")
    raw = load_raw_results(raw_path)

    if arguments.summarize_only:
        if raw.empty:
            raise RuntimeError("no raw results exist to summarize")
    else:
        if (
            raw_path.exists()
            and not arguments.resume
            and not arguments.rerun_pg_x_lambda
            and not arguments.rerun_pg_budget
        ):
            raise RuntimeError(
                "raw_results.csv already exists; use --resume to preserve and "
                "continue it"
            )
        dimensions = (300, 600) if arguments.smoke else P_LIST
        for p in dimensions:
            pending_methods = []
            method_scope = (
                ("LDPM-PG",)
                if arguments.rerun_pg_x_lambda or arguments.rerun_pg_budget
                else METHODS
            )
            for method in method_scope:
                prior = existing_row(raw, method, p, 0)
                if should_skip_current_contract(
                    prior,
                    method,
                    p,
                    arguments.rerun_failed,
                ):
                    print(
                        "SKIP %s p=%d rep=0 status=%s"
                        % (method, p, prior["status"]),
                        flush=True,
                    )
                else:
                    pending_methods.append(method)
            if not pending_methods:
                continue

            try:
                arrays, metadata = load_or_create_dataset(output_dir, p, 0)
                verify_dataset_hash_against_raw(raw, metadata)
            except Exception as exc:
                arrays = {}
                n_train, n_validation, n_test = sample_sizes(p)
                metadata = {
                    "p": p,
                    "M": group_count_for(p),
                    "group_size": group_size_for(p),
                    "n_train": n_train,
                    "n_validation": n_validation,
                    "n_test": n_test,
                    "rep": 0,
                    "seed": seed_for(p, 0),
                    "dataset_path": str(_dataset_file(output_dir, p, 0).resolve()),
                    "dataset_hash": "",
                    "actual_snr": np.nan,
                }
                dataset_traceback = traceback.format_exc()
                for method in pending_methods:
                    contract = algorithm_contract(method, p, group_count_for(p))
                    row = _empty_row(method, p, 0, metadata, contract, commit)
                    row["termination_message"] = "dataset exception: %s: %s" % (
                        type(exc).__name__,
                        exc,
                    )
                    row["finished_at"] = utc_now()
                    log_path = (
                        output_dir
                        / "logs"
                        / (
                            "%s_p%d_rep0.json"
                            % (method.lower().replace("-", "_"), p)
                        )
                    )
                    atomic_write_json(
                        log_path,
                        {
                            "row": row,
                            "algorithm_config": contract,
                            "traceback": dataset_traceback,
                        },
                    )
                    raw = upsert_raw_result(raw_path, raw, row)
                continue

            for method in pending_methods:
                print(
                    "START %s p=%d rep=0 seed=%d dataset=%s"
                    % (
                        method,
                        p,
                        metadata["seed"],
                        str(metadata["dataset_hash"])[:12],
                    ),
                    flush=True,
                )
                row = execute_run(
                    output_dir,
                    arrays,
                    metadata,
                    method,
                    commit,
                )
                raw = upsert_raw_result(raw_path, raw, row)
                print(
                    "DONE %s p=%d status=%s iter=%s runtime=%.6fs"
                    % (
                        method,
                        p,
                        row["status"],
                        row["outer_iterations"],
                        float(row["algorithm_runtime_sec"]),
                    ),
                    flush=True,
                )

    plot_error = refresh_summary_and_plots(output_dir)
    raw = load_raw_results(raw_path)
    write_readme(output_dir, raw, run_config, plot_error)
    print_final_report(output_dir, raw, commit, plot_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
