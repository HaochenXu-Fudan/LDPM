"""Run the paired sparse/low-rank matrix-sensing campaign.

This file is intentionally only an orchestration layer.  The data generator,
VF-iDCA implementation, and LDPM-CS implementation remain in
``matrix_algorithms.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

try:
    import scs
except ImportError:  # pragma: no cover - cvxpy reports the missing solver.
    scs = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from matrix_algorithms import (  # noqa: E402
    LDPM,
    VF_iDCA,
    MatrixSetting,
    generate_matrix_completion_data,
)


METHOD_LABELS = {
    "vf-idca": "VF-iDCA",
    "ldpm-cs": "LDPM-CS",
    "ldpm-cs-c": "LDPM-CS-C",
}
METHOD_SLUGS = {
    "vf-idca": "vf_idca",
    "ldpm-cs": "ldpm_cs",
    "ldpm-cs-c": "ldpm_cs_c",
}
NUMERIC_SUMMARY_FIELDS = (
    "wall_time",
    "internal_time",
    "iteration",
    "validation_error",
    "test_error",
    "stop_residual",
)


def _csv_list(value, cast=str):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _finite_or_none(value):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if np.isfinite(number) else None


def _json_clean(value):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return _finite_or_none(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_json_clean(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _dataset_fingerprint(data_info):
    digest = hashlib.sha256()
    for array in (
        data_info.data.S_true,
        data_info.data.b_train,
        data_info.data.b_val,
        data_info.data.b_test,
    ):
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def _mean_std(series):
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return None, None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size >= 2 else None
    return mean, std


def _fmt(value, digits=6):
    value = _finite_or_none(value)
    if value is None:
        return "--"
    return f"{value:.{digits}g}"


def _fmt_pair(mean, std, digits=6):
    if _finite_or_none(mean) is None:
        return "--"
    if _finite_or_none(std) is None:
        return _fmt(mean, digits)
    return f"{_fmt(mean, digits)} ± {_fmt(std, digits)}"


def _aggregate(raw_rows):
    raw = pd.DataFrame(raw_rows)
    if raw.empty:
        return pd.DataFrame()
    aggregate_rows = []
    for (size, method), group in raw.groupby(["size", "method"], sort=False):
        finite_mask = group["status"] != "error"
        for field in NUMERIC_SUMMARY_FIELDS:
            finite_mask &= np.isfinite(pd.to_numeric(group[field], errors="coerce"))
        valid = group[finite_mask].copy()
        item = {
            "size": int(size),
            "method": method,
            "n_requested": int(len(group)),
            "n_valid": int(len(valid)),
            "n_converged": int(valid["converged"].fillna(False).astype(bool).sum()),
            "n_cap_reached": int(valid["cap_reached"].fillna(False).astype(bool).sum()),
        }
        for field in NUMERIC_SUMMARY_FIELDS:
            mean, std = _mean_std(valid[field]) if field in valid else (None, None)
            item[f"{field}_mean"] = mean
            item[f"{field}_std"] = std
        aggregate_rows.append(item)
    return pd.DataFrame(aggregate_rows)


def _write_report(output_dir, config, raw_rows, complete):
    raw = pd.DataFrame(raw_rows)
    aggregate = _aggregate(raw_rows)
    raw.to_csv(output_dir / "raw_runs.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)

    lines = [
        "# Sparse low-rank matrix-sensing campaign",
        "",
        f"Campaign status: **{'complete' if complete else 'in progress'}**.",
        "",
        "The reported errors are `0.5 * mean squared residual` on independent "
        "Gaussian validation/test measurements. Time is the complete wall-clock "
        "duration of the method call; data generation and artifact writing are excluded.",
        "",
        "## Aggregate results",
        "",
        "| Size | Method | Valid | Converged | Cap reached | Time (s) | Internal time (s) | Iter. | Val. error | Test error | Stop residual |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate.to_dict("records"):
        lines.append(
            "| {size}×{size} | {method} | {n_valid}/{n_requested} | "
            "{n_converged}/{n_valid} | {n_cap_reached}/{n_valid} | {wall} | "
            "{internal} | {iteration} | {validation} | {test} | {stop} |".format(
                size=int(row["size"]),
                method=row["method"],
                n_valid=int(row["n_valid"]),
                n_requested=int(row["n_requested"]),
                n_converged=int(row["n_converged"]),
                n_cap_reached=int(row["n_cap_reached"]),
                wall=_fmt_pair(row["wall_time_mean"], row["wall_time_std"]),
                internal=_fmt_pair(row["internal_time_mean"], row["internal_time_std"]),
                iteration=_fmt_pair(row["iteration_mean"], row["iteration_std"]),
                validation=_fmt_pair(
                    row["validation_error_mean"], row["validation_error_std"]
                ),
                test=_fmt_pair(row["test_error_mean"], row["test_error_std"]),
                stop=_fmt_pair(row["stop_residual_mean"], row["stop_residual_std"]),
            )
        )

    lines.extend(
        [
            "",
            "Standard deviations are sample standard deviations (`ddof=1`).",
            "",
            "## Per-run results",
            "",
            "| Size | Seed | Order | Method | Status | Iter. | Time (s) | Val. error | Test error | Stop residual | Cap reached |",
            "|---:|---:|---:|:---|:---|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for row in raw_rows:
        lines.append(
            "| {size}×{size} | {seed} | {order} | {method} | {status} | "
            "{iteration} | {wall} | {validation} | {test} | {stop} | {cap} |".format(
                size=row["size"],
                seed=row["seed"],
                order=row["execution_order"],
                method=row["method"],
                status=row["status"],
                iteration=_fmt(row.get("iteration"), 7),
                wall=_fmt(row.get("wall_time"), 7),
                validation=_fmt(row.get("validation_error"), 7),
                test=_fmt(row.get("test_error"), 7),
                stop=_fmt(row.get("stop_residual"), 7),
                cap="yes" if row.get("cap_reached") else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Fixed contract",
            "",
            f"- Seeds: `{config['experiment']['seeds']}`.",
            f"- Sizes: `{config['experiment']['sizes']}`.",
            f"- Algorithm tolerance: `{config['algorithm']['tol']}`.",
            f"- SCS tolerance: `{config['solver']['tol']}`.",
            f"- VF-iDCA maximum iterations: `{config['algorithm']['vf_idca']['max_iter']}`.",
            "- LDPM-CS beta schedule: "
            f"`beta0={config['algorithm']['ldpm']['beta0']}, "
            f"q={config['algorithm']['ldpm']['beta_power']}, "
            f"beta_max={config['algorithm']['ldpm']['uncapped_beta_max']}`.",
            "- LDPM-CS-C beta schedule: "
            f"`beta0={config['algorithm']['ldpm']['beta0']}, "
            f"q={config['algorithm']['ldpm']['beta_power']}, "
            f"beta_max={config['algorithm']['ldpm']['capped_beta_max']}`.",
            "",
            "VF-iDCA is intentionally omitted at sizes other than 60×60 unless the "
            "smoke-test-only override is supplied. `internal_time` preserves the "
            "algorithm's legacy timer; for VF-iDCA it excludes the initialization "
            "lower solve and final constrained solve, so the primary comparison uses "
            "the complete `wall_time`.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(
        output_dir / "progress.json",
        {
            "complete": complete,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "runs_recorded": len(raw_rows),
        },
    )


def _save_state(path, result):
    state = result.attrs.get("solution_states", {}).get("latest", {})
    arrays = {}
    for key, value in state.items():
        if isinstance(value, (str, bytes)):
            continue
        arrays[key] = np.asarray(value)
    if arrays:
        np.savez_compressed(path, **arrays)


def _run_method(
    method_key,
    data_info,
    args,
    size,
    seed,
    execution_order,
    dataset_id,
    run_dir,
):
    common_solver = {
        "cvxpy_solver": args.cvxpy_solver,
        "cvxpy_tol": args.cvxpy_tol,
        "cvxpy_max_iter": args.cvxpy_max_iter,
    }
    if method_key == "vf-idca":
        setting = {
            **common_solver,
            "MAX_ITERATION": args.vf_max_iter,
            "MIN_ITERATION": args.vf_min_iter,
            "TOL": args.tol,
            "rho": args.vf_rho,
            "alpha0": args.vf_alpha0,
            "initial_lambda": [
                args.vf_initial_lambda_l1,
                args.vf_initial_lambda_nuclear,
            ],
        }
        runner = VF_iDCA
        max_iter = args.vf_max_iter
        min_iter = args.vf_min_iter
        beta_max = None
    else:
        beta_max = (
            args.uncapped_beta_max
            if method_key == "ldpm-cs"
            else args.capped_beta_max
        )
        setting = {
            **common_solver,
            "MAX_ITERATION": args.ldpm_max_iter,
            "MIN_ITERATION": args.ldpm_min_iter,
            "TOL": args.tol,
            "step_size": args.ldpm_step_size,
            "gamma": args.ldpm_gamma,
            "beta0": args.ldpm_beta0,
            "beta_power": args.ldpm_beta_power,
            "beta_max": beta_max,
        }
        runner = LDPM
        max_iter = args.ldpm_max_iter
        min_iter = args.ldpm_min_iter

    method_label = METHOD_LABELS[method_key]
    print(
        f"[{size}x{size} seed={seed}] start {method_label} "
        f"(order={execution_order})",
        flush=True,
    )
    started_at = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    try:
        result = runner(data_info, setting)
        wall_time = time.perf_counter() - wall_start
        if result.empty:
            raise RuntimeError(f"{method_label} returned an empty result table.")
        final = result.iloc[-1]
        iteration = int(final["iteration"])
        if method_key == "vf-idca":
            stop_residual = max(
                float(final.get("step_err", np.inf)),
                float(final.get("vfidca_t", np.inf)),
            )
        else:
            stop_residual = max(
                float(final.get("step_err", np.inf)),
                float(final.get("consensus_residual", np.inf)),
            )
        critical_endpoint = {
            "wall_time": wall_time,
            "internal_time": final["time"],
            "train_error": final["train_error"],
            "validation_error": final["validation_error"],
            "test_error": final["test_error"],
            "stop_residual": stop_residual,
        }
        nonfinite = [
            name
            for name, value in critical_endpoint.items()
            if not np.isfinite(float(value))
        ]
        if nonfinite:
            raise FloatingPointError(
                f"{method_label} returned non-finite endpoint fields: {nonfinite}"
            )
        converged = bool(iteration >= min_iter and stop_residual <= args.tol)
        status = "converged" if converged else (
            "max_iter" if iteration >= max_iter else "stopped_without_tolerance"
        )
        beta_values = (
            pd.to_numeric(result["beta"], errors="coerce").dropna().to_numpy(dtype=float)
            if "beta" in result
            else np.array([], dtype=float)
        )
        cap_reached = bool(
            method_key == "ldpm-cs-c"
            and beta_values.size
            and np.max(beta_values) >= float(beta_max) * (1.0 - 1e-12)
        )
        row = {
            "size": size,
            "seed": seed,
            "execution_order": execution_order,
            "dataset_id": dataset_id,
            "method_key": method_key,
            "method": method_label,
            "status": status,
            "converged": converged,
            "iteration": iteration,
            "wall_time": float(wall_time),
            "internal_time": float(final["time"]),
            "train_error": float(final["train_error"]),
            "validation_error": float(final["validation_error"]),
            "test_error": float(final["test_error"]),
            "stop_residual": float(stop_residual),
            "step_err": _finite_or_none(final.get("step_err")),
            "consensus_residual": _finite_or_none(final.get("consensus_residual")),
            "vfidca_t": _finite_or_none(final.get("vfidca_t")),
            "ldpm_psi": _finite_or_none(final.get("ldpm_psi")),
            "ll_feasibility": _finite_or_none(final.get("ll_feasibility")),
            "lambda_l1": _finite_or_none(final.get("lambda_l1")),
            "lambda_nuclear": _finite_or_none(final.get("lambda_nuclear")),
            "beta_final": _finite_or_none(final.get("beta")),
            "beta_max": beta_max,
            "cap_reached": cap_reached,
            "tol": args.tol,
            "cvxpy_tol": args.cvxpy_tol,
            "max_iter": max_iter,
            "min_iter": min_iter,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_type": None,
            "error_message": None,
        }
        result.to_csv(run_dir / "history.csv", index=False)
        _save_state(run_dir / "final_state.npz", result)
        _write_json(run_dir / "setting.json", setting)
        _write_json(run_dir / "summary.json", row)
        print(
            f"[{size}x{size} seed={seed}] done {method_label}: "
            f"status={status}, iter={iteration}, wall={wall_time:.3f}s, "
            f"val={row['validation_error']:.6e}, "
            f"test={row['test_error']:.6e}, residual={stop_residual:.3e}",
            flush=True,
        )
        return row
    except Exception as exc:
        wall_time = time.perf_counter() - wall_start
        error_text = traceback.format_exc()
        (run_dir / "traceback.txt").write_text(error_text, encoding="utf-8")
        row = {
            "size": size,
            "seed": seed,
            "execution_order": execution_order,
            "dataset_id": dataset_id,
            "method_key": method_key,
            "method": method_label,
            "status": "error",
            "converged": False,
            "iteration": None,
            "wall_time": float(wall_time),
            "internal_time": None,
            "train_error": None,
            "validation_error": None,
            "test_error": None,
            "stop_residual": None,
            "step_err": None,
            "consensus_residual": None,
            "vfidca_t": None,
            "ldpm_psi": None,
            "ll_feasibility": None,
            "lambda_l1": None,
            "lambda_nuclear": None,
            "beta_final": None,
            "beta_max": beta_max,
            "cap_reached": False,
            "tol": args.tol,
            "cvxpy_tol": args.cvxpy_tol,
            "max_iter": max_iter,
            "min_iter": min_iter,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        _write_json(run_dir / "setting.json", setting)
        _write_json(run_dir / "summary.json", row)
        print(
            f"[{size}x{size} seed={seed}] ERROR {method_label} after "
            f"{wall_time:.3f}s: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return row


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", default="60,100")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument(
        "--methods",
        default="vf-idca,ldpm-cs,ldpm-cs-c",
        help="comma-separated subset of vf-idca,ldpm-cs,ldpm-cs-c",
    )
    parser.add_argument("--allow-vf-other-sizes", action="store_true")
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--sparsity", type=float, default=0.2)
    parser.add_argument("--snr", type=float, default=5.0)
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--tol", type=float, default=1e-4)

    parser.add_argument("--cvxpy-solver", default="SCS")
    parser.add_argument("--cvxpy-tol", type=float, default=1e-4)
    parser.add_argument("--cvxpy-max-iter", type=int, default=2500)

    parser.add_argument("--vf-max-iter", type=int, default=5)
    parser.add_argument("--vf-min-iter", type=int, default=1)
    parser.add_argument("--vf-rho", type=float, default=1e-2)
    parser.add_argument("--vf-alpha0", type=float, default=10.0)
    parser.add_argument("--vf-initial-lambda-l1", type=float, default=1e-3)
    parser.add_argument("--vf-initial-lambda-nuclear", type=float, default=1e-3)

    parser.add_argument("--ldpm-max-iter", type=int, default=2000)
    parser.add_argument("--ldpm-min-iter", type=int, default=100)
    parser.add_argument("--ldpm-step-size", type=float, default=2e-2)
    parser.add_argument("--ldpm-gamma", type=float, default=10.0)
    parser.add_argument("--ldpm-beta0", type=float, default=1e-3)
    parser.add_argument("--ldpm-beta-power", type=float, default=0.3)
    parser.add_argument("--uncapped-beta-max", type=float, default=1e6)
    parser.add_argument("--capped-beta-max", type=float, default=5e-3)
    return parser


def main():
    args = _build_parser().parse_args()
    sizes = _csv_list(args.sizes, int)
    seeds = _csv_list(args.seeds, int)
    methods = _csv_list(args.methods, str)
    unknown = sorted(set(methods) - set(METHOD_LABELS))
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}")
    if not sizes or not seeds or not methods:
        raise ValueError("sizes, seeds, and methods must all be nonempty.")
    if args.output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; use a fresh path: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True)

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "sizes": sizes,
            "seeds": seeds,
            "methods": [METHOD_LABELS[item] for item in methods],
            "same_data_object_within_size_seed": True,
            "vf_idca_size_restriction": None
            if args.allow_vf_other_sizes
            else "60x60 only",
            "execution_order_rule": (
                "At 60x60, rotate [VF-iDCA, LDPM-CS, LDPM-CS-C] by repeat "
                "position (three-method Latin rotation). At other sizes, alternate "
                "the two LDPM orders by repeat position."
            ),
        },
        "data": {
            "rank": args.rank,
            "sparsity": args.sparsity,
            "snr": args.snr,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.val_fraction,
            "test_fraction": args.test_fraction,
        },
        "solver": {
            "name": args.cvxpy_solver,
            "tol": args.cvxpy_tol,
            "max_iter": args.cvxpy_max_iter,
        },
        "algorithm": {
            "tol": args.tol,
            "vf_idca": {
                "max_iter": args.vf_max_iter,
                "min_iter": args.vf_min_iter,
                "rho": args.vf_rho,
                "alpha0": args.vf_alpha0,
                "initial_lambda": [
                    args.vf_initial_lambda_l1,
                    args.vf_initial_lambda_nuclear,
                ],
            },
            "ldpm": {
                "max_iter": args.ldpm_max_iter,
                "min_iter": args.ldpm_min_iter,
                "step_size": args.ldpm_step_size,
                "gamma": args.ldpm_gamma,
                "beta0": args.ldpm_beta0,
                "beta_power": args.ldpm_beta_power,
                "uncapped_beta_max": args.uncapped_beta_max,
                "capped_beta_max": args.capped_beta_max,
            },
        },
        "timing": {
            "primary": "perf_counter around complete method call",
            "excluded": ["data generation", "artifact serialization", "aggregation"],
            "legacy_internal_timer_also_saved": True,
        },
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "cvxpy": cp.__version__,
            "scs": None if scs is None else scs.__version__,
            "installed_cvxpy_solvers": cp.installed_solvers(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
    }
    _write_json(args.output_dir / "config.json", config)

    raw_rows = []
    for size in sizes:
        for seed_position, seed in enumerate(seeds):
            setting = MatrixSetting(
                num_rows=size,
                num_cols=size,
                rank=args.rank,
                sparsity=args.sparsity,
                snr=args.snr,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                print_flag=True,
            )
            data_info = generate_matrix_completion_data(setting, seed=seed)
            data_info.data_index = seed
            dataset_id = f"size_{size:04d}_seed_{seed:04d}"
            dataset_dir = args.output_dir / "runs" / dataset_id
            dataset_dir.mkdir(parents=True)
            _write_json(
                dataset_dir / "dataset.json",
                {
                    "dataset_id": dataset_id,
                    "fingerprint": _dataset_fingerprint(data_info),
                    "size": [size, size],
                    "seed": seed,
                    "rank": args.rank,
                    "requested_sparsity": args.sparsity,
                    "actual_sparsity": float(
                        np.mean(np.abs(data_info.data.S_sparse) > 0.0)
                    ),
                    "snr": args.snr,
                    "num_train": len(data_info.data.b_train),
                    "num_validation": len(data_info.data.b_val),
                    "num_test": len(data_info.data.b_test),
                    "noise_scale": data_info.data.noise_std,
                },
            )

            vf_allowed = size == 60 or args.allow_vf_other_sizes
            if vf_allowed:
                base_order = ["vf-idca", "ldpm-cs", "ldpm-cs-c"]
                offset = seed_position % len(base_order)
                rotated = base_order[offset:] + base_order[:offset]
                order = [item for item in rotated if item in methods]
            else:
                base_order = (
                    ["ldpm-cs", "ldpm-cs-c"]
                    if seed_position % 2 == 0
                    else ["ldpm-cs-c", "ldpm-cs"]
                )
                order = [item for item in base_order if item in methods]
            for execution_order, method_key in enumerate(order, 1):
                run_dir = dataset_dir / METHOD_SLUGS[method_key]
                run_dir.mkdir()
                raw_rows.append(
                    _run_method(
                        method_key,
                        data_info,
                        args,
                        size,
                        seed,
                        execution_order,
                        dataset_id,
                        run_dir,
                    )
                )
                _write_report(args.output_dir, config, raw_rows, complete=False)

            del data_info

    _write_report(args.output_dir, config, raw_rows, complete=True)
    errors = [row for row in raw_rows if row["status"] == "error"]
    print(
        f"Campaign complete: {len(raw_rows) - len(errors)}/{len(raw_rows)} "
        f"runs produced results. Report: {args.output_dir / 'report.md'}",
        flush=True,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
