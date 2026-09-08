#!/usr/bin/env python3
"""Direct overlapping group Lasso experiments on the LIBSVM splice data.

This is a thin data/output adapter around the LDPM-CS, capped LDPM-CS,
LDMMA, and pinned VF-iDCA implementations in ``../methods.py``. Overlapping
groups act directly on the shared coefficient vector.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
OGL_DIR = SCRIPT_DIR.parent
for module_dir in (OGL_DIR, SCRIPT_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from data import SpliceSettings, load_splice  # noqa: E402
from methods import (  # noqa: E402
    DirectOGLProblem,
    MethodTimeout,
    OGLData,
    UPSTREAM_COMMIT,
    UPSTREAM_GENERIC_FIXED_CONFIG,
    UPSTREAM_REPOSITORY,
    UPSTREAM_SGL_FIXED_CONFIG,
    UPSTREAM_SOURCE_FILES,
    make_overlapping_groups,
    normalize_overlap_groups,
    run_ldmma,
    run_vfidca_upstream_generic_adapter,
    run_vfidca_upstream_sgl_adapter,
    save_state,
    time_limit,
    write_history,
)


def positive_int_or_none(value):
    if value in (None, "", "none", "None", "auto"):
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_methods(raw: str) -> List[str]:
    aliases = {
        "ldpm": "ldpm",
        "ldpm-cs": "ldpm",
        "uncapped": "ldpm",
        "ldpm-capped": "ldpm-capped",
        "ldpm-cs-c": "ldpm-capped",
        "capped": "ldpm-capped",
        "ldmma": "ldmma",
        "mm": "ldmma",
        "vf-idca": "vf-idca",
        "vfidca": "vf-idca",
        "vf": "vf-idca",
        "dc": "vf-idca",
        "vf-idca-generic": "vf-idca-generic",
        "vf-generic": "vf-idca-generic",
    }
    methods: List[str] = []
    for piece in raw.split(","):
        key = piece.strip().lower()
        if not key:
            continue
        if key not in aliases:
            raise ValueError("unknown method %r" % piece)
        canonical = aliases[key]
        if canonical not in methods:
            methods.append(canonical)
    if not methods:
        raise ValueError("at least one method is required")
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="ldpm,ldpm-capped,ldmma")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--standardize", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num-groups", type=int, default=11)
    parser.add_argument("--group-size", type=positive_int_or_none, default=10)
    parser.add_argument("--stride", type=positive_int_or_none, default=5)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument(
        "--results-dir",
        default=str(SCRIPT_DIR / "results" / "ogl_splice_direct"),
    )

    parser.add_argument("--ldpm-max-iter", type=int, default=100000)
    parser.add_argument("--ldpm-record-interval", type=int, default=100)
    parser.add_argument(
        "--record-snapshots",
        action="store_true",
        help="Store full x and lambda vectors at every recorded checkpoint.",
    )
    parser.add_argument(
        "--ldpm-stop-patience",
        type=int,
        default=1,
        help=(
            "Require this many consecutive full-state and consensus-residual "
            "hits; this prevents one-step crossings."
        ),
    )
    parser.add_argument("--beta0", type=float, default=1.04)
    parser.add_argument("--beta-power", type=float, default=0.3)
    parser.add_argument("--beta-max-capped", type=float, default=7.5)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--initial-lambda", type=float, default=0.1)
    parser.add_argument("--initial-r", type=float, default=0.1)
    parser.add_argument("--ldpm-init", choices=["ridge", "zero"], default="ridge")
    parser.add_argument("--ldpm-init-ridge", type=float, default=1e-3)
    parser.add_argument("--ldpm-init-dual", choices=["zero", "fenchel"], default="zero")
    parser.add_argument("--ldpm-step", type=float, default=1e-3)
    parser.add_argument("--ldpm-line-search-max-step", type=float, default=2e-2)
    parser.add_argument("--ldpm-line-search-min-step", type=float, default=1e-12)
    parser.add_argument("--ldpm-line-search-decay", type=float, default=0.5)
    parser.add_argument("--ldpm-line-search-growth", type=float, default=1.25)
    parser.add_argument("--ldpm-line-search-max-iter", type=int, default=50)

    parser.add_argument("--baseline-timeout", type=float, default=180.0)
    parser.add_argument("--ldmma-max-iter", type=int, default=1000)
    parser.add_argument("--ldmma-epsilon", type=float, default=1e-4)
    parser.add_argument("--ldmma-eta", type=float, default=1e-3)
    parser.add_argument("--solver", default="SCS")
    parser.add_argument("--solver-tol", type=float, default=1e-5)
    parser.add_argument("--solver-max-iters", type=int, default=20000)
    parser.add_argument("--cvxpy-verbose", action="store_true")
    parser.add_argument(
        "--vfidca-initial-r",
        type=float,
        default=None,
        help=(
            "Optional VF-iDCA initial group radius. Omit it to preserve the "
            "synthetic SGL example value 10; use 0.1 for the released iP_DCA "
            "algorithm fallback."
        ),
    )
    return parser


def prepare_problem(args: argparse.Namespace):
    settings = SpliceSettings(0, 0, 0, 0, "splice")
    settings.seed = args.seed
    settings.validation_fraction = args.validation_fraction
    settings.standardize = args.standardize
    source = load_splice(settings)
    groups = make_overlapping_groups(
        settings.num_features,
        args.num_groups,
        group_size=args.group_size,
        stride=args.stride,
    )
    groups = normalize_overlap_groups(
        groups,
        settings.num_features,
        add_singletons=False,
        require_coverage=True,
    )
    data = OGLData(
        source.X_train,
        source.y_train,
        source.X_validate,
        source.y_validate,
        source.X_test,
        source.y_test,
        groups,
    )
    return DirectOGLProblem(data, loss_scale="mean"), settings


def final_accuracy(problem: DirectOGLProblem, state: Dict[str, np.ndarray]):
    x = np.asarray(state["x"], dtype=float).reshape(-1)
    val_prediction = np.where(problem.a_val @ x >= 0.0, 1.0, -1.0)
    test_prediction = np.where(problem.a_test @ x >= 0.0, 1.0, -1.0)
    return (
        float(np.mean(val_prediction == problem.b_val)),
        float(np.mean(test_prediction == problem.b_test)),
    )


def state_feasibility(problem: DirectOGLProblem, state: Dict[str, np.ndarray]):
    if not {"x", "lambda", "r", "rho"}.issubset(state):
        return None, None
    x = np.asarray(state["x"], dtype=float).reshape(-1)
    lam = np.asarray(state["lambda"], dtype=float).reshape(-1)
    r = np.asarray(state["r"], dtype=float).reshape(-1)
    rho = np.asarray(state["rho"], dtype=float).reshape(-1)
    primal = max(
        0.0,
        float(np.max(problem.group_norms_x(x) - r)),
        float(np.max(-r)),
    )
    dual = max(
        0.0,
        float(np.max(problem.group_norms_rho(rho) - lam)),
        float(np.max(-lam)),
    )
    return primal, dual


def summarize(
    method: str,
    status: str,
    records: List[Dict[str, float]],
    state: Dict[str, np.ndarray],
    problem: DirectOGLProblem,
    tol: float,
    message: str = "",
):
    if not records:
        return {
            "method": method,
            "status": status,
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
            "message": message,
        }
    last = records[-1]
    val_accuracy, test_accuracy = final_accuracy(problem, state)
    primal_violation, dual_violation = state_feasibility(problem, state)
    if "x_lambda_stop" in last:
        stop_value = float(last["x_lambda_stop"])
        stop_metric = "x_lambda_stop"
        converged = bool(np.isfinite(stop_value) and stop_value <= tol)
        x_lambda_stop_value = stop_value
    elif "upstream_stop_value" in last:
        stop_value = float(last["upstream_stop_value"])
        stop_metric = "max(step_err, penalty)"
        converged = bool(last.get("native_converged", False))
        x_lambda_stop_value = None
    else:
        stop_value = float("nan")
        stop_metric = "unknown"
        converged = False
        x_lambda_stop_value = None
    return {
        "method": method,
        "status": status,
        "iterations": int(last["iteration"]),
        "time": float(last["time"]),
        "val_loss": float(last["val_loss"]),
        "test_loss": float(last["test_loss"]),
        "validation_accuracy": val_accuracy,
        "test_accuracy": test_accuracy,
        "x_lambda_stop": x_lambda_stop_value,
        "stop_value": stop_value if np.isfinite(stop_value) else None,
        "stop_metric": stop_metric,
        "step_err": float(last["step_err"]) if "step_err" in last else None,
        "penalty": float(last["penalty"]) if "penalty" in last else None,
        "beta": float(last["beta"]) if "beta" in last else None,
        "converged": converged,
        "psi": float(last["psi"]) if "psi" in last else None,
        "primal_violation": primal_violation,
        "dual_violation": dual_violation,
        "message": message,
    }


def run_ldpm(problem: DirectOGLProblem, args: argparse.Namespace, beta_max):
    return problem.run_ldpm_cs(
        max_iter=args.ldpm_max_iter,
        tol=args.tol,
        beta0=args.beta0,
        beta_power=args.beta_power,
        beta_max=beta_max,
        gamma=args.gamma,
        initial_lambda=args.initial_lambda,
        initial_r=args.initial_r,
        init_mode=args.ldpm_init,
        init_ridge=args.ldpm_init_ridge,
        init_dual=args.ldpm_init_dual,
        initial_step=args.ldpm_step,
        max_step=args.ldpm_line_search_max_step,
        min_step=args.ldpm_line_search_min_step,
        line_search_decay=args.ldpm_line_search_decay,
        line_search_growth=args.ldpm_line_search_growth,
        max_line_search_iter=args.ldpm_line_search_max_iter,
        record_interval=args.ldpm_record_interval,
        psi_target=None,
        stop_patience=args.ldpm_stop_patience,
        stop_mode="full_state",
        record_snapshots=args.record_snapshots,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    problem, settings = prepare_problem(args)
    tol_token = ("%.0e" % args.tol).replace("-", "m").replace("+", "")
    run_dir = Path(args.results_dir) / ("seed%d_tol%s" % (args.seed, tol_token))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Data: splice train=%d val=%d official_test=%d features=%d groups=%d"
        % (
            len(problem.b_tr),
            len(problem.b_val),
            len(problem.b_test),
            problem.p,
            problem.group_count,
        ),
        flush=True,
    )
    print(
        "Direct OGL stopping: full-state relative change and normalized "
        "consensus residual <= %.1e; LDPM confirmation=%d hits"
        % (args.tol, args.ldpm_stop_patience),
        flush=True,
    )

    summary = []

    def persist(slug, records, state):
        write_history(run_dir / (slug + "_history.csv"), records)
        save_state(run_dir / (slug + "_state.npz"), state)

    if "ldpm" in methods:
        print("Running direct LDPM-CS (uncapped).", flush=True)
        records, state = run_ldpm(problem, args, None)
        persist("ldpm", records, state)
        summary.append(summarize("LDPM-CS", "ok", records, state, problem, args.tol))

    if "ldpm-capped" in methods:
        print(
            "Running direct LDPM-CS-C (beta_max=%.6g)." % args.beta_max_capped,
            flush=True,
        )
        records, state = run_ldpm(problem, args, args.beta_max_capped)
        persist("ldpm_capped", records, state)
        summary.append(summarize("LDPM-CS-C", "ok", records, state, problem, args.tol))

    if "ldmma" in methods:
        print(
            "Running direct LDMMA (timeout %.1fs)." % args.baseline_timeout,
            flush=True,
        )
        try:
            with time_limit(args.baseline_timeout, "LDMMA"):
                records, state = run_ldmma(
                    problem,
                    args,
                    record_snapshots=args.record_snapshots,
                )
            persist("ldmma", records, state)
            row = summarize("LDMMA", "ok", records, state, problem, args.tol)
            feasibility_limit = max(10.0 * args.solver_tol, 1e-6)
            if max(row["primal_violation"], row["dual_violation"]) > feasibility_limit:
                row["status"] = "solver_inaccurate"
                row["message"] = (
                    "stopping residual met, but cone feasibility exceeds %.1e"
                    % feasibility_limit
                )
            summary.append(row)
        except MethodTimeout as exc:
            print(str(exc), flush=True)
            summary.append(summarize("LDMMA", "timeout", [], {}, problem, args.tol, str(exc)))
        except Exception as exc:  # noqa: BLE001
            print("LDMMA failed: %s" % exc, flush=True)
            summary.append(summarize("LDMMA", "failed", [], {}, problem, args.tol, str(exc)))

    if "vf-idca" in methods:
        vfidca_initial_r = (
            UPSTREAM_SGL_FIXED_CONFIG["initial_guess"]
            if args.vfidca_initial_r is None
            else args.vfidca_initial_r
        )
        print(
            "Running pinned upstream VF-iDCA SGL adapter "
            "(commit %s, fixed max_iter=%d, initial_r=%.6g%s)."
            % (
                UPSTREAM_COMMIT[:8],
                UPSTREAM_SGL_FIXED_CONFIG["MAX_ITERATION"],
                vfidca_initial_r,
                " official" if args.vfidca_initial_r is None else " override",
            ),
            flush=True,
        )
        try:
            records, state = run_vfidca_upstream_sgl_adapter(
                problem,
                args.tol,
                record_snapshots=args.record_snapshots,
                initial_r=args.vfidca_initial_r,
            )
            persist("vf_idca_upstream", records, state)
            vf_row = summarize(
                "VF-iDCA-upstream",
                "ok" if records[-1].get("native_converged", False) else "max_iter",
                records,
                state,
                problem,
                args.tol,
                (
                    "Pinned SGL config; TOL and OGL constraints changed; "
                    "initial_r=%g%s."
                    % (
                        vfidca_initial_r,
                        " (official)"
                        if args.vfidca_initial_r is None
                        else " (explicit override)",
                    )
                ),
            )
            summary.append(vf_row)
        except Exception as exc:  # noqa: BLE001
            print("VF-iDCA failed: %s" % exc, flush=True)
            summary.append(
                summarize(
                    "VF-iDCA-upstream",
                    "failed",
                    [],
                    {},
                    problem,
                    args.tol,
                    str(exc),
                )
            )

    if "vf-idca-generic" in methods:
        print(
            "Running literal upstream VF_iDCA.py/wLasso.py adapter "
            "(commit %s, fixed max_iter=%d)."
            % (UPSTREAM_COMMIT[:8], UPSTREAM_GENERIC_FIXED_CONFIG["MAX_ITERATION"]),
            flush=True,
        )
        try:
            records, state = run_vfidca_upstream_generic_adapter(problem, args.tol)
            persist("vf_idca_generic_upstream", records, state)
            summary.append(
                summarize(
                    "VF-iDCA-generic-upstream",
                    "ok" if records[-1].get("native_converged", False) else "max_iter",
                    records,
                    state,
                    problem,
                    args.tol,
                    "Literal root defaults; only TOL and OGL constraints changed.",
                )
            )
        except Exception as exc:  # noqa: BLE001
            print("VF-iDCA generic failed: %s" % exc, flush=True)
            summary.append(
                summarize(
                    "VF-iDCA-generic-upstream",
                    "failed",
                    [],
                    {},
                    problem,
                    args.tol,
                    str(exc),
                )
            )

    if args.vfidca_initial_r is None:
        vfidca_initial_r_source = "upstream_synthetic_sgl_example"
        vfidca_tuned_on_splice = False
    elif np.isclose(args.vfidca_initial_r, 0.1, rtol=0.0, atol=1e-15):
        vfidca_initial_r_source = "released_ipdca_algorithm_fallback"
        vfidca_tuned_on_splice = False
    else:
        vfidca_initial_r_source = "cli_override"
        vfidca_tuned_on_splice = True

    metadata = {
        "dataset": "splice",
        "test_dataset": "splice_test",
        "data_source": settings.data_source,
        "seed": args.seed,
        "split_hash": settings.split_hash,
        "dataset_fingerprint": settings.dataset_fingerprint,
        "data_shapes": {
            "train": list(problem.a_tr.shape),
            "validation": list(problem.a_val.shape),
            "test": list(problem.a_test.shape),
        },
        "standardized": bool(args.standardize),
        "formulation": "direct_overlapping_group_penalty",
        "groups": [group.tolist() for group in problem.groups],
        "tol": args.tol,
        "stopping_criterion": (
            "full-state relative change <= tol and normalized consensus "
            "residual <= tol; LDPM requires ldpm_stop_patience consecutive hits"
        ),
        "ldpm_subproblem_scaling": "validation_loss / beta_k + psi + consensus_AL",
        "beta0": args.beta0,
        "beta_power": args.beta_power,
        "beta_max_capped": args.beta_max_capped,
        "gamma": args.gamma,
        "initial_lambda": args.initial_lambda,
        "initial_r": args.initial_r,
        "ldpm_init": args.ldpm_init,
        "ldpm_init_ridge": args.ldpm_init_ridge,
        "ldpm_init_dual": args.ldpm_init_dual,
        "ldpm_step": args.ldpm_step,
        "ldpm_line_search_max_step": args.ldpm_line_search_max_step,
        "ldpm_line_search_min_step": args.ldpm_line_search_min_step,
        "ldpm_line_search_decay": args.ldpm_line_search_decay,
        "ldpm_line_search_growth": args.ldpm_line_search_growth,
        "ldpm_line_search_max_iter": args.ldpm_line_search_max_iter,
        "record_snapshots": bool(args.record_snapshots),
        "ldpm_stop_patience": args.ldpm_stop_patience,
        "ldmma_timeout_seconds": args.baseline_timeout,
        "solver": args.solver,
        "solver_tol": args.solver_tol,
        "ldmma_feasibility_limit": max(10.0 * args.solver_tol, 1e-6),
        "vf_idca_upstream": {
            "repository": UPSTREAM_REPOSITORY,
            "commit": UPSTREAM_COMMIT,
            "source_files": list(UPSTREAM_SOURCE_FILES),
            "fixed_sgl_config": UPSTREAM_SGL_FIXED_CONFIG,
            "requested_initial_r": args.vfidca_initial_r,
            "effective_initial_r": (
                UPSTREAM_SGL_FIXED_CONFIG["initial_guess"]
                if args.vfidca_initial_r is None
                else args.vfidca_initial_r
            ),
            "initial_r_source": vfidca_initial_r_source,
            "only_changes": (
                [
                    "TOL set to this run's requested tolerance",
                    "SGL non-overlapping group/L1 constraints replaced by direct overlapping group constraints",
                ]
                + (
                    []
                    if args.vfidca_initial_r is None
                    else [
                        "initial group radius r_0 changed from the synthetic SGL example value 10"
                    ]
                )
            ),
            "stopping_criterion": "step_err < TOL and penalty < TOL",
            "tuned_on_splice": vfidca_tuned_on_splice,
            "validation_or_test_tuning": False,
            "generic_root_config": UPSTREAM_GENERIC_FIXED_CONFIG,
        },
    }
    with (run_dir / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    print("\nMethod summary")
    for row in summary:
        print(
            "%-12s %-8s iter=%-6s val=%-10s test=%-10s acc=%-8s stop=%s"
            % (
                row["method"],
                row["status"],
                row["iterations"],
                "%.6g" % row["val_loss"] if row["val_loss"] is not None else "-",
                "%.6g" % row["test_loss"] if row["test_loss"] is not None else "-",
                "%.4f" % row["test_accuracy"] if row["test_accuracy"] is not None else "-",
                "%.3e" % row["stop_value"] if row["stop_value"] is not None else row["message"],
            )
        )
    print("Saved results under %s" % run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
