#!/usr/bin/env python3
"""Validation-only screening for a constant LDPM beta on School data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from school_experiment import (
    LDPMConfig,
    SchoolLDPMProblem,
    TaskLossOperator,
    feasible_lower_solve,
    load_and_preprocess_school,
    lower_objective,
    run_ldpm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("school.mat"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "fixed_beta_validation_screen",
    )
    parser.add_argument("--betas", default="0.5,1,2,5,10,15,20,30")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--record-interval", type=int, default=100)
    return parser.parse_args()


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    betas = [float(value.strip()) for value in args.betas.split(",") if value.strip()]
    if not betas or any(beta <= 0.0 for beta in betas):
        raise ValueError("all fixed beta candidates must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_preprocess_school(args.data, args.seed)
    training = TaskLossOperator(data.train_a, data.train_b)
    validation = TaskLossOperator(data.validation_a, data.validation_b)
    problem = SchoolLDPMProblem(training, validation)
    rows: List[Dict[str, object]] = []

    for beta in betas:
        config = LDPMConfig(
            max_iter=args.max_iter,
            tol=args.tol,
            beta0=beta,
            beta_power=0.0,
            beta_max=beta,
            max_time=600.0,
            record_interval=args.record_interval,
        )
        print(f"Running fixed beta={beta:g}...", flush=True)
        _, state, summary, _ = run_ldpm(problem, "ldpm", config)
        raw_w = state["W"]
        lambda_l1 = state["lambda_l1"]
        lambda_nuclear = float(state["lambda_nuclear"])
        feasible_w, lower = feasible_lower_solve(
            training, lambda_l1, lambda_nuclear, raw_w
        )
        raw_objective = lower_objective(
            training, raw_w, lambda_l1, lambda_nuclear
        )
        gap = raw_objective - float(lower["lower_objective"])
        row: Dict[str, object] = {
            "fixed_beta": beta,
            "iterations": int(summary["iterations"]),
            "time": float(summary["time"]),
            "status": str(summary["status"]),
            "x_lambda_stop": float(summary["x_lambda_stop"]),
            "validation_rmse_raw": math.sqrt(validation.mse(raw_w)),
            "validation_rmse_feasible": math.sqrt(validation.mse(feasible_w)),
            "feasibility": max(gap, 0.0) / validation.n,
            "lower_status": str(lower["status"]),
        }
        rows.append(row)
        np.savez_compressed(
            args.output_dir / f"fixed_beta_{beta:g}_state.npz",
            W=raw_w,
            feasible_W=feasible_w,
            lambda_l1=lambda_l1,
            lambda_nuclear=np.asarray(lambda_nuclear),
        )
        print(
            f"  val_rmse={row['validation_rmse_feasible']:.6f} "
            f"raw={row['validation_rmse_raw']:.6f} "
            f"feasibility={row['feasibility']:.3e}",
            flush=True,
        )

    selected = min(rows, key=lambda row: float(row["validation_rmse_feasible"]))
    save_csv(args.output_dir / "all_runs.csv", rows)
    (args.output_dir / "selected_validation_config.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "test_split_used": False,
                "selection_metric": "minimum feasible validation RMSE",
                "betas": betas,
                "seed": args.seed,
                "max_iter": args.max_iter,
                "tol": args.tol,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Selected from validation only:", json.dumps(selected, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
