#!/usr/bin/env python3
"""Validation-only beta screening for the School LDPM experiment.

The test split is deliberately not constructed or evaluated in this script.
After a configuration is selected from ``pair_summary.csv``, run the main
experiment once to attach test metrics.
"""

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
    METHOD_LABELS,
    SchoolLDPMProblem,
    TaskLossOperator,
    feasible_lower_solve,
    load_and_preprocess_school,
    lower_objective,
    run_ldpm,
)


DEFAULT_CONFIGS = (
    ("b3_q006_c4p5", 3.0, 0.06, 4.5),
    ("b3p5_q004_c4p5", 3.5, 0.04, 4.5),
    ("b3p8_q003_c4p5", 3.8, 0.03, 4.5),
    ("b4p2_q002_c4p5", 4.2, 0.02, 4.5),
    ("b1_q015_c4p5", 1.0, 0.15, 4.5),
    ("b1_q016_c4p5", 1.0, 0.16, 4.5),
    ("b1_q0165_c4p5", 1.0, 0.165, 4.5),
    ("b1_q017_c4p5", 1.0, 0.17, 4.5),
    ("b1_q018_c4p5", 1.0, 0.18, 4.5),
    ("b3_q005_c4p5", 3.0, 0.05, 4.5),
    ("b2p5_q007_c4p4", 2.5, 0.07, 4.4),
    ("b1_q02_c4p5", 1.0, 0.2, 4.5),
    ("b1_q02_c5", 1.0, 0.2, 5.0),
    ("b2_q01_c4p5", 2.0, 0.1, 4.5),
    ("b4_q002_c4p5", 4.0, 0.02, 4.5),
    ("b2_q03_c10", 2.0, 0.3, 10.0),
    ("b5_q03_c25", 5.0, 0.3, 25.0),
    ("b1_q04_c10", 1.0, 0.4, 10.0),
    ("b1_q04_c15", 1.0, 0.4, 15.0),
    ("b1_q04_c20", 1.0, 0.4, 20.0),
    ("b1_q04_c25", 1.0, 0.4, 25.0),
    ("b1_q04_c27", 1.0, 0.4, 27.0),
    ("b2_q04_c15", 2.0, 0.4, 15.0),
)


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("school.mat"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "beta_validation_screen_seed2026_iter3000",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-time", type=float, default=300.0)
    parser.add_argument("--consensus-tol", type=float, default=1e-4)
    parser.add_argument("--convergence-window", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--initial-step", type=float, default=0.1)
    parser.add_argument("--max-step", type=float, default=0.1)
    parser.add_argument("--beta-step-scale", type=float)
    parser.add_argument("--lower-rho", type=float, default=0.1)
    parser.add_argument("--lower-max-iter", type=int, default=20000)
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated configuration names; default runs all candidates",
    )
    parser.add_argument(
        "--methods",
        default="ldpm,ldpm-capped",
        help="comma-separated methods: ldpm and/or ldpm-capped",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_preprocess_school(args.data, args.seed)
    training = TaskLossOperator(data.train_a, data.train_b)
    validation = TaskLossOperator(data.validation_a, data.validation_b)
    problem = SchoolLDPMProblem(training, validation)
    requested = {value.strip() for value in args.only.split(",") if value.strip()}
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown_methods = sorted(set(methods) - {"ldpm", "ldpm-capped"})
    if not methods or unknown_methods:
        raise ValueError(f"unsupported methods: {unknown_methods}")
    configs = [
        config for config in DEFAULT_CONFIGS if not requested or config[0] in requested
    ]
    if requested != {config[0] for config in configs}:
        missing = sorted(requested - {config[0] for config in configs})
        raise ValueError(f"unknown beta configurations: {missing}")
    run_rows: List[Dict[str, object]] = []
    pair_rows: List[Dict[str, object]] = []

    for name, beta0, beta_power, beta_max in configs:
        first_cap_iteration = int(math.ceil((beta_max / beta0) ** (1.0 / beta_power)))
        states: Dict[str, Dict[str, np.ndarray]] = {}
        config_rows: Dict[str, Dict[str, object]] = {}
        for method in methods:
            config = LDPMConfig(
                max_iter=args.max_iter,
                min_iter=1,
                tol=args.tol,
                beta0=beta0,
                beta_power=beta_power,
                beta_max=beta_max,
                gamma=args.gamma,
                initial_step=args.initial_step,
                max_step=args.max_step,
                beta_step_scale=args.beta_step_scale,
                consensus_tol=args.consensus_tol,
                convergence_window=args.convergence_window,
                max_time=args.max_time,
                record_interval=100,
            )
            print(
                f"Running {name} {METHOD_LABELS[method]} "
                f"(beta0={beta0:g}, q={beta_power:g}, cap={beta_max:g})...",
                flush=True,
            )
            _, state, summary, _ = run_ldpm(problem, method, config)
            raw_w = state["W"]
            lambda_l1 = state["lambda_l1"]
            lambda_nuclear = float(state["lambda_nuclear"])
            feasible_w, lower = feasible_lower_solve(
                training,
                lambda_l1,
                lambda_nuclear,
                raw_w,
                rho=args.lower_rho,
                max_iter=args.lower_max_iter,
            )
            raw_objective = lower_objective(
                training, raw_w, lambda_l1, lambda_nuclear
            )
            feasibility = max(
                raw_objective - float(lower["lower_objective"]), 0.0
            ) / validation.n
            row: Dict[str, object] = {
                "config": name,
                "method": METHOD_LABELS[method],
                "method_key": method,
                "beta0": beta0,
                "beta_power": beta_power,
                "beta_max": beta_max,
                "first_cap_iteration": first_cap_iteration,
                "status": summary["status"],
                "iterations": summary["iterations"],
                "time": summary["time"],
                "x_lambda_stop": summary["x_lambda_stop"],
                "final_beta": summary["final_beta"],
                "cap_reached": summary["cap_reached"],
                "raw_validation_rmse": math.sqrt(validation.mse(raw_w)),
                "feasible_validation_rmse": math.sqrt(validation.mse(feasible_w)),
                "feasibility": feasibility,
                "h_norm": summary["h_norm"],
                "consensus_residual": summary["consensus_residual"],
                "lower_status": lower["status"],
                "lower_iterations": lower["iterations"],
            }
            run_rows.append(row)
            config_rows[method] = row
            states[method] = state
            np.savez_compressed(
                args.output_dir / f"{name}_{method.replace('-', '_')}_state.npz",
                **state,
            )
            print(
                f"  val_rmse={row['feasible_validation_rmse']:.6f} "
                f"raw_val={row['raw_validation_rmse']:.6f} "
                f"cap={row['cap_reached']} feasibility={feasibility:.3e}",
                flush=True,
            )

        if set(methods) == {"ldpm", "ldpm-capped"}:
            uncapped = config_rows["ldpm"]
            capped = config_rows["ldpm-capped"]
            w_difference = float(
                np.max(np.abs(states["ldpm"]["W"] - states["ldpm-capped"]["W"]))
            )
            lambda_difference = float(
                np.max(
                    np.abs(
                        states["ldpm"]["lambda_l1"]
                        - states["ldpm-capped"]["lambda_l1"]
                    )
                )
            )
            validation_values = [
                float(uncapped["feasible_validation_rmse"]),
                float(capped["feasible_validation_rmse"]),
            ]
            pair_rows.append(
                {
                    "config": name,
                    "beta0": beta0,
                    "beta_power": beta_power,
                    "beta_max": beta_max,
                    "first_cap_iteration": first_cap_iteration,
                    "cap_reached": bool(capped["cap_reached"]),
                    "ldpm_validation_rmse": validation_values[0],
                    "capped_validation_rmse": validation_values[1],
                    "mean_validation_rmse": float(np.mean(validation_values)),
                    "worst_validation_rmse": float(np.max(validation_values)),
                    "validation_rmse_difference": abs(validation_values[0] - validation_values[1]),
                    "final_W_max_abs_difference": w_difference,
                    "final_lambda_l1_max_abs_difference": lambda_difference,
                }
            )

    save_csv(args.output_dir / "all_runs.csv", run_rows)
    if pair_rows:
        eligible = [
            row
            for row in pair_rows
            if bool(row["cap_reached"])
            and float(row["final_W_max_abs_difference"]) >= 0.02
        ]
        save_csv(args.output_dir / "pair_summary.csv", pair_rows)
        selected = min(
            eligible if eligible else pair_rows,
            key=lambda row: float(row["mean_validation_rmse"]),
        )
        selected["selection_gate_satisfied"] = bool(eligible)
    else:
        selected = min(
            run_rows, key=lambda row: float(row["feasible_validation_rmse"])
        )
    (args.output_dir / "selected_validation_config.json").write_text(
        json.dumps(selected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "protocol.json").write_text(
        json.dumps(
            {
                "test_split_used": False,
                "selection_metric": "minimum pair mean feasible validation RMSE",
                "separation_gate": "cap_reached and final W max absolute difference >= 0.02",
                "seed": args.seed,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "methods": methods,
                "consensus_tol": args.consensus_tol,
                "convergence_window": args.convergence_window,
                "gamma": args.gamma,
                "initial_step": args.initial_step,
                "max_step": args.max_step,
                "beta_step_scale": args.beta_step_scale,
                "configs": [
                    {
                        "name": name,
                        "beta0": beta0,
                        "beta_power": beta_power,
                        "beta_max": beta_max,
                    }
                    for name, beta0, beta_power, beta_max in configs
                ],
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
