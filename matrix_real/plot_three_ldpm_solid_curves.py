#!/usr/bin/env python3
"""Plot the three LDPM error-time curves as two all-solid PNG figures."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DYNAMIC_DIR = RESULTS / "seed2026_tol1em4_b1_q04_cap27"
FIXED_DIR = RESULTS / "seed2026_tol1em4_fixed_beta4p5"
OUTPUT_DIR = RESULTS / "three_ldpm_solid_curves"

SERIES = (
    (
        "LDPM-CS",
        DYNAMIC_DIR / "error_time_curves.csv",
        "ldpm",
        "#0072B2",
    ),
    (
        "LDPM-CS-C",
        DYNAMIC_DIR / "error_time_curves.csv",
        "ldpm-capped",
        "#D55E00",
    ),
    (
        "LDPM-CS(fixed)",
        FIXED_DIR / "error_time_curves.csv",
        "ldpm",
        "#009E73",
    ),
)


def read_series(path: Path, method_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: List[Dict[str, str]] = [
            row for row in csv.DictReader(handle) if row["method_key"] == method_key
        ]
    return (
        np.asarray([float(row["time"]) for row in rows]),
        np.asarray([float(row["validation_rmse_best_so_far"]) for row in rows]),
        np.asarray([float(row["test_rmse_best_so_far"]) for row in rows]),
    )


def plot_metric(metric: str, output_name: str) -> None:
    cache = Path(tempfile.gettempdir()) / "school_experiment_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.6, 5.2))
    value_index = 1 if metric == "validation" else 2
    for label, path, method_key, color in SERIES:
        values = read_series(path, method_key)
        axis.plot(
            values[0],
            values[value_index],
            label=label,
            color=color,
            linestyle="-",
            linewidth=2.35,
        )

    axis.set_title("School")
    axis.set_xlabel("time")
    axis.set_ylabel("validation error" if metric == "validation" else "test error")
    axis.grid(True, color="#D0D0D0", linewidth=0.65, linestyle="-", alpha=0.7)
    axis.legend(frameon=False, loc="best")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / output_name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_metric("validation", "validation_rmse_vs_time.png")
    plot_metric("test", "test_rmse_vs_time.png")
    print(OUTPUT_DIR / "validation_rmse_vs_time.png")
    print(OUTPUT_DIR / "test_rmse_vs_time.png")


if __name__ == "__main__":
    main()
