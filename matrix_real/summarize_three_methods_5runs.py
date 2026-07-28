#!/usr/bin/env python3
"""Summarize and plot five paired School runs for three LDPM-CS variants."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "school_three_methods_5runs"
SEEDS = (2026, 2027, 2028, 2029, 2030)
METRICS = (
    "time",
    "validation_error",
    "test_error",
    "test_error_infeasibility",
    "validation_rmse",
    "test_rmse",
    "test_rmse_infeasibility",
    "feasibility",
)
SERIES = (
    ("LDPM-CS", OUTPUT_DIR, "dynamic", "ldpm", "#0072B2"),
    ("LDPM-CS-C", OUTPUT_DIR, "dynamic", "ldpm-capped", "#D55E00"),
    ("LDPM-CS(fixed)", OUTPUT_DIR, "fixed", "ldpm", "#009E73"),
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_run_metrics() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed in SEEDS:
        for label, source_dir, subdir, method_key, _ in SERIES:
            summary_path = source_dir / f"seed{seed}" / subdir / "summary.csv"
            matches = [
                row for row in read_csv(summary_path) if row["method_key"] == method_key
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one {method_key} row in {summary_path}, got {len(matches)}"
                )
            source = matches[0]
            rows.append(
                {
                    "seed": seed,
                    "method": label,
                    "time": float(source["time"]),
                    "validation_error": float(source["validation_error"]),
                    "test_error": float(source["test_error"]),
                    "test_error_infeasibility": float(
                        source["test_error_infeasibility"]
                    ),
                    "validation_rmse": float(source["validation_rmse"]),
                    "test_rmse": float(source["test_rmse"]),
                    "test_rmse_infeasibility": float(
                        source["test_rmse_infeasibility"]
                    ),
                    "feasibility": float(source["feasibility"]),
                    "status": source["status"],
                    "iterations": int(source["iterations"]),
                    "cap_reached": source["cap_reached"],
                    "lower_status": source["lower_status"],
                }
            )
    return rows


def summarize_run_metrics(
    run_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for label, _, _, _, _ in SERIES:
        method_rows = [row for row in run_rows if row["method"] == label]
        row: Dict[str, object] = {
            "method": label,
            "n": len(method_rows),
        }
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in method_rows])
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        row["successful_or_budget_complete"] = sum(
            str(item["status"]) in {"success", "max_iter"} for item in method_rows
        )
        row["lower_converged"] = sum(
            str(item["lower_status"]) == "converged" for item in method_rows
        )
        row["cap_reached_count"] = sum(
            str(item["cap_reached"]).lower() == "true" for item in method_rows
        )
        rows.append(row)
    return rows


def read_curve(
    seed: int, source_dir: Path, subdir: str, method_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = source_dir / f"seed{seed}" / subdir / "error_time_curves.csv"
    rows = [row for row in read_csv(path) if row["method_key"] == method_key]
    if not rows:
        raise ValueError(f"missing curve for {method_key} in {path}")
    return (
        np.asarray([float(row["time"]) for row in rows]),
        np.asarray(
            [float(row["validation_rmse_best_so_far"]) for row in rows]
        ),
        np.asarray([float(row["test_rmse_best_so_far"]) for row in rows]),
    )


def aggregate_curves() -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    all_curves: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    terminal_times: List[float] = []
    for label, source_dir, subdir, method_key, _ in SERIES:
        method_curves = [
            read_curve(seed, source_dir, subdir, method_key) for seed in SEEDS
        ]
        all_curves[label] = method_curves
        terminal_times.extend(float(curve[0][-1]) for curve in method_curves)

    # After a run satisfies its stopping rule, its returned iterate is unchanged.
    # Carry that terminal error forward so all five runs contribute on one common
    # wall-clock grid and the plot shows the post-convergence plateau explicitly.
    common_time = np.linspace(0.0, 1.1 * max(terminal_times), 421)
    aggregated: Dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for label, _, _, _, _ in SERIES:
        validation = np.stack(
            [
                np.interp(
                    common_time,
                    curve[0],
                    curve[1],
                    left=curve[1][0],
                    right=curve[1][-1],
                )
                for curve in all_curves[label]
            ]
        )
        test = np.stack(
            [
                np.interp(
                    common_time,
                    curve[0],
                    curve[2],
                    left=curve[2][0],
                    right=curve[2][-1],
                )
                for curve in all_curves[label]
            ]
        )
        aggregated[label] = (
            common_time,
            np.mean(validation, axis=0),
            np.std(validation, axis=0, ddof=1),
            np.mean(test, axis=0),
            np.std(test, axis=0, ddof=1),
        )
    return aggregated


def curve_rows(
    aggregated: Dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> Iterable[Dict[str, object]]:
    for label, _, _, _, _ in SERIES:
        time_values, validation_mean, validation_std, test_mean, test_std = (
            aggregated[label]
        )
        for time_value, val_mean, val_std, tst_mean, tst_std in zip(
            time_values, validation_mean, validation_std, test_mean, test_std
        ):
            yield {
                "method": label,
                "time": float(time_value),
                "validation_error_mean": float(val_mean),
                "validation_error_std": float(val_std),
                "test_error_mean": float(tst_mean),
                "test_error_std": float(tst_std),
                "n": len(SEEDS),
            }


def plot_curves(
    aggregated: Dict[
        str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ],
) -> None:
    cache = Path(tempfile.gettempdir()) / "school_experiment_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    for metric, output_name in (
        ("validation", "validation_error_vs_time_mean_std.png"),
        ("test", "test_error_vs_time_mean_std.png"),
    ):
        figure, axis = plt.subplots(figsize=(8.6, 5.2))
        for label, _, _, _, color in SERIES:
            time_values, validation_mean, validation_std, test_mean, test_std = (
                aggregated[label]
            )
            if metric == "validation":
                mean_values, std_values = validation_mean, validation_std
            else:
                mean_values, std_values = test_mean, test_std
            axis.fill_between(
                time_values,
                mean_values - std_values,
                mean_values + std_values,
                color=color,
                alpha=0.16,
                linewidth=0.0,
            )
            axis.plot(
                time_values,
                mean_values,
                label=label,
                color=color,
                linestyle="-",
                linewidth=2.35,
            )

        # Magnify the converged tail where capped and fixed nearly overlap.
        # Keep the inset free of uncertainty bands so their mean curves remain
        # distinguishable even when the standard-deviation bands overlap.
        focus_labels = ("LDPM-CS-C", "LDPM-CS(fixed)")
        common_time = aggregated[focus_labels[0]][0]
        zoom_start = 0.76 * float(common_time[-1])
        zoom_mask = common_time >= zoom_start
        zoom_values: List[float] = []
        for focus_label in focus_labels:
            curve = (
                aggregated[focus_label][1]
                if metric == "validation"
                else aggregated[focus_label][3]
            )
            zoom_values.extend(float(value) for value in curve[zoom_mask])
        zoom_low = min(zoom_values)
        zoom_high = max(zoom_values)
        zoom_pad = max(0.15 * (zoom_high - zoom_low), 1e-3)

        inset = axis.inset_axes([0.50, 0.46, 0.44, 0.30])
        for label, _, _, _, color in SERIES:
            time_values, validation_mean, _, test_mean, _ = aggregated[label]
            mean_values = validation_mean if metric == "validation" else test_mean
            inset.plot(
                time_values,
                mean_values,
                color=color,
                linestyle="-",
                linewidth=2.0,
            )
        inset.set_xlim(zoom_start, float(common_time[-1]))
        inset.set_ylim(zoom_low - zoom_pad, zoom_high + zoom_pad)
        inset.set_xticks(np.linspace(zoom_start, float(common_time[-1]), 3))
        inset.set_yticks(
            np.linspace(zoom_low - zoom_pad, zoom_high + zoom_pad, 3)
        )
        inset.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        inset.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        inset.tick_params(axis="both", labelsize=7)
        inset.grid(
            True,
            color="#D0D0D0",
            linewidth=0.5,
            linestyle="-",
            alpha=0.65,
        )
        axis.indicate_inset_zoom(
            inset,
            edgecolor="#666666",
            linewidth=0.8,
            alpha=0.65,
        )
        axis.set_title("School")
        axis.set_xlabel("time")
        axis.set_ylabel(
            "validation error" if metric == "validation" else "test error"
        )
        axis.grid(True, color="#D0D0D0", linewidth=0.65, linestyle="-", alpha=0.7)
        axis.legend(frameon=False, loc="best")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / output_name, dpi=220, bbox_inches="tight")
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing seed2026 through seed2030 subdirectories.",
    )
    parser.add_argument(
        "--fixed-source-dir",
        type=Path,
        default=None,
        help=(
            "Optional prior campaign containing seed*/fixed results. When set, "
            "the output directory is expected to contain seed*/original and "
            "seed*/capped results."
        ),
    )
    return parser.parse_args()


def main() -> None:
    global OUTPUT_DIR, SERIES
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    if args.fixed_source_dir is None:
        SERIES = (
            ("LDPM-CS", OUTPUT_DIR, "dynamic", "ldpm", "#0072B2"),
            ("LDPM-CS-C", OUTPUT_DIR, "dynamic", "ldpm-capped", "#D55E00"),
            ("LDPM-CS(fixed)", OUTPUT_DIR, "fixed", "ldpm", "#009E73"),
        )
    else:
        SERIES = (
            ("LDPM-CS", OUTPUT_DIR, "original", "ldpm", "#0072B2"),
            ("LDPM-CS-C", OUTPUT_DIR, "capped", "ldpm-capped", "#D55E00"),
            (
                "LDPM-CS(fixed)",
                args.fixed_source_dir,
                "fixed",
                "ldpm",
                "#009E73",
            ),
        )
    run_rows = collect_run_metrics()
    summary_rows = summarize_run_metrics(run_rows)
    aggregated = aggregate_curves()
    write_csv(OUTPUT_DIR / "per_run_metrics.csv", run_rows)
    write_csv(OUTPUT_DIR / "summary_statistics.csv", summary_rows)
    write_csv(OUTPUT_DIR / "mean_std_error_time_curves.csv", list(curve_rows(aggregated)))
    plot_curves(aggregated)
    print(OUTPUT_DIR / "summary_statistics.csv")
    print(OUTPUT_DIR / "validation_error_vs_time_mean_std.png")
    print(OUTPUT_DIR / "test_error_vs_time_mean_std.png")


if __name__ == "__main__":
    main()
