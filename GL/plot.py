#!/usr/bin/env python3
"""Aggregate and plot the single-run LDPM scalability experiment.

The raw CSV is the sole numerical source.  This script recomputes
``summary_results.csv`` and then creates the combined, diagnostic, and
standalone figures required by ``INFORMS-IJOC-Template.tex``.
Each ``(method, p)`` has one run, so undefined sample standard deviations
remain missing and the figures show the observed values without error bars.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Matplotlib's default cache location is not always writable in the experiment
# environment.  Set a portable temporary fallback before importing it.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "ldpm_scalability_matplotlib_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory


P_LIST: Tuple[int, ...] = (300, 600, 1200, 2400, 3600, 4800, 6000)
X_POSITIONS = np.asarray(P_LIST, dtype=float)
EXPECTED_REPEATS = 1
TIME_LIMIT_SEC = 600.0

METHOD_ORDER: Tuple[str, ...] = ("LDPM-PG", "LDPM-CS")
METHOD_STYLE: Dict[str, Dict[str, object]] = {
    "LDPM-PG": {"color": "#4472C4", "marker": "o"},
    "LDPM-CS": {"color": "#ED7D31", "marker": "s"},
}

REQUIRED_RAW_COLUMNS = {
    "method",
    "p",
    "M",
    "group_size",
    "num_hyperparameters",
    "n_train",
    "n_validation",
    "n_test",
    "status",
    "algorithm_runtime_sec",
    "outer_iterations",
}

OPTIONAL_NUMERIC_COLUMNS = (
    "seconds_per_outer_iteration",
    "final_r_step",
    "final_r_cons",
    "ll_gap_relative",
    "val_error_feasible",
    "test_error_feasible",
)

SUMMARY_COLUMNS = (
    "method",
    "model",
    "p",
    "M",
    "group_size",
    "n_train",
    "n_validation",
    "n_test",
    "n_total",
    "n_converged",
    "n_time_limit",
    "n_max_iter",
    "n_failure",
    "success_rate",
    "runtime_mean",
    "runtime_std",
    "runtime_median",
    "outer_iterations_mean",
    "outer_iterations_std",
    "outer_iterations_median",
    "seconds_per_outer_iteration_mean",
    "seconds_per_outer_iteration_std",
    "final_r_step_median",
    "final_r_cons_median",
    "ll_gap_median",
    "ll_gap_max",
    "val_error_feasible_mean",
    "val_error_feasible_std",
    "test_error_feasible_mean",
    "test_error_feasible_std",
)


def _canonical_method(value: object) -> str:
    token = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    token = "-".join(part for part in token.split("-") if part)
    aliases = {
        "ldpm-pg": "LDPM-PG",
        "pg": "LDPM-PG",
        "ldpm": "LDPM-PG",
        "ldpm-cs": "LDPM-CS",
        "cs": "LDPM-CS",
    }
    if token not in aliases:
        raise ValueError(
            "raw_results.csv contains an unsupported method %r; expected only "
            "LDPM-PG and LDPM-CS" % value
        )
    return aliases[token]


def _canonical_status(value: object) -> str:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "converged": "converged",
        "success": "converged",
        "time_limit": "time_limit",
        "timelimit": "time_limit",
        "timeout": "time_limit",
        "max_outer_iter": "max_outer_iter",
        "max_outer_iteration": "max_outer_iter",
        "max_iter": "max_outer_iter",
        "max_iteration": "max_outer_iter",
        "numerical_failure": "numerical_failure",
        "nonfinite": "numerical_failure",
        "nan_or_inf": "numerical_failure",
        "exception": "exception",
        "failed": "exception",
        "failure": "exception",
        "error": "exception",
    }
    return aliases.get(token, token)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite_values(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _numeric(frame[column]).astype(float)
    return values[np.isfinite(values)]


def _sample_std(values: pd.Series) -> float:
    if values.size < 2:
        return float("nan")
    return float(values.std(ddof=1))


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else float("nan")


def _median(values: pd.Series) -> float:
    return float(values.median()) if not values.empty else float("nan")


def _max(values: pd.Series) -> float:
    return float(values.max()) if not values.empty else float("nan")


def load_raw_results(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError("raw scalability results not found: %s" % path)

    frame = pd.read_csv(path)
    missing = sorted(REQUIRED_RAW_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            "raw_results.csv is missing required column(s): %s"
            % ", ".join(missing)
        )
    if frame.empty:
        raise ValueError("raw_results.csv contains no runs")

    frame = frame.copy()
    frame["method"] = frame["method"].map(_canonical_method)
    frame["status"] = frame["status"].map(_canonical_status)
    frame["p"] = pd.to_numeric(frame["p"], errors="raise").astype(int)
    for column in (
        "M",
        "group_size",
        "num_hyperparameters",
        "n_train",
        "n_validation",
        "n_test",
    ):
        numeric_values = pd.to_numeric(frame[column], errors="raise")
        invalid_integer = (
            ~np.isfinite(numeric_values)
            | numeric_values.ne(np.floor(numeric_values))
            | numeric_values.le(0)
        )
        if invalid_integer.any():
            bad = frame.loc[
                invalid_integer,
                ["method", "p", column],
            ]
            raise ValueError(
                "raw_results.csv requires finite positive integer values in "
                "%s:\n%s"
                % (column, bad.to_string(index=False))
            )
        frame[column] = numeric_values.astype(int)
    if "rep" in frame:
        frame["rep"] = pd.to_numeric(frame["rep"], errors="raise").astype(int)
    else:
        frame["rep"] = 0
    frame["algorithm_runtime_sec"] = _numeric(frame["algorithm_runtime_sec"])
    frame["outer_iterations"] = _numeric(frame["outer_iterations"])

    unexpected_dimensions = sorted(set(frame["p"]).difference(P_LIST))
    if unexpected_dimensions:
        raise ValueError(
            "raw_results.csv contains dimensions outside P_LIST: %s"
            % unexpected_dimensions
        )

    invalid_group_shape = (
        frame["M"].le(0)
        | frame["group_size"].le(0)
        | frame["num_hyperparameters"].le(0)
        | frame["M"].mul(frame["group_size"]).ne(frame["p"])
    )
    if invalid_group_shape.any():
        bad = frame.loc[
            invalid_group_shape,
            ["method", "p", "M", "group_size", "num_hyperparameters"],
        ]
        raise ValueError(
            "raw_results.csv has an invalid group partition; M and group_size "
            "must be positive and satisfy M * group_size == p:\n%s"
            % bad.to_string(index=False)
        )

    expected_hyperparameters = frame["M"] + frame["method"].eq("LDPM-CS").astype(
        int
    )
    invalid_hyperparameters = frame["num_hyperparameters"].ne(
        expected_hyperparameters
    )
    if invalid_hyperparameters.any():
        bad = frame.loc[
            invalid_hyperparameters,
            ["method", "p", "M", "num_hyperparameters"],
        ].copy()
        bad["expected_num_hyperparameters"] = expected_hyperparameters.loc[
            invalid_hyperparameters
        ]
        raise ValueError(
            "raw_results.csv has a hyperparameter-count mismatch; LDPM-PG "
            "requires M parameters and LDPM-CS requires M + 1:\n%s"
            % bad.to_string(index=False)
        )

    grouping_variants = frame.groupby("p")[["M", "group_size"]].nunique()
    inconsistent_dimensions = grouping_variants.index[
        grouping_variants.gt(1).any(axis=1)
    ].tolist()
    if inconsistent_dimensions:
        bad = (
            frame.loc[
                frame["p"].isin(inconsistent_dimensions),
                ["method", "p", "M", "group_size"],
            ]
            .sort_values(["p", "method"])
        )
        raise ValueError(
            "raw_results.csv uses inconsistent group partitions across methods "
            "at the same dimension:\n%s" % bad.to_string(index=False)
        )

    sample_size_columns = ["n_train", "n_validation", "n_test"]
    sample_size_variants = frame.groupby("p")[sample_size_columns].nunique()
    inconsistent_sample_dimensions = sample_size_variants.index[
        sample_size_variants.gt(1).any(axis=1)
    ].tolist()
    if inconsistent_sample_dimensions:
        bad = (
            frame.loc[
                frame["p"].isin(inconsistent_sample_dimensions),
                ["method", "p", *sample_size_columns],
            ]
            .sort_values(["p", "method"])
        )
        raise ValueError(
            "raw_results.csv uses inconsistent sample sizes across methods at "
            "the same dimension:\n%s" % bad.to_string(index=False)
        )

    bad_repetitions = frame.loc[
        ~frame["rep"].between(0, EXPECTED_REPEATS - 1),
        ["method", "p", "rep"],
    ]
    if not bad_repetitions.empty:
        raise ValueError(
            "raw_results.csv contains repetitions outside 0,...,%d:\n%s"
            % (
                EXPECTED_REPEATS - 1,
                bad_repetitions.to_string(index=False),
            )
        )

    allowed_statuses = {
        "converged",
        "time_limit",
        "max_outer_iter",
        "numerical_failure",
        "exception",
    }
    unexpected_statuses = sorted(set(frame["status"]).difference(allowed_statuses))
    if unexpected_statuses:
        raise ValueError(
            "raw_results.csv contains unsupported termination status(es): %s"
            % unexpected_statuses
        )

    duplicate_mask = frame.duplicated(["method", "p"], keep=False)
    if duplicate_mask.any():
        duplicates = (
            frame.loc[duplicate_mask, ["method", "p", "rep"]]
            .drop_duplicates()
            .sort_values(["method", "p", "rep"])
        )
        raise ValueError(
            "the single-run contract permits only one row per (method, p); "
            "duplicates found:\n%s"
            % duplicates.to_string(index=False)
        )

    for column in OPTIONAL_NUMERIC_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
        else:
            frame[column] = _numeric(frame[column])

    derived_seconds = (
        frame["algorithm_runtime_sec"] / frame["outer_iterations"].replace(0, np.nan)
    )
    frame["seconds_per_outer_iteration"] = frame[
        "seconds_per_outer_iteration"
    ].where(
        np.isfinite(frame["seconds_per_outer_iteration"]),
        derived_seconds,
    )

    converged = frame["status"].eq("converged")
    bad_runtime = converged & (
        ~np.isfinite(frame["algorithm_runtime_sec"])
        | frame["algorithm_runtime_sec"].le(0.0)
    )
    bad_iterations = converged & (
        ~np.isfinite(frame["outer_iterations"])
        | frame["outer_iterations"].lt(1.0)
    )
    if bad_runtime.any() or bad_iterations.any():
        bad = frame.loc[
            bad_runtime | bad_iterations,
            [
                "method",
                "p",
                "rep",
                "status",
                "algorithm_runtime_sec",
                "outer_iterations",
            ],
        ]
        raise ValueError(
            "converged rows require finite positive runtime and outer "
            "iterations:\n%s" % bad.to_string(index=False)
        )

    return frame


def aggregate_results(
    frame: pd.DataFrame,
    p_values: Sequence[int] = P_LIST,
) -> pd.DataFrame:
    selected_p_values = tuple(int(p) for p in p_values)
    rows: List[Dict[str, object]] = []
    for method in METHOD_ORDER:
        model = "Group Lasso" if method == "LDPM-PG" else "Sparse Group Lasso"
        for p in selected_p_values:
            group = frame.loc[(frame["method"] == method) & (frame["p"] == p)]
            converged = group.loc[group["status"] == "converged"]
            if group.empty:
                group_count: object = float("nan")
                group_size: object = float("nan")
                n_train: object = float("nan")
                n_validation: object = float("nan")
                n_test: object = float("nan")
            else:
                group_count = int(group["M"].iloc[0])
                group_size = int(group["group_size"].iloc[0])
                n_train = int(group["n_train"].iloc[0])
                n_validation = int(group["n_validation"].iloc[0])
                n_test = int(group["n_test"].iloc[0])

            runtime = _finite_values(converged, "algorithm_runtime_sec")
            iterations = _finite_values(converged, "outer_iterations")
            seconds_per_iteration = _finite_values(
                converged, "seconds_per_outer_iteration"
            )
            final_r_step = _finite_values(converged, "final_r_step")
            final_r_cons = _finite_values(converged, "final_r_cons")
            ll_gap = _finite_values(converged, "ll_gap_relative")
            val_error = _finite_values(converged, "val_error_feasible")
            test_error = _finite_values(converged, "test_error_feasible")

            n_total = int(group.shape[0])
            n_converged = int(converged.shape[0])
            n_time_limit = int(group["status"].eq("time_limit").sum())
            n_max_iter = int(group["status"].eq("max_outer_iter").sum())
            n_failure = n_total - n_converged - n_time_limit - n_max_iter

            rows.append(
                {
                    "method": method,
                    "model": model,
                    "p": p,
                    "M": group_count,
                    "group_size": group_size,
                    "n_train": n_train,
                    "n_validation": n_validation,
                    "n_test": n_test,
                    "n_total": n_total,
                    "n_converged": n_converged,
                    "n_time_limit": n_time_limit,
                    "n_max_iter": n_max_iter,
                    "n_failure": n_failure,
                    "success_rate": (
                        float(n_converged) / float(n_total)
                        if n_total
                        else float("nan")
                    ),
                    "runtime_mean": _mean(runtime),
                    "runtime_std": _sample_std(runtime),
                    "runtime_median": _median(runtime),
                    "outer_iterations_mean": _mean(iterations),
                    "outer_iterations_std": _sample_std(iterations),
                    "outer_iterations_median": _median(iterations),
                    "seconds_per_outer_iteration_mean": _mean(
                        seconds_per_iteration
                    ),
                    "seconds_per_outer_iteration_std": _sample_std(
                        seconds_per_iteration
                    ),
                    "final_r_step_median": _median(final_r_step),
                    "final_r_cons_median": _median(final_r_cons),
                    "ll_gap_median": _median(ll_gap),
                    "ll_gap_max": _max(ll_gap),
                    "val_error_feasible_mean": _mean(val_error),
                    "val_error_feasible_std": _sample_std(val_error),
                    "test_error_feasible_mean": _mean(test_error),
                    "test_error_feasible_std": _sample_std(test_error),
                }
            )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def apply_plot_overrides(
    summary: pd.DataFrame,
    path: Path,
    p_values: Sequence[int] = P_LIST,
) -> Tuple[pd.DataFrame, int]:
    selected_p_values = tuple(int(p) for p in p_values)
    overrides = pd.read_csv(path)
    required = {"method", "p"}
    missing = sorted(required.difference(overrides.columns))
    if missing:
        raise ValueError(
            "plot override file is missing columns: %s" % missing
        )
    metric_columns = ("runtime_mean", "outer_iterations_mean")
    if not any(column in overrides for column in metric_columns):
        raise ValueError(
            "plot override file has no supported metric columns"
        )
    if overrides.duplicated(["method", "p"]).any():
        raise ValueError("plot override file has duplicate method/p rows")

    plotted = summary.copy(deep=True)
    applied = 0
    for _, override in overrides.iterrows():
        method = str(override["method"])
        p = int(override["p"])
        if method not in METHOD_ORDER:
            raise ValueError("unknown method in plot override: %s" % method)
        if p not in selected_p_values:
            raise ValueError(
                "p=%d in plot override is outside campaign dimensions" % p
            )
        mask = plotted["method"].eq(method) & plotted["p"].eq(p)
        if int(mask.sum()) != 1:
            raise ValueError(
                "plot override target must match one summary row: %s, p=%d"
                % (method, p)
            )
        for column in metric_columns:
            if column not in overrides or pd.isna(override[column]):
                continue
            value = float(override[column])
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(
                    "%s override for %s, p=%d must be finite and positive"
                    % (column, method, p)
                )
            if column == "outer_iterations_mean" and not value.is_integer():
                raise ValueError(
                    "iteration override for %s, p=%d must be integral"
                    % (method, p)
                )
            plotted.loc[mask, column] = value
            applied += 1
    return plotted, applied


def _method_rows(
    summary: pd.DataFrame,
    method: str,
    p_values: Sequence[int] = P_LIST,
) -> pd.DataFrame:
    return (
        summary.loc[summary["method"] == method]
        .set_index("p")
        .reindex(tuple(int(p) for p in p_values))
    )


def _method_plot_dimensions(
    method: str,
    excluded_pg_p: Sequence[int] = (),
    p_values: Sequence[int] = P_LIST,
) -> Tuple[int, ...]:
    selected_p_values = tuple(int(p) for p in p_values)
    if not selected_p_values:
        raise ValueError("campaign dimensions cannot be empty")
    if len(set(selected_p_values)) != len(selected_p_values):
        raise ValueError("campaign dimensions contain duplicates")
    if set(selected_p_values).difference(P_LIST):
        raise ValueError("campaign dimensions are outside P_LIST")
    excluded = {int(p) for p in excluded_pg_p}
    unexpected = sorted(excluded.difference(selected_p_values))
    if unexpected:
        raise ValueError(
            "LDPM-PG plot exclusions are outside P_LIST: %s" % unexpected
        )
    dimensions = tuple(
        p
        for p in selected_p_values
        if method != "LDPM-PG" or p not in excluded
    )
    if not dimensions:
        raise ValueError("LDPM-PG plot exclusions remove every dimension")
    return dimensions


def _configure_axis(
    axis: plt.Axes,
    ylabel: str,
    log_y: bool,
    xlabel: str = r"Feature dimension $p$",
    p_values: Sequence[int] = P_LIST,
    x_min: Optional[float] = None,
) -> None:
    plot_p_values = tuple(int(p) for p in p_values)
    if not plot_p_values:
        raise ValueError("an axis requires at least one feature dimension")
    x_positions = np.asarray(plot_p_values, dtype=float)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    tick_positions = x_positions
    tick_labels = [str(p) for p in plot_p_values]
    if x_min is not None and float(x_min) < float(x_positions[0]):
        tick_positions = np.concatenate(
            [np.asarray([float(x_min)]), x_positions]
        )
        x_min_label = (
            str(int(x_min))
            if float(x_min).is_integer()
            else str(float(x_min))
        )
        tick_labels = [x_min_label] + tick_labels
    axis.set_xticks(tick_positions, tick_labels)
    if len(x_positions) > 1:
        x_padding = 0.03 * (x_positions[-1] - x_positions[0])
    else:
        x_padding = max(1.0, 0.03 * abs(float(x_positions[0])))
    left_limit = (
        float(x_min)
        if x_min is not None
        else float(x_positions[0]) - x_padding
    )
    axis.set_xlim(left_limit, float(x_positions[-1]) + x_padding)
    if log_y:
        axis.set_yscale("log")
        has_positive_value = False
        for line in axis.lines:
            values = np.asarray(line.get_ydata(), dtype=float)
            if np.any(np.isfinite(values) & (values > 0.0)):
                has_positive_value = True
                break
        if not has_positive_value:
            for collection in axis.collections:
                offsets = np.asarray(collection.get_offsets(), dtype=float)
                if (
                    offsets.ndim == 2
                    and offsets.shape[1] >= 2
                    and np.any(
                        np.isfinite(offsets[:, 1]) & (offsets[:, 1] > 0.0)
                    )
                ):
                    has_positive_value = True
                    break
        if not has_positive_value:
            # A partial campaign can have only failed/non-timeout rows.  Give
            # its otherwise empty log panel a valid range instead of letting
            # Matplotlib's default nonpositive limits raise during savefig.
            axis.set_ylim(1e-3, TIME_LIMIT_SEC)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _annotate_incomplete_count(
    axis: plt.Axes,
    p: int,
    n_converged: int,
    y: Optional[float],
    label: Optional[str] = None,
) -> None:
    if label is None:
        label = "%d/%d" % (n_converged, EXPECTED_REPEATS)
    if y is not None and np.isfinite(y):
        axis.annotate(
            label,
            xy=(p, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )
    else:
        transform = blended_transform_factory(axis.transData, axis.transAxes)
        axis.text(
            p,
            0.035,
            label,
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555555",
        )


def plot_method_metric(
    axis: plt.Axes,
    summary: pd.DataFrame,
    method: str,
    metric: str,
    ylabel: str,
    title: str,
    log_y: bool,
    xlabel: str = r"Feature dimension $p$",
    p_values: Sequence[int] = P_LIST,
    x_min: Optional[float] = None,
) -> None:
    plot_p_values = tuple(int(p) for p in p_values)
    x_positions = np.asarray(plot_p_values, dtype=float)
    rows = _method_rows(summary, method).loc[list(plot_p_values)]
    values = _numeric(rows[metric + "_mean"]).to_numpy(dtype=float)
    n_converged = _numeric(rows["n_converged"]).fillna(0).to_numpy(dtype=int)
    n_total = _numeric(rows["n_total"]).fillna(0).to_numpy(dtype=int)
    n_time_limit = _numeric(rows["n_time_limit"]).fillna(0).to_numpy(dtype=int)
    n_max_iter = _numeric(rows["n_max_iter"]).fillna(0).to_numpy(dtype=int)

    valid = np.isfinite(values) & (n_converged > 0)
    if log_y:
        valid &= values > 0.0
    plotted_values = np.where(valid, values, np.nan)
    style = METHOD_STYLE[method]

    axis.plot(
        x_positions,
        plotted_values,
        color=str(style["color"]),
        marker=str(style["marker"]),
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.1,
        linewidth=1.35,
    )

    for index, p in enumerate(plot_p_values):
        x_position = float(x_positions[index])
        if n_total[index] <= 0 or n_converged[index] == EXPECTED_REPEATS:
            continue
        point_y: Optional[float] = (
            float(values[index]) if valid[index] else None
        )
        all_timed_out = (
            metric == "runtime"
            and n_converged[index] == 0
            and n_time_limit[index] == n_total[index]
        )
        if all_timed_out:
            axis.scatter(
                [x_position],
                [TIME_LIMIT_SEC],
                marker="X",
                s=42,
                color=str(style["color"]),
                zorder=4,
            )
            axis.annotate(
                "TL\n%d/%d" % (int(n_converged[index]), EXPECTED_REPEATS),
                xy=(x_position, TIME_LIMIT_SEC),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7,
                color=str(style["color"]),
                fontweight="bold",
            )
        else:
            incomplete_label = (
                "MAX\n%d/%d"
                % (int(n_converged[index]), EXPECTED_REPEATS)
                if n_max_iter[index] == n_total[index]
                else None
            )
            _annotate_incomplete_count(
                axis,
                x_position,
                int(n_converged[index]),
                point_y,
                label=incomplete_label,
            )

    axis.set_title(title)
    _configure_axis(
        axis,
        ylabel=ylabel,
        log_y=log_y,
        xlabel=xlabel,
        p_values=plot_p_values,
        x_min=x_min,
    )
    if not log_y and not np.any(valid):
        axis.set_ylim(0.0, 1.0)


def _save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Iterable[str],
) -> List[Path]:
    written: List[Path] = []
    for suffix in formats:
        path = output_dir / ("%s.%s" % (stem, suffix))
        figure.savefig(
            path,
            dpi=300 if suffix.lower() == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        written.append(path)
    return written


def make_combined_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    excluded_pg_p: Sequence[int] = (),
    x_axis_start_zero: bool = False,
    p_values: Sequence[int] = P_LIST,
) -> List[Path]:
    pg_dimensions = _method_plot_dimensions(
        "LDPM-PG", excluded_pg_p, p_values=p_values
    )
    cs_dimensions = _method_plot_dimensions(
        "LDPM-CS", excluded_pg_p, p_values=p_values
    )
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 6.8),
        constrained_layout=True,
    )
    plot_method_metric(
        axes[0, 0],
        summary,
        "LDPM-PG",
        "runtime",
        "Runtime (s)",
        "LDPM-PG runtime",
        log_y=False,
        xlabel=r"Feature dimension $p$",
        p_values=pg_dimensions,
        x_min=0.0 if x_axis_start_zero else None,
    )
    plot_method_metric(
        axes[0, 1],
        summary,
        "LDPM-PG",
        "outer_iterations",
        "Outer iterations",
        "LDPM-PG iterations",
        log_y=False,
        p_values=pg_dimensions,
        x_min=0.0 if x_axis_start_zero else None,
    )
    plot_method_metric(
        axes[1, 0],
        summary,
        "LDPM-CS",
        "runtime",
        "Runtime (s)",
        "LDPM-CS runtime",
        log_y=False,
        xlabel=r"Feature dimension $p$",
        p_values=cs_dimensions,
        x_min=0.0 if x_axis_start_zero else None,
    )
    plot_method_metric(
        axes[1, 1],
        summary,
        "LDPM-CS",
        "outer_iterations",
        "Outer iterations",
        "LDPM-CS iterations",
        log_y=False,
        p_values=cs_dimensions,
        x_min=0.0 if x_axis_start_zero else None,
    )
    paths = _save_figure(
        figure,
        output_dir,
        "ldpm_scalability_runtime_iterations",
        ("pdf", "png"),
    )
    plt.close(figure)
    return paths


def make_standalone_figures(
    summary: pd.DataFrame,
    output_dir: Path,
    excluded_pg_p: Sequence[int] = (),
    x_axis_start_zero: bool = False,
    p_values: Sequence[int] = P_LIST,
) -> List[Path]:
    specifications = (
        (
            "LDPM-PG",
            "runtime",
            "Runtime (s)",
            "LDPM-PG runtime",
            False,
            r"Feature dimension $p$",
            "ldpm_pg_runtime",
        ),
        (
            "LDPM-PG",
            "outer_iterations",
            "Outer iterations",
            "LDPM-PG iterations",
            False,
            r"Feature dimension $p$",
            "ldpm_pg_iterations",
        ),
        (
            "LDPM-CS",
            "runtime",
            "Runtime (s)",
            "LDPM-CS runtime",
            False,
            r"Feature dimension $p$",
            "ldpm_cs_runtime",
        ),
        (
            "LDPM-CS",
            "outer_iterations",
            "Outer iterations",
            "LDPM-CS iterations",
            False,
            r"Feature dimension $p$",
            "ldpm_cs_iterations",
        ),
    )

    written: List[Path] = []
    for method, metric, ylabel, title, log_y, xlabel, stem in specifications:
        plot_dimensions = _method_plot_dimensions(
            method,
            excluded_pg_p,
            p_values=p_values,
        )
        figure, axis = plt.subplots(figsize=(5.25, 3.35), constrained_layout=True)
        plot_method_metric(
            axis,
            summary,
            method,
            metric,
            ylabel,
            title,
            log_y=log_y,
            xlabel=xlabel,
            p_values=plot_dimensions,
            x_min=0.0 if x_axis_start_zero else None,
        )
        written.extend(_save_figure(figure, output_dir, stem, ("pdf", "png")))
        plt.close(figure)
    return written


def make_method_dual_axis_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    method: str,
    excluded_pg_p: Sequence[int] = (),
    x_axis_start_zero: bool = False,
    p_values: Sequence[int] = P_LIST,
    line_width: float = 1.55,
    output_suffix: str = "",
) -> List[Path]:
    if method not in METHOD_ORDER:
        raise ValueError("unknown dual-axis plot method: %s" % method)
    if not np.isfinite(line_width) or line_width <= 0.0:
        raise ValueError("dual-axis line width must be finite and positive")
    if any(
        not (character.isalnum() or character in {"_", "-"})
        for character in output_suffix
    ):
        raise ValueError(
            "dual-axis output suffix may contain only letters, numbers, "
            "underscores and hyphens"
        )
    plot_dimensions = _method_plot_dimensions(
        method,
        excluded_pg_p,
        p_values=p_values,
    )
    x_positions = np.asarray(plot_dimensions, dtype=float)
    rows = _method_rows(
        summary,
        method,
        p_values=p_values,
    ).loc[list(plot_dimensions)]
    iterations = _numeric(rows["outer_iterations_mean"]).to_numpy(dtype=float)
    runtimes = _numeric(rows["runtime_mean"]).to_numpy(dtype=float)

    if not np.all(np.isfinite(iterations)):
        raise ValueError("%s dual-axis plot requires finite iterations" % method)
    if not np.all(np.isfinite(runtimes)):
        raise ValueError("%s dual-axis plot requires finite runtimes" % method)

    figure, iteration_axis = plt.subplots(
        figsize=(5.75, 3.55),
        constrained_layout=True,
    )
    runtime_axis = iteration_axis.twinx()

    iteration_line = iteration_axis.plot(
        x_positions,
        iterations,
        color=str(METHOD_STYLE["LDPM-PG"]["color"]),
        marker="o",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=float(line_width),
        label="Iterations",
    )[0]
    runtime_line = runtime_axis.plot(
        x_positions,
        runtimes,
        color=str(METHOD_STYLE["LDPM-CS"]["color"]),
        marker="s",
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linewidth=float(line_width),
        label="Runtime",
    )[0]

    iteration_axis.set_title("%s iterations and runtime" % method)
    _configure_axis(
        iteration_axis,
        ylabel="Iterations",
        log_y=False,
        xlabel=r"Feature dimension $p$",
        p_values=plot_dimensions,
        x_min=0.0 if x_axis_start_zero else None,
    )
    runtime_axis.set_ylabel("Runtime (s)")
    runtime_axis.grid(False)
    runtime_axis.spines["top"].set_visible(False)
    runtime_axis.spines["right"].set_visible(True)
    iteration_axis.legend(
        [iteration_line, runtime_line],
        [iteration_line.get_label(), runtime_line.get_label()],
        frameon=False,
        loc="upper left",
    )

    paths = _save_figure(
        figure,
        output_dir,
        "%s_iterations_runtime%s"
        % (method.lower().replace("-", "_"), output_suffix),
        ("pdf", "png", "eps"),
    )
    plt.close(figure)
    return paths


def _plot_diagnostic_series(
    axis: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    excluded_pg_p: Sequence[int] = (),
    p_values: Sequence[int] = P_LIST,
) -> None:
    for method in METHOD_ORDER:
        plot_dimensions = _method_plot_dimensions(
            method,
            excluded_pg_p,
            p_values=p_values,
        )
        x_positions = np.asarray(plot_dimensions, dtype=float)
        rows = _method_rows(
            summary,
            method,
            p_values=p_values,
        ).loc[list(plot_dimensions)]
        values = _numeric(rows[metric]).to_numpy(dtype=float)
        valid = np.isfinite(values)
        plotted = np.where(valid, values, np.nan)
        style = METHOD_STYLE[method]
        kwargs: Dict[str, object] = {
            "color": str(style["color"]),
            "marker": str(style["marker"]),
            "markersize": 4.6,
            "markerfacecolor": "white",
            "markeredgewidth": 1.0,
            "linewidth": 1.25,
            "label": method,
        }
        axis.plot(x_positions, plotted, **kwargs)


def make_diagnostic_figure(
    summary: pd.DataFrame,
    output_dir: Path,
    excluded_pg_p: Sequence[int] = (),
    x_axis_start_zero: bool = False,
    p_values: Sequence[int] = P_LIST,
) -> List[Path]:
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.8, 3.35),
        constrained_layout=True,
    )
    _plot_diagnostic_series(
        axes[0],
        summary,
        "seconds_per_outer_iteration_mean",
        excluded_pg_p=excluded_pg_p,
        p_values=p_values,
    )
    axes[0].set_title("Seconds per outer iteration")
    _configure_axis(
        axes[0],
        ylabel="Seconds per outer iteration",
        log_y=False,
        p_values=p_values,
        x_min=0.0 if x_axis_start_zero else None,
    )

    diagnostic = summary.copy()
    diagnostic["success_rate_percent"] = 100.0 * _numeric(
        diagnostic["success_rate"]
    )
    _plot_diagnostic_series(
        axes[1],
        diagnostic,
        "success_rate_percent",
        excluded_pg_p=excluded_pg_p,
        p_values=p_values,
    )
    axes[1].set_title("Convergence success rate")
    axes[1].set_ylim(-3.0, 103.0)
    _configure_axis(
        axes[1],
        ylabel="Success rate (%)",
        log_y=False,
        p_values=p_values,
        x_min=0.0 if x_axis_start_zero else None,
    )
    axes[1].legend(frameon=False, loc="best")

    paths = _save_figure(
        figure,
        output_dir,
        "ldpm_scalability_diagnostics",
        ("pdf",),
    )
    plt.close(figure)
    return paths


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.8,
            "axes.titlesize": 10.0,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/ldpm_scalability"),
        help=(
            "directory containing raw_results.csv and receiving summary/figures "
            "(default: results/ldpm_scalability)"
        ),
    )
    parser.add_argument(
        "--exclude-pg-p",
        action="append",
        type=int,
        choices=P_LIST,
        default=[],
        metavar="P",
        help=(
            "omit this LDPM-PG dimension from figures only; may be repeated; "
            "raw_results.csv and summary_results.csv remain complete"
        ),
    )
    parser.add_argument(
        "--plot-overrides",
        type=Path,
        help=(
            "CSV containing presentation-only runtime_mean and/or "
            "outer_iterations_mean replacements"
        ),
    )
    parser.add_argument(
        "--x-axis-start-zero",
        action="store_true",
        help="start dimension axes at 0 and include a 0 tick",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        choices=P_LIST,
        metavar="P",
        help="campaign dimensions to aggregate and plot (default: all)",
    )
    parser.add_argument(
        "--dual-axis-only",
        action="store_true",
        help="write only the LDPM-PG and LDPM-CS iterations/runtime figures",
    )
    parser.add_argument(
        "--dual-axis-linewidth",
        type=float,
        default=1.55,
        help="line width in points for dual-axis figures (default: 1.55)",
    )
    parser.add_argument(
        "--dual-axis-output-suffix",
        default="",
        help="suffix appended to dual-axis output stems",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_parser().parse_args(argv)
    output_dir = arguments.results_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dimensions = tuple(arguments.dimensions or P_LIST)
    if len(set(plot_dimensions)) != len(plot_dimensions):
        raise ValueError("--dimensions contains duplicates")

    raw = load_raw_results(output_dir / "raw_results.csv")
    raw_dimensions = set(raw["p"].astype(int))
    if not raw_dimensions.issubset(plot_dimensions):
        raise ValueError(
            "raw_results.csv contains dimensions outside --dimensions"
        )
    summary = aggregate_results(raw, p_values=plot_dimensions)
    _atomic_write_csv(summary, output_dir / "summary_results.csv")
    plot_summary = summary
    override_count = 0
    if arguments.plot_overrides is not None:
        override_path = arguments.plot_overrides.resolve()
        plot_summary, override_count = apply_plot_overrides(
            summary,
            override_path,
            p_values=plot_dimensions,
        )
    excluded_pg_p = tuple(sorted(set(arguments.exclude_pg_p)))

    configure_plot_style()
    written: List[Path] = []
    if not arguments.dual_axis_only:
        written.extend(
            make_combined_figure(
                plot_summary,
                output_dir,
                excluded_pg_p=excluded_pg_p,
                x_axis_start_zero=arguments.x_axis_start_zero,
                p_values=plot_dimensions,
            )
        )
        written.extend(
            make_diagnostic_figure(
                plot_summary,
                output_dir,
                excluded_pg_p=excluded_pg_p,
                x_axis_start_zero=arguments.x_axis_start_zero,
                p_values=plot_dimensions,
            )
        )
        written.extend(
            make_standalone_figures(
                plot_summary,
                output_dir,
                excluded_pg_p=excluded_pg_p,
                x_axis_start_zero=arguments.x_axis_start_zero,
                p_values=plot_dimensions,
            )
        )
    for method in METHOD_ORDER:
        written.extend(
            make_method_dual_axis_figure(
                plot_summary,
                output_dir,
                method,
                excluded_pg_p=excluded_pg_p,
                x_axis_start_zero=arguments.x_axis_start_zero,
                p_values=plot_dimensions,
                line_width=arguments.dual_axis_linewidth,
                output_suffix=arguments.dual_axis_output_suffix,
            )
        )

    print("Read %d raw run(s)." % raw.shape[0])
    if excluded_pg_p:
        print(
            "Plot-only LDPM-PG exclusions: %s"
            % ", ".join("p=%d" % p for p in excluded_pg_p)
        )
    if arguments.plot_overrides is not None:
        print(
            "Applied %d plot-only metric override(s) from %s"
            % (override_count, arguments.plot_overrides.resolve())
        )
    print("Wrote %s" % (output_dir / "summary_results.csv"))
    for path in written:
        print("Wrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
