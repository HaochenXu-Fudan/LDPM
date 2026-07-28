import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from matrix_algorithms import (  # noqa: E402
    IGJO,
    LDPM,
    LDMMA,
    MatrixSetting,
    Grid_Search,
    Random_Search,
    TPE_Search,
    VF_iDCA,
    attach_lower_level_quality,
    generate_matrix_completion_data,
)


def _env_list(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _method_label(method):
    return {
        "GS": "Grid Search",
        "RS": "Random Search",
        "TPE": "TPE",
        "HC": "IGJO",
        "DC": "VF-iDCA",
        "VF": "VF-iDCA",
        "MM": "LDMMA",
        "LDMMA": "LDMMA",
        "PM": "LDPM-CS",
        "LDPM": "LDPM-CS",
    }.get(method, method)


def _latest_summary(result, method):
    row = result.iloc[-1]
    quality = result.attrs.get("lower_quality", {})
    return {
        "method": _method_label(method),
        "iteration": int(row["iteration"]),
        "time": float(row["time"]),
        "validation_error": float(row["validation_error"]),
        "test_error": float(row["test_error"]),
        "lower_common_gap": float(quality.get("lower_quality_common_gap", np.nan)),
        "lower_native_metric": quality.get("lower_quality_native_name", ""),
        "lower_native_value": float(quality.get("lower_quality_native_value", np.nan)),
        "refit_validation_error": float(quality.get("refit_validation_error", np.nan)),
        "refit_test_error": float(quality.get("refit_test_error", np.nan)),
    }


def _print_summary(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    print("\nFinal summary")
    print(
        df[
            [
                "method",
                "iteration",
                "time",
                "validation_error",
                "test_error",
                "lower_common_gap",
                "lower_native_metric",
                "lower_native_value",
                "refit_validation_error",
                "refit_test_error",
            ]
        ].to_string(
            index=False,
            formatters={
                "time": "{:.2f}".format,
                "validation_error": "{:.4e}".format,
                "test_error": "{:.4e}".format,
                "lower_common_gap": "{:.4e}".format,
                "lower_native_value": "{:.4e}".format,
                "refit_validation_error": "{:.4e}".format,
                "refit_test_error": "{:.4e}".format,
            },
        )
    )


def main():
    num_repeat = _env_int("MATRIX_REPEAT", 1)
    methods = _env_list("MATRIX_METHODS", ["DC", "MM", "PM"])
    rows = _env_int("MATRIX_ROWS", 40)
    cols = _env_int("MATRIX_COLS", 40)
    rank = _env_int("MATRIX_RANK", 3)
    sparsity = _env_float("MATRIX_SPARSITY", 0.2)
    snr = _env_float("MATRIX_SNR", 5.0)
    save_results = os.environ.get("MATRIX_SAVE", "0") not in {"0", "false", "False", "no", "No"}
    compute_quality = os.environ.get("MATRIX_COMPUTE_QUALITY", "1") not in {
        "0",
        "false",
        "False",
        "no",
        "No",
    }

    result_path = os.path.join(PARENT_DIR, "results")
    if save_results:
        os.makedirs(os.path.join(result_path, "matrix"), exist_ok=True)

    marker = f"_{rows}_{cols}_rank{rank}_sp{sparsity:g}"
    common_solver = {
        "cvxpy_solver": os.environ.get("MATRIX_CVXPY_SOLVER", "SCS"),
        "cvxpy_tol": _env_float("MATRIX_CVXPY_TOL", 1e-3),
        "cvxpy_max_iter": _env_int("MATRIX_CVXPY_MAX_ITER", 2500),
        "quality_cvxpy_tol": _env_float("MATRIX_QUALITY_CVXPY_TOL", 1e-4),
        "quality_cvxpy_max_iter": _env_int("MATRIX_QUALITY_CVXPY_MAX_ITER", 6000),
    }

    search_setting = {
        **common_solver,
        "grid_size": _env_int("MATRIX_GRID_SIZE", 5),
        "n_eval": _env_int("MATRIX_RANDOM_EVAL", 20),
        "log_bounds": (
            _env_float("MATRIX_LOG_LAMBDA_LOW", -3.0),
            _env_float("MATRIX_LOG_LAMBDA_HIGH", -0.5),
        ),
    }
    dc_setting = {
        **common_solver,
        "MAX_ITERATION": _env_int("MATRIX_DC_MAX_ITER", _env_int("MATRIX_MAX_ITER", 5)),
        "MIN_ITERATION": _env_int("MATRIX_DC_MIN_ITER", 1),
        "TOL": _env_float("MATRIX_DC_TOL", 5e-3),
        "rho": _env_float("MATRIX_DC_RHO", 1e-2),
        "alpha0": _env_float("MATRIX_DC_ALPHA0", 10.0),
        "initial_lambda": [
            _env_float("MATRIX_INITIAL_LAMBDA_L1", 0.001),
            _env_float("MATRIX_INITIAL_LAMBDA_NUCLEAR", 0.001),
        ],
    }
    mm_setting = {
        **common_solver,
        "MAX_ITERATION": _env_int("MATRIX_MM_MAX_ITER", _env_int("MATRIX_MAX_ITER", 6)),
        "MIN_ITERATION": _env_int("MATRIX_MM_MIN_ITER", 1),
        "TOL": _env_float("MATRIX_MM_TOL", 5e-3),
        "epsilon": _env_float("MATRIX_MM_EPSILON", 1e-2),
        "prox_beta": _env_float("MATRIX_MM_PROX_BETA", 1e-3),
    }
    pm_setting = {
        **common_solver,
        "MAX_ITERATION": _env_int("MATRIX_PM_MAX_ITER", 2000),
        "MIN_ITERATION": _env_int("MATRIX_PM_MIN_ITER", 100),
        "TOL": _env_float("MATRIX_PM_TOL", 1e-5),
        "step_size": _env_float("MATRIX_PM_STEP_SIZE", 2e-2),
        "gamma": _env_float("MATRIX_PM_GAMMA", 10.0),
        "beta0": _env_float("MATRIX_PM_BETA0", 1e-3),
        "beta_power": _env_float("MATRIX_PM_BETA_POWER", 0.3),
        "beta_max": _env_float("MATRIX_PM_BETA_MAX", 1e6),
    }

    summary_rows = []
    for repeat in range(num_repeat):
        print(f"Matrix sparse-low-rank experiment {repeat + 1}/{num_repeat}")
        setting = MatrixSetting(
            num_rows=rows,
            num_cols=cols,
            rank=rank,
            sparsity=sparsity,
            snr=snr,
            num_train=_env_int("MATRIX_NUM_TRAIN", 0),
            num_val=_env_int("MATRIX_NUM_VAL", 0),
            num_test=_env_int("MATRIX_NUM_TEST", 0),
            train_fraction=_env_float("MATRIX_TRAIN_FRACTION", 0.25),
            val_fraction=_env_float("MATRIX_VAL_FRACTION", 0.10),
            test_fraction=_env_float("MATRIX_TEST_FRACTION", 0.10),
            print_flag=True,
        )
        data_info = generate_matrix_completion_data(setting, seed=repeat + 1)
        data_info.data_index = repeat + 1

        for method in methods:
            method_key = method.upper()
            print(f"\nRunning {_method_label(method_key)}")
            if method_key == "GS":
                result = Grid_Search(data_info, search_setting)
                file_key = "GS"
            elif method_key == "RS":
                local_setting = dict(search_setting)
                local_setting["seed"] = repeat + 1
                result = Random_Search(data_info, local_setting)
                file_key = "RS"
            elif method_key == "TPE":
                local_setting = dict(search_setting)
                local_setting["seed"] = repeat + 1
                result = TPE_Search(data_info, local_setting)
                file_key = "TPE"
            elif method_key == "HC":
                result = IGJO(data_info, search_setting)
                file_key = "HC"
            elif method_key in {"DC", "VF"}:
                result = VF_iDCA(data_info, dc_setting)
                file_key = "DC"
            elif method_key in {"MM", "LDMMA"}:
                result = LDMMA(data_info, mm_setting)
                file_key = "MM"
            elif method_key in {"PM", "LDPM"}:
                result = LDPM(data_info, pm_setting)
                file_key = "PM"
            else:
                raise ValueError(f"Unknown MATRIX_METHODS entry: {method}")

            if compute_quality:
                if file_key == "DC":
                    result = attach_lower_level_quality(result, data_info, dc_setting, "latest")
                elif file_key == "MM":
                    result = attach_lower_level_quality(result, data_info, mm_setting, "latest")
                elif file_key == "PM":
                    result = attach_lower_level_quality(result, data_info, pm_setting, "latest")
                else:
                    result = attach_lower_level_quality(result, data_info, search_setting, "best")

            row = _latest_summary(result, method_key)
            summary_rows.append(row)
            print(
                f"{row['method']:>12s} | iter {row['iteration']:4d}, "
                f"time {row['time']:.2f}s, val {row['validation_error']:.4e}, "
                f"test {row['test_error']:.4e}, lower {row['lower_native_metric']}="
                f"{row['lower_native_value']:.4e}"
            )

            if save_results:
                result.to_pickle(f"{result_path}/matrix/{file_key}_{data_info.data_index}{marker}.pkl")

    _print_summary(summary_rows)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
