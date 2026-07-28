"""Matrix-free Group-Lasso primitives for the AGILS synthetic data protocol."""

from __future__ import annotations

import time
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None


Array = np.ndarray


def _spectral_norm_squared(matrix: Array, iterations: int = 40) -> float:
    rng = np.random.default_rng(0)
    vector = rng.normal(size=matrix.shape[1])
    vector /= max(float(np.linalg.norm(vector)), 1e-15)
    for _ in range(iterations):
        vector = matrix.T @ (matrix @ vector)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-15:
            return 0.0
        vector /= norm
    image = matrix @ vector
    return float(np.dot(image, image))


class MatrixSparseGroupLassoProblem:
    """Direct-matrix problem with separable group penalties.

    The synthetic experiment in :mod:`group_lasso_synthetic_experiment` passes
    five ``group_l2`` regularizers and no L1 regularizer.  The generic L1 code
    paths only keep this primitive reusable; they are not part of the reported
    experiment.
    """

    direct_sgl = True

    def __init__(self, data_info, regularizers: Sequence[dict], setting=None):
        self.data_info = data_info
        self.data = data_info.data
        self.settings = data_info.settings
        self.setting = dict(setting or {})
        self.regularizers = [dict(item) for item in regularizers]
        self.slices = [
            item["slice"] for item in self.regularizers if item["type"] == "group_l2"
        ]
        self.p = int(self.settings.num_features)
        self.group_count = len(self.regularizers)
        self.time_origin = float(self.setting.get("time_origin", time.perf_counter()))

        self.x_train = np.asarray(self.data.X_train, dtype=float)
        self.y_train = np.asarray(self.data.y_train, dtype=float).reshape(-1)
        self.x_validate = np.asarray(self.data.X_validate, dtype=float)
        self.y_validate = np.asarray(self.data.y_validate, dtype=float).reshape(-1)
        self.x_test = np.asarray(self.data.X_test, dtype=float)
        self.y_test = np.asarray(self.data.y_test, dtype=float).reshape(-1)
        self.n_train = max(1, self.y_train.size)
        self.n_validate = max(1, self.y_validate.size)
        self.n_test = max(1, self.y_test.size)
        self.train_lipschitz = max(
            _spectral_norm_squared(self.x_train) / self.n_train, 1e-12
        )
        self.val_lipschitz = max(
            _spectral_norm_squared(self.x_validate) / self.n_validate, 1e-12
        )

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.time_origin)

    @staticmethod
    def _loss(matrix: Array, response: Array, coefficient: Array) -> float:
        residual = matrix @ coefficient - response
        return float(0.5 * np.dot(residual, residual) / max(1, response.size))

    def train_loss(self, coefficient: Array) -> float:
        return self._loss(self.x_train, self.y_train, coefficient)

    def validation_loss(self, coefficient: Array) -> float:
        return self._loss(self.x_validate, self.y_validate, coefficient)

    def test_loss(self, coefficient: Array) -> float:
        return self._loss(self.x_test, self.y_test, coefficient)

    def train_grad(self, coefficient: Array) -> Array:
        return self.x_train.T @ (self.x_train @ coefficient - self.y_train) / self.n_train

    def val_grad(self, coefficient: Array) -> Array:
        return (
            self.x_validate.T @ (self.x_validate @ coefficient - self.y_validate)
            / self.n_validate
        )

    def group_norms(self, coefficient: Array) -> Array:
        values = []
        for regularizer in self.regularizers:
            if regularizer["type"] == "group_l2":
                values.append(np.linalg.norm(coefficient[regularizer["slice"]]))
            elif regularizer["type"] == "l1":
                values.append(np.linalg.norm(coefficient, 1))
            else:
                raise ValueError("unsupported regularizer %r" % regularizer["type"])
        return np.asarray(values, dtype=float)

    def lower_objective(self, lam: Array, coefficient: Array) -> float:
        return float(self.train_loss(coefficient) + np.dot(lam, self.group_norms(coefficient)))

    def prox(self, value: Array, step: float, lam: Array) -> Array:
        output = np.asarray(value, dtype=float).copy()
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] == "l1":
                threshold = step * float(lam[index])
                output = np.sign(output) * np.maximum(np.abs(output) - threshold, 0.0)
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] != "group_l2":
                continue
            block = regularizer["slice"]
            norm = float(np.linalg.norm(output[block]))
            shrink = max(0.0, 1.0 - step * float(lam[index]) / max(norm, 1e-15))
            output[block] *= shrink
        return output

    def lower_solve(
        self,
        lam: Array,
        x0: Optional[Array] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Tuple[Array, int]:
        lam = np.asarray(lam, dtype=float).reshape(self.group_count)
        max_iter = int(max_iter or self.setting.get("lower_max_iter", 5000))
        tol = float(tol or self.setting.get("lower_tol", 1e-9))
        step = 1.0 / self.train_lipschitz
        coefficient = (
            np.zeros(self.p) if x0 is None else np.asarray(x0, dtype=float).copy()
        )
        extrapolated = coefficient.copy()
        momentum = 1.0
        for iteration in range(1, max_iter + 1):
            old = coefficient.copy()
            coefficient = self.prox(
                extrapolated - step * self.train_grad(extrapolated), step, lam
            )
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = coefficient + ((momentum - 1.0) / next_momentum) * (
                coefficient - old
            )
            momentum = next_momentum
            if np.linalg.norm(coefficient - old) / max(1.0, np.linalg.norm(old)) <= tol:
                return coefficient, iteration
        return coefficient, max_iter

    def proximal_lower_solve(
        self,
        lam: Array,
        center: Array,
        gamma: float,
        x0: Optional[Array] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Tuple[Array, int]:
        lam = np.asarray(lam, dtype=float).reshape(self.group_count)
        center = np.asarray(center, dtype=float).reshape(self.p)
        max_iter = int(max_iter or self.setting.get("lower_max_iter", 5000))
        tol = float(tol or self.setting.get("lower_tol", 1e-9))
        step = 1.0 / (self.train_lipschitz + 1.0 / gamma)
        coefficient = center.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
        extrapolated = coefficient.copy()
        momentum = 1.0
        for iteration in range(1, max_iter + 1):
            old = coefficient.copy()
            gradient = self.train_grad(extrapolated) + (extrapolated - center) / gamma
            coefficient = self.prox(extrapolated - step * gradient, step, lam)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = coefficient + ((momentum - 1.0) / next_momentum) * (
                coefficient - old
            )
            momentum = next_momentum
            if np.linalg.norm(coefficient - old) / max(1.0, np.linalg.norm(old)) <= tol:
                return coefficient, iteration
        return coefficient, max_iter

    def cvx_train_loss(self, coefficient):
        if cp is None:
            raise ImportError("cvxpy is required")
        return 0.5 / self.n_train * cp.sum_squares(
            self.x_train @ coefficient - self.y_train
        )

    def cvx_validation_loss(self, coefficient):
        if cp is None:
            raise ImportError("cvxpy is required")
        return 0.5 / self.n_validate * cp.sum_squares(
            self.x_validate @ coefficient - self.y_validate
        )

    def cvx_epigraph_constraints(self, coefficient, radius):
        if cp is None:
            raise ImportError("cvxpy is required")
        constraints = []
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] == "group_l2":
                constraints.append(
                    cp.norm(coefficient[regularizer["slice"]], 2) <= radius[index]
                )
            elif regularizer["type"] == "l1":
                constraints.append(cp.norm1(coefficient) <= radius[index])
            else:
                raise ValueError("unsupported regularizer %r" % regularizer["type"])
        return constraints

    def cvx_epigraph_residuals(self, coefficient, radius):
        if cp is None:
            raise ImportError("cvxpy is required")
        residuals = []
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] == "group_l2":
                residuals.append(
                    cp.norm(coefficient[regularizer["slice"]], 2) - radius[index]
                )
            elif regularizer["type"] == "l1":
                residuals.append(cp.norm1(coefficient) - radius[index])
            else:
                raise ValueError("unsupported regularizer %r" % regularizer["type"])
        return residuals

    def scaled_training_data(self) -> Tuple[Array, Array]:
        root = np.sqrt(float(self.n_train))
        return self.x_train / root, self.y_train / root

    def scaled_validation_data(self) -> Tuple[Array, Array]:
        root = np.sqrt(float(self.n_validate))
        return self.x_validate / root, self.y_validate / root

    def record(self, iteration: int, coefficient: Array, lam: Array, stop: float, **extra):
        row = {
            "iteration": int(iteration),
            "time": self.elapsed(),
            "train_error": self.train_loss(coefficient),
            "validation_error": self.validation_loss(coefficient),
            "test_error": self.test_loss(coefficient),
            "x_lambda_stop": float(stop),
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "lambda_l2": float(np.linalg.norm(lam)),
            "lambda_values": ";".join(
                "%.17g" % value for value in np.asarray(lam, dtype=float)
            ),
        }
        row.update(extra)
        return row
