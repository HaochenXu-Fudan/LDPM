#!/usr/bin/env python3
"""School weighted-l1 plus nuclear-norm bilevel experiment.

This is the minimum experiment requested by
``school_weighted_l1_nuclear_experiment.md``: one task-wise split, all 139
tasks, and LDPM-CS / capped LDPM-CS-C only.  Optimization is test-blind.  A
fixed-hyperparameter lower problem is solved after each LDPM run to obtain the
feasible validation/test errors used by the paper-style reporting contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import loadmat
from scipy.linalg import cho_factor, cho_solve


Array = np.ndarray
METHOD_LABELS = {"ldpm": "LDPM-CS", "ldpm-capped": "LDPM-CS-C"}


def project_coordinate_epigraph(values: Array, radii: Array) -> Tuple[Array, Array]:
    """Project independent pairs onto ``{(u,r): |u| <= r}``."""

    values = np.asarray(values, dtype=float)
    radii = np.asarray(radii, dtype=float)
    absolute = np.abs(values)
    inside = absolute <= radii
    polar = absolute <= -radii
    middle = ~(inside | polar)
    projected_values = values.copy()
    projected_radii = radii.copy()
    projected_values[polar] = 0.0
    projected_radii[polar] = 0.0
    scale = 0.5 * (absolute[middle] + radii[middle])
    projected_values[middle] = np.sign(values[middle]) * scale
    projected_radii[middle] = scale
    return projected_values, projected_radii


def project_l1_epigraph(values: Array, radius: float) -> Tuple[Array, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    radius = float(radius)
    absolute = np.abs(flat)
    if radius >= 0.0 and float(np.sum(absolute)) <= radius:
        return flat.copy(), radius
    low = 0.0
    high = max(float(np.max(absolute)) if absolute.size else 0.0, -radius, 1.0)

    def residual(threshold: float) -> float:
        return float(np.sum(np.maximum(absolute - threshold, 0.0)) - radius - threshold)

    while residual(high) > 0.0:
        high *= 2.0
    for _ in range(70):
        middle = 0.5 * (low + high)
        if residual(middle) > 0.0:
            low = middle
        else:
            high = middle
    threshold = high
    projected = np.sign(flat) * np.maximum(absolute - threshold, 0.0)
    return projected, max(radius + threshold, 0.0)


def project_linf_epigraph(values: Array, radius: float) -> Tuple[Array, float]:
    flat = np.asarray(values, dtype=float).reshape(-1)
    radius = float(radius)
    absolute = np.abs(flat)
    maximum = float(np.max(absolute)) if absolute.size else 0.0
    if radius >= 0.0 and maximum <= radius:
        return flat.copy(), radius

    def derivative(threshold: float) -> float:
        active = absolute > threshold
        return float(threshold - radius + np.sum(threshold - absolute[active]))

    if derivative(0.0) >= 0.0:
        threshold = 0.0
    else:
        low = 0.0
        high = max(maximum, radius, 1.0)
        while derivative(high) < 0.0:
            high *= 2.0
        for _ in range(70):
            middle = 0.5 * (low + high)
            if derivative(middle) < 0.0:
                low = middle
            else:
                high = middle
        threshold = high
    return np.clip(flat, -threshold, threshold), max(threshold, 0.0)


def _singular_values_and_transform(
    matrix: Array,
    transform,
) -> Tuple[Array, Array]:
    """Apply a scalar transform to singular values using the smaller Gram matrix."""

    matrix = np.asarray(matrix, dtype=float)
    rows, columns = matrix.shape
    if rows <= columns:
        eigenvalues, left = np.linalg.eigh(matrix @ matrix.T)
        order = np.argsort(eigenvalues)[::-1]
        singular = np.sqrt(np.maximum(eigenvalues[order], 0.0))
        left = left[:, order]
        transformed = np.asarray(transform(singular), dtype=float)
        cutoff = (
            np.finfo(float).eps
            * max(rows, columns)
            * max(float(singular[0]) if singular.size else 0.0, 1.0)
        )
        ratio = np.zeros_like(singular)
        nonzero = singular > cutoff
        ratio[nonzero] = transformed[nonzero] / singular[nonzero]
        projected = left @ (ratio[:, None] * (left.T @ matrix))
        return projected, singular
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    transformed = np.asarray(transform(singular), dtype=float)
    return (left * transformed.reshape(1, -1)) @ right, singular


def project_nuclear_epigraph(matrix: Array, radius: float) -> Tuple[Array, float]:
    projected_radius = float(radius)

    def transform(singular: Array) -> Array:
        nonlocal projected_radius
        projected, projected_radius = project_l1_epigraph(singular, radius)
        return projected

    projected_matrix, _ = _singular_values_and_transform(matrix, transform)
    return projected_matrix, projected_radius


def project_spectral_epigraph(matrix: Array, radius: float) -> Tuple[Array, float]:
    projected_radius = float(radius)

    def transform(singular: Array) -> Array:
        nonlocal projected_radius
        projected, projected_radius = project_linf_epigraph(singular, radius)
        return projected

    projected_matrix, _ = _singular_values_and_transform(matrix, transform)
    return projected_matrix, projected_radius


def singular_value_threshold(matrix: Array, threshold: float) -> Array:
    if threshold <= 0.0:
        return np.asarray(matrix, dtype=float).copy()
    projected, _ = _singular_values_and_transform(
        matrix, lambda singular: np.maximum(singular - threshold, 0.0)
    )
    return projected


@dataclass
class SchoolData:
    train_a: List[Array]
    train_b: List[Array]
    validation_a: List[Array]
    validation_b: List[Array]
    test_a: List[Array]
    test_b: List[Array]
    split_indices: List[Tuple[Array, Array, Array]]
    pooled_mean: Array
    pooled_std: Array
    task_feature_mean: Array
    task_response_mean: Array
    removed_columns: Array
    source_sha256: str

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.train_a[0].shape[1]), len(self.train_a)


def _split_counts(sample_count: int) -> Tuple[int, int, int]:
    train = int(math.floor(0.60 * sample_count))
    validation = int(math.floor(0.20 * sample_count))
    test = sample_count - train - validation
    if min(train, validation, test) <= 0:
        raise ValueError("each School task must have samples in all three subsets")
    return train, validation, test


def load_and_preprocess_school(path: Path, seed: int) -> SchoolData:
    raw = loadmat(path, squeeze_me=True)
    if "X" not in raw or "Y" not in raw:
        raise ValueError("school.mat must contain X and Y task cell arrays")
    raw_x = [np.asarray(value, dtype=float) for value in np.ravel(raw["X"])]
    raw_y = [np.asarray(value, dtype=float).reshape(-1) for value in np.ravel(raw["Y"])]
    if len(raw_x) != 139 or len(raw_y) != 139:
        raise ValueError(f"expected 139 School tasks, found X={len(raw_x)}, Y={len(raw_y)}")
    if any(len(x) != len(y) for x, y in zip(raw_x, raw_y)):
        raise ValueError("a School task has inconsistent X/Y sample counts")

    pooled_raw = np.vstack(raw_x)
    constant = np.ptp(pooled_raw, axis=0) <= 1e-12
    removed_columns = np.flatnonzero(constant)
    kept_columns = np.flatnonzero(~constant)
    if len(kept_columns) != 27:
        raise ValueError(
            f"expected 27 nonconstant features after intercept removal, found {len(kept_columns)}"
        )
    raw_x = [value[:, kept_columns] for value in raw_x]

    train_indices: List[Array] = []
    validation_indices: List[Array] = []
    test_indices: List[Array] = []
    for task, values in enumerate(raw_x):
        # A separate deterministic stream per task makes split regeneration stable.
        permutation = np.random.default_rng(seed + 104729 * task).permutation(len(values))
        n_train, n_validation, _ = _split_counts(len(values))
        train_indices.append(np.sort(permutation[:n_train]))
        validation_indices.append(
            np.sort(permutation[n_train : n_train + n_validation])
        )
        test_indices.append(np.sort(permutation[n_train + n_validation :]))

    pooled_training = np.vstack(
        [values[index] for values, index in zip(raw_x, train_indices)]
    )
    pooled_mean = np.mean(pooled_training, axis=0)
    pooled_std = np.std(pooled_training, axis=0)
    pooled_std = np.where(pooled_std <= 1e-12, 1.0, pooled_std)
    standardized = [(values - pooled_mean) / pooled_std for values in raw_x]

    train_a: List[Array] = []
    train_b: List[Array] = []
    validation_a: List[Array] = []
    validation_b: List[Array] = []
    test_a: List[Array] = []
    test_b: List[Array] = []
    task_feature_mean = np.zeros((27, 139), dtype=float)
    task_response_mean = np.zeros(139, dtype=float)
    for task, (features, response) in enumerate(zip(standardized, raw_y)):
        feature_mean = np.mean(features[train_indices[task]], axis=0)
        response_mean = float(np.mean(response[train_indices[task]]))
        task_feature_mean[:, task] = feature_mean
        task_response_mean[task] = response_mean
        centered_features = features - feature_mean
        centered_response = response - response_mean
        train_a.append(centered_features[train_indices[task]])
        train_b.append(centered_response[train_indices[task]])
        validation_a.append(centered_features[validation_indices[task]])
        validation_b.append(centered_response[validation_indices[task]])
        test_a.append(centered_features[test_indices[task]])
        test_b.append(centered_response[test_indices[task]])

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return SchoolData(
        train_a=train_a,
        train_b=train_b,
        validation_a=validation_a,
        validation_b=validation_b,
        test_a=test_a,
        test_b=test_b,
        split_indices=list(zip(train_indices, validation_indices, test_indices)),
        pooled_mean=pooled_mean,
        pooled_std=pooled_std,
        task_feature_mean=task_feature_mean,
        task_response_mean=task_response_mean,
        removed_columns=removed_columns,
        source_sha256=digest,
    )


class TaskLossOperator:
    """Pooled per-observation block-diagonal least-squares operator."""

    def __init__(self, features: Sequence[Array], responses: Sequence[Array]) -> None:
        self.features = [np.asarray(value, dtype=float) for value in features]
        self.responses = [np.asarray(value, dtype=float).reshape(-1) for value in responses]
        self.counts = np.asarray([len(value) for value in self.responses], dtype=int)
        self.offsets = np.concatenate(([0], np.cumsum(self.counts)))
        self.n = int(np.sum(self.counts))
        self.scale = math.sqrt(self.n)
        self.b = np.concatenate(self.responses) / self.scale

    def apply(self, matrix: Array) -> Array:
        return np.concatenate(
            [features @ matrix[:, task] for task, features in enumerate(self.features)]
        ) / self.scale

    def adjoint(self, values: Array) -> Array:
        values = np.asarray(values, dtype=float).reshape(self.n)
        columns = []
        for task, features in enumerate(self.features):
            section = values[self.offsets[task] : self.offsets[task + 1]]
            columns.append(features.T @ section / self.scale)
        return np.column_stack(columns)

    def loss(self, matrix: Array) -> float:
        residual = self.apply(matrix) - self.b
        return 0.5 * float(np.dot(residual, residual))

    def gradient(self, matrix: Array) -> Array:
        return self.adjoint(self.apply(matrix) - self.b)

    def mse(self, matrix: Array) -> float:
        residual = self.apply(matrix) - self.b
        return float(np.dot(residual, residual))


@dataclass
class LDPMConfig:
    max_iter: int = 5000
    min_iter: int = 1
    tol: float = 1e-4
    beta0: float = 1.0
    beta_power: float = 0.4
    beta_max: float = 27.0
    gamma: float = 10.0
    initial_step: float = 0.1
    max_step: float = 0.1
    min_step: float = 1e-12
    line_search_decay: float = 0.5
    line_search_growth: float = 1.0
    max_line_search_iter: int = 60
    beta_step_scale: Optional[float] = None
    consensus_tol: Optional[float] = None
    convergence_window: int = 1
    max_time: Optional[float] = 600.0
    record_interval: int = 10


class SchoolLDPMProblem:
    """Two-block Cartesian weighted-l1 / nuclear LDPM-CS problem."""

    def __init__(self, training: TaskLossOperator, validation: TaskLossOperator) -> None:
        self.training = training
        self.validation = validation
        self.d = int(training.features[0].shape[1])
        self.tasks = len(training.features)
        self.matrix_size = self.d * self.tasks
        gradient_zero = training.gradient(np.zeros((self.d, self.tasks), dtype=float))
        self.lambda_l1_max = float(np.max(np.abs(gradient_zero)))
        self.lambda_nuclear_max = float(np.linalg.norm(gradient_zero, ord=2))
        if min(self.lambda_l1_max, self.lambda_nuclear_max) <= 0.0:
            raise ValueError("data-driven lambda upper bounds must be positive")
        position = 0
        self.w_slice = slice(position, position + self.matrix_size)
        position += self.matrix_size
        self.lambda_slice = slice(position, position + self.matrix_size + 1)
        position += self.matrix_size + 1
        self.p_slice = slice(position, position + self.matrix_size)
        position += self.matrix_size
        self.q_slice = slice(position, position + self.matrix_size)
        position += self.matrix_size
        self.radius_slice = slice(position, position + self.matrix_size + 1)
        position += self.matrix_size + 1
        self.z_dim = position
        self.lambda_upper = np.concatenate(
            (np.full(self.matrix_size, self.lambda_l1_max), [self.lambda_nuclear_max])
        )

    def pack(self, w: Array, lam: Array, p: Array, q: Array, radius: Array) -> Array:
        result = np.concatenate(
            (
                np.asarray(w, dtype=float).reshape(-1),
                np.asarray(lam, dtype=float).reshape(-1),
                np.asarray(p, dtype=float).reshape(-1),
                np.asarray(q, dtype=float).reshape(-1),
                np.asarray(radius, dtype=float).reshape(-1),
            )
        )
        if len(result) != self.z_dim:
            raise AssertionError("packed LDPM state has the wrong dimension")
        return result

    def unpack(self, z: Array) -> Tuple[Array, Array, Array, Array, Array]:
        z = np.asarray(z, dtype=float).reshape(self.z_dim)
        w = z[self.w_slice].reshape(self.d, self.tasks).copy()
        lam = z[self.lambda_slice].copy()
        p = z[self.p_slice].reshape(self.d, self.tasks).copy()
        q = z[self.q_slice].reshape(self.d, self.tasks).copy()
        radius = z[self.radius_slice].copy()
        return w, lam, p, q, radius

    def initial_state(self) -> Tuple[Array, Array]:
        w = np.zeros((self.d, self.tasks), dtype=float)
        lam = 0.05 * self.lambda_upper
        p = np.zeros_like(w)
        q = np.zeros_like(w)
        radius = np.zeros(self.matrix_size + 1, dtype=float)
        xi = np.zeros(self.training.n, dtype=float)
        return self.pack(w, lam, p, q, radius), xi

    def project_common(self, z: Array) -> Array:
        w, lam, p, q, radius = self.unpack(z)
        lam = np.minimum(np.maximum(lam, 0.0), self.lambda_upper)
        radius = np.maximum(radius, 0.0)
        return self.pack(w, lam, p, q, radius)

    def project_local(self, z: Array, block: int) -> Array:
        w, lam, p, q, radius = self.unpack(z)
        lam_l1 = lam[:-1].reshape(self.d, self.tasks)
        radius_l1 = radius[:-1].reshape(self.d, self.tasks)
        if block == 0:
            w, radius_l1 = project_coordinate_epigraph(w, radius_l1)
        elif block == 1:
            w, radius[-1] = project_nuclear_epigraph(w, radius[-1])
        else:
            raise IndexError("School LDPM has exactly two consensus blocks")
        p, lam_l1 = project_coordinate_epigraph(p, lam_l1)
        q, lam[-1] = project_spectral_epigraph(q, lam[-1])
        lam[:-1] = lam_l1.reshape(-1)
        radius[:-1] = np.maximum(radius_l1.reshape(-1), 0.0)
        return self.pack(w, lam, p, q, radius)

    def components(self, z: Array, xi: Array) -> Dict[str, float | Array]:
        w, lam, p, q, radius = self.unpack(z)
        h = self.training.adjoint(xi) + p + q
        p_value = (
            self.training.loss(w)
            + float(np.dot(lam, radius))
            + 0.5 * float(np.dot(xi, xi))
            + float(np.dot(xi, self.training.b))
        )
        raw_gap = p_value - float(np.sum(w * h))
        h_norm = float(np.linalg.norm(h, "fro"))
        loss_residual = self.training.apply(w) - self.training.b - xi
        loss_gap = 0.5 * float(np.dot(loss_residual, loss_residual))
        l1_gap = float(np.dot(lam[:-1], radius[:-1]) - np.sum(w * p))
        nuclear_gap = float(lam[-1] * radius[-1] - np.sum(w * q))
        stable_gap = loss_gap + l1_gap + nuclear_gap
        return {
            "h": h,
            "p_value": p_value,
            "raw_gap": raw_gap,
            "raw_psi": raw_gap + 0.5 * h_norm * h_norm,
            "stable_gap": stable_gap,
            "stable_psi": stable_gap + 0.5 * h_norm * h_norm,
            "h_norm": h_norm,
            "loss_gap": loss_gap,
            "l1_gap": l1_gap,
            "nuclear_gap": nuclear_gap,
        }

    def smooth_value(
        self,
        z: Array,
        xi: Array,
        copies: Array,
        multipliers: Array,
        beta: float,
        gamma: float,
    ) -> float:
        w, _, _, _, _ = self.unpack(z)
        values = self.components(z, xi)
        difference = copies - z[None, :]
        consensus = float(
            np.sum(multipliers * difference) + 0.5 * gamma * np.sum(difference * difference)
        )
        return (
            self.validation.loss(w)
            + beta * (float(values["raw_psi"]) - 0.5 * float(np.dot(xi, xi)))
            + consensus
        )

    def gradients(
        self,
        z: Array,
        xi: Array,
        copies: Array,
        multipliers: Array,
        beta: float,
        gamma: float,
    ) -> Tuple[Array, Array]:
        w, lam, p, q, radius = self.unpack(z)
        h = self.training.adjoint(xi) + p + q
        gradient_w = self.validation.gradient(w) + beta * (self.training.gradient(w) - h)
        difference = h - w
        gradient_z = self.pack(
            gradient_w,
            beta * radius,
            beta * difference,
            beta * difference,
            beta * lam,
        )
        gradient_z += -np.sum(multipliers, axis=0) + gamma * (
            len(copies) * z - np.sum(copies, axis=0)
        )
        gradient_xi = beta * (self.training.b + self.training.apply(difference))
        return gradient_z, gradient_xi

    def stopping_change(self, new: Array, old: Array) -> float:
        w_new, lam_new, _, _, _ = self.unpack(new)
        w_old, lam_old, _, _, _ = self.unpack(old)
        return float(np.linalg.norm(w_new - w_old, "fro")) / max(
            1.0, float(np.linalg.norm(w_old, "fro"))
        ) + float(np.linalg.norm(lam_new - lam_old)) / max(
            1.0, float(np.linalg.norm(lam_old))
        )

    def diagnostics(self, z: Array, xi: Array, copies: Array) -> Dict[str, float]:
        w, lam, p, q, radius = self.unpack(z)
        singular_w = np.linalg.svd(w, compute_uv=False)
        singular_q = np.linalg.svd(q, compute_uv=False)
        consensus = max(
            (float(np.linalg.norm(copy - z)) for copy in copies), default=0.0
        ) / max(1.0, float(np.linalg.norm(z)))
        values = self.components(z, xi)
        return {
            "raw_psi": float(values["raw_psi"]),
            "stable_psi": float(values["stable_psi"]),
            "h_norm": float(values["h_norm"]),
            "consensus_residual": consensus,
            "primal_l1_violation": max(
                0.0, float(np.max(np.abs(w).reshape(-1) - radius[:-1]))
            ),
            "primal_nuclear_violation": max(0.0, float(np.sum(singular_w) - radius[-1])),
            "dual_l1_violation": max(
                0.0, float(np.max(np.abs(p).reshape(-1) - lam[:-1]))
            ),
            "dual_nuclear_violation": max(
                0.0, float((singular_q[0] if singular_q.size else 0.0) - lam[-1])
            ),
            "sparsity": float(np.mean(np.abs(w) <= 1e-6)),
            "effective_rank": float(
                np.sum(singular_w > 1e-4 * singular_w[0]) if singular_w.size and singular_w[0] > 0 else 0
            ),
        }


def _history_row(
    problem: SchoolLDPMProblem,
    z: Array,
    xi: Array,
    copies: Array,
    iteration: int,
    elapsed: float,
    beta: float,
    step: float,
    line_search_trials: int,
    change: float,
    cap_reached: bool,
) -> Dict[str, object]:
    w, lam, _, _, _ = problem.unpack(z)
    diagnostics = problem.diagnostics(z, xi, copies)
    row: Dict[str, object] = {
        "iteration": int(iteration),
        "time": float(elapsed),
        "beta": float(beta),
        "cap_reached": bool(cap_reached),
        "accepted_step": float(step),
        "line_search_trials": int(line_search_trials),
        "x_lambda_stop": float(change),
        "validation_mse_infeasible": problem.validation.mse(w),
        "validation_rmse_infeasible": math.sqrt(problem.validation.mse(w)),
        "lambda_l1_min": float(np.min(lam[:-1])),
        "lambda_l1_max": float(np.max(lam[:-1])),
        "lambda_nuclear": float(lam[-1]),
    }
    row.update(diagnostics)
    return row


def run_ldpm(
    problem: SchoolLDPMProblem, method: str, config: LDPMConfig
) -> Tuple[
    List[Dict[str, object]], Dict[str, Array], Dict[str, object], Dict[str, Array]
]:
    if method not in METHOD_LABELS:
        raise ValueError(f"unknown method: {method}")
    z, xi = problem.initial_state()
    multipliers = np.zeros((2, problem.z_dim), dtype=float)
    copies = np.stack([problem.project_local(z, block) for block in range(2)])
    history: List[Dict[str, object]] = []
    trajectory_w: List[Array] = []
    started = time.perf_counter()
    next_step = float(config.initial_step)
    status = "max_iter"
    message = "maximum iteration budget reached"
    cap_reached = False
    convergence_streak = 0
    beta = float(config.beta0)
    accepted_step = 0.0
    change = float("inf")
    trials = 0
    iteration = 0

    for iteration in range(1, config.max_iter + 1):
        elapsed = time.perf_counter() - started
        if config.max_time is not None and elapsed >= config.max_time:
            status = "time_limit"
            message = "wall-clock time limit reached"
            break
        uncapped_beta = config.beta0 * float(iteration**config.beta_power)
        if method == "ldpm-capped":
            beta = min(uncapped_beta, config.beta_max)
            cap_reached = cap_reached or uncapped_beta >= config.beta_max
        else:
            beta = uncapped_beta
        old_z = z.copy()
        gradient_z, gradient_xi = problem.gradients(
            z, xi, copies, multipliers, beta, config.gamma
        )
        if not np.all(np.isfinite(gradient_z)) or not np.all(np.isfinite(gradient_xi)):
            status = "nonfinite"
            message = "non-finite smooth gradient"
            break
        current = problem.smooth_value(z, xi, copies, multipliers, beta, config.gamma)
        beta_step_cap = (
            config.max_step
            if config.beta_step_scale is None
            else min(config.max_step, config.beta_step_scale / beta)
        )
        trial_step = min(beta_step_cap, max(config.min_step, next_step))
        accepted = None
        for trials in range(1, config.max_line_search_iter + 1):
            trial_z = problem.project_common(z - trial_step * gradient_z)
            trial_xi = (xi - trial_step * gradient_xi) / (1.0 + trial_step * beta)
            if np.all(np.isfinite(trial_z)) and np.all(np.isfinite(trial_xi)):
                trial_value = problem.smooth_value(
                    trial_z, trial_xi, copies, multipliers, beta, config.gamma
                )
                delta_z = trial_z - z
                delta_xi = trial_xi - xi
                upper = (
                    current
                    + float(np.dot(gradient_z, delta_z) + np.dot(gradient_xi, delta_xi))
                    + 0.5 / trial_step * float(np.dot(delta_z, delta_z) + np.dot(delta_xi, delta_xi))
                    + 1e-12 * max(1.0, abs(current))
                )
                if np.isfinite(trial_value) and trial_value <= upper:
                    accepted = (trial_z, trial_xi, trial_step)
                    break
            trial_step *= config.line_search_decay
            if trial_step < config.min_step:
                break
        if accepted is None:
            status = "line_search_failure"
            message = "backtracking step underflow or no sufficient-decrease step"
            break
        z, xi, accepted_step = accepted
        try:
            copies = np.stack(
                [
                    problem.project_local(z - multipliers[block] / config.gamma, block)
                    for block in range(2)
                ]
            )
        except np.linalg.LinAlgError as error:
            status = "svd_failure"
            message = f"full SVD projection failed: {error}"
            break
        multipliers = multipliers + config.gamma * (copies - z[None, :])
        change = problem.stopping_change(z, old_z)
        next_step = min(
            config.max_step, max(config.min_step, accepted_step * config.line_search_growth)
        )
        consensus_change = max(
            float(np.linalg.norm(copy - z)) for copy in copies
        ) / max(1.0, float(np.linalg.norm(z)))
        consensus_ok = (
            config.consensus_tol is None
            or consensus_change <= config.consensus_tol
        )
        if iteration >= config.min_iter and change <= config.tol and consensus_ok:
            convergence_streak += 1
        else:
            convergence_streak = 0
        converged = convergence_streak >= config.convergence_window
        if iteration == 1 or iteration % config.record_interval == 0 or converged:
            history.append(
                _history_row(
                    problem,
                    z,
                    xi,
                    copies,
                    iteration,
                    time.perf_counter() - started,
                    beta,
                    accepted_step,
                    trials,
                    change,
                    cap_reached,
                )
            )
            trajectory_w.append(problem.unpack(z)[0])
        if converged:
            status = "success"
            message = "relative W/lambda change met the requested tolerance"
            break

    elapsed = time.perf_counter() - started
    if not history or int(history[-1]["iteration"]) != iteration:
        history.append(
            _history_row(
                problem,
                z,
                xi,
                copies,
                iteration,
                elapsed,
                beta,
                accepted_step,
                trials,
                change,
                cap_reached,
            )
        )
        trajectory_w.append(problem.unpack(z)[0])
    w, lam, p, q, radius = problem.unpack(z)
    summary: Dict[str, object] = dict(history[-1])
    summary.update(
        {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "status": status,
            "message": message,
            "time": float(elapsed),
            "iterations": int(iteration),
            "cap_reached": bool(cap_reached),
            "final_beta": float(beta),
        }
    )
    state = {
        "W": w,
        "lambda_l1": lam[:-1].reshape(problem.d, problem.tasks),
        "lambda_nuclear": np.asarray(lam[-1]),
        "dual_l1": p,
        "dual_nuclear": q,
        "radius_l1": radius[:-1].reshape(problem.d, problem.tasks),
        "radius_nuclear": np.asarray(radius[-1]),
        "xi": xi,
        "local_copies": copies,
        "consensus_multipliers": multipliers,
    }
    trajectory = {
        "iteration": np.asarray([row["iteration"] for row in history], dtype=int),
        "time": np.asarray([row["time"] for row in history], dtype=float),
        "W": np.stack(trajectory_w),
    }
    return history, state, summary, trajectory


def lower_objective(
    operator: TaskLossOperator, matrix: Array, lambda_l1: Array, lambda_nuclear: float
) -> float:
    return (
        operator.loss(matrix)
        + float(np.sum(lambda_l1 * np.abs(matrix)))
        + float(lambda_nuclear * np.sum(np.linalg.svd(matrix, compute_uv=False)))
    )


def feasible_lower_solve(
    operator: TaskLossOperator,
    lambda_l1: Array,
    lambda_nuclear: float,
    initial: Array,
    *,
    rho: float = 0.1,
    max_iter: int = 5000,
    abs_tol: float = 1e-8,
    rel_tol: float = 1e-7,
) -> Tuple[Array, Dict[str, object]]:
    """High-accuracy two-copy consensus ADMM lower solve at fixed lambdas."""

    d, tasks = initial.shape
    w = np.asarray(initial, dtype=float).copy()
    sparse_copy = w.copy()
    nuclear_copy = w.copy()
    dual_sparse = np.zeros_like(w)
    dual_nuclear = np.zeros_like(w)
    factors = []
    right_data = []
    for features, response in zip(operator.features, operator.responses):
        hessian = features.T @ features / operator.n + 2.0 * rho * np.eye(d)
        factors.append(cho_factor(hessian, lower=True, check_finite=False))
        right_data.append(features.T @ response / operator.n)
    root_p = math.sqrt(d * tasks)
    root_2p = math.sqrt(2 * d * tasks)
    status = "max_iter"
    primal = float("inf")
    dual = float("inf")
    primal_tolerance = float("nan")
    dual_tolerance = float("nan")
    started = time.perf_counter()
    for iteration in range(1, max_iter + 1):
        for task in range(tasks):
            right = right_data[task] + rho * (
                sparse_copy[:, task]
                - dual_sparse[:, task]
                + nuclear_copy[:, task]
                - dual_nuclear[:, task]
            )
            w[:, task] = cho_solve(factors[task], right, check_finite=False)
        old_sparse = sparse_copy.copy()
        old_nuclear = nuclear_copy.copy()
        sparse_copy = np.sign(w + dual_sparse) * np.maximum(
            np.abs(w + dual_sparse) - lambda_l1 / rho, 0.0
        )
        nuclear_copy = singular_value_threshold(w + dual_nuclear, lambda_nuclear / rho)
        dual_sparse += w - sparse_copy
        dual_nuclear += w - nuclear_copy
        primal = math.sqrt(
            float(np.linalg.norm(w - sparse_copy, "fro") ** 2)
            + float(np.linalg.norm(w - nuclear_copy, "fro") ** 2)
        )
        dual = rho * math.sqrt(
            float(np.linalg.norm(sparse_copy - old_sparse, "fro") ** 2)
            + float(np.linalg.norm(nuclear_copy - old_nuclear, "fro") ** 2)
        )
        primal_tolerance = root_2p * abs_tol + rel_tol * max(
            math.sqrt(2.0) * float(np.linalg.norm(w, "fro")),
            math.sqrt(
                float(np.linalg.norm(sparse_copy, "fro") ** 2)
                + float(np.linalg.norm(nuclear_copy, "fro") ** 2)
            ),
        )
        dual_tolerance = root_p * abs_tol + rel_tol * rho * math.sqrt(
            float(np.linalg.norm(dual_sparse, "fro") ** 2)
            + float(np.linalg.norm(dual_nuclear, "fro") ** 2)
        )
        if primal <= primal_tolerance and dual <= dual_tolerance:
            status = "converged"
            break
    return w, {
        "status": status,
        "iterations": int(iteration),
        "time": float(time.perf_counter() - started),
        "primal_residual": float(primal),
        "dual_residual": float(dual),
        "primal_tolerance": float(primal_tolerance),
        "dual_tolerance": float(dual_tolerance),
        "lower_objective": lower_objective(operator, w, lambda_l1, lambda_nuclear),
    }


def _save_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_split_and_preprocessing(output: Path, data: SchoolData) -> None:
    split_values: Dict[str, Array] = {}
    for task, (train, validation, test) in enumerate(data.split_indices):
        split_values[f"train_{task:03d}"] = train
        split_values[f"validation_{task:03d}"] = validation
        split_values[f"test_{task:03d}"] = test
    np.savez_compressed(output / "split_indices.npz", **split_values)
    np.savez_compressed(
        output / "preprocessing.npz",
        pooled_mean=data.pooled_mean,
        pooled_std=data.pooled_std,
        task_feature_mean=data.task_feature_mean,
        task_response_mean=data.task_response_mean,
        removed_columns=data.removed_columns,
    )


def _plot_heatmaps(output: Path, matrices: Dict[str, Array], task_means: Array) -> None:
    # Codex/sandboxed runs often have a read-only home-level matplotlib cache.
    cache = Path(tempfile.gettempdir()) / "school_experiment_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    order = np.argsort(task_means)
    maximum = max(float(np.max(np.abs(value))) for value in matrices.values())
    maximum = max(maximum, np.finfo(float).eps)
    figure, axes = plt.subplots(len(matrices), 1, figsize=(12, 5.5), squeeze=False)
    image = None
    for axis, (method, matrix) in zip(axes[:, 0], matrices.items()):
        image = axis.imshow(
            np.abs(matrix[:, order]), aspect="auto", interpolation="nearest", vmin=0.0, vmax=maximum
        )
        axis.set_title(f"{method}: feasible |W|")
        axis.set_ylabel("Feature")
    axes[-1, 0].set_xlabel("School sorted by training response mean")
    if image is not None:
        figure.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.018, pad=0.02)
    figure.subplots_adjust(left=0.07, right=0.9, bottom=0.1, top=0.92, hspace=0.35)
    figure.savefig(output / "feasible_abs_W_heatmaps.png", dpi=180)
    plt.close(figure)


def _running_minimum(values: Sequence[float]) -> Array:
    return np.minimum.accumulate(np.asarray(values, dtype=float))


def _trajectory_error_rows(
    method: str,
    trajectory: Dict[str, Array],
    validation: TaskLossOperator,
    test: TaskLossOperator,
) -> List[Dict[str, object]]:
    validation_rmse = np.asarray(
        [math.sqrt(validation.mse(matrix)) for matrix in trajectory["W"]], dtype=float
    )
    test_rmse = np.asarray(
        [math.sqrt(test.mse(matrix)) for matrix in trajectory["W"]], dtype=float
    )
    validation_best = _running_minimum(validation_rmse)
    test_best = _running_minimum(test_rmse)
    return [
        {
            "method": METHOD_LABELS[method],
            "method_key": method,
            "iteration": int(iteration),
            "time": float(elapsed),
            "validation_rmse_raw": float(validation_value),
            "test_rmse_raw": float(test_value),
            "validation_rmse_best_so_far": float(validation_envelope),
            "test_rmse_best_so_far": float(test_envelope),
        }
        for iteration, elapsed, validation_value, test_value, validation_envelope, test_envelope in zip(
            trajectory["iteration"],
            trajectory["time"],
            validation_rmse,
            test_rmse,
            validation_best,
            test_best,
        )
    ]


def _plot_error_time_curves(
    output: Path,
    rows: Sequence[Dict[str, object]],
    cap_activation_time: Optional[float] = None,
) -> None:
    cache = Path(tempfile.gettempdir()) / "school_experiment_matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharex=True)
    styles = {
        "LDPM-CS": {"color": "#0072B2", "linestyle": "-", "linewidth": 2.0},
        "LDPM-CS-C": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.8},
    }
    for method in ("LDPM-CS-C", "LDPM-CS"):
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        elapsed = np.asarray([row["time"] for row in method_rows], dtype=float)
        validation_values = np.asarray(
            [row["validation_rmse_best_so_far"] for row in method_rows], dtype=float
        )
        test_values = np.asarray(
            [row["test_rmse_best_so_far"] for row in method_rows], dtype=float
        )
        axes[0].plot(elapsed, validation_values, label=method, **styles[method])
        axes[1].plot(elapsed, test_values, label=method, **styles[method])
    axes[0].set_title("Validation RMSE: best-so-far")
    axes[1].set_title("Test RMSE: best-so-far (post hoc)")
    for axis in axes:
        if cap_activation_time is not None:
            axis.axvline(
                cap_activation_time,
                color="#555555",
                linestyle=":",
                linewidth=1.2,
                label="cap active",
            )
        axis.set_xlabel("LDPM wall-clock time (s)")
        axis.set_ylabel("RMSE")
        axis.grid(True, color="#D0D0D0", linewidth=0.6, alpha=0.7)
        axis.legend(frameon=False)
    figure.suptitle(
        "School weighted-l1 + nuclear: non-increasing error envelopes\n"
        "Test labels are attached only after optimization; LDPM-CS-C uses its configured beta ceiling"
    )
    figure.tight_layout()
    figure.savefig(output / "best_so_far_error_vs_time.png", dpi=190)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("school.mat"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("results") / "seed2026_tol1em4_b1_q04_cap27",
    )
    parser.add_argument("--methods", default="ldpm,ldpm-capped")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--min-iter", type=int, default=1)
    parser.add_argument("--max-time", type=float, default=600.0)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument("--beta0", type=float, default=1.0)
    parser.add_argument("--beta-power", type=float, default=0.4)
    parser.add_argument("--beta-max", type=float, default=27.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--initial-step", type=float, default=0.1)
    parser.add_argument("--max-step", type=float, default=None)
    parser.add_argument("--line-search-growth", type=float, default=1.0)
    parser.add_argument("--beta-step-scale", type=float)
    parser.add_argument("--consensus-tol", type=float)
    parser.add_argument("--convergence-window", type=int, default=1)
    parser.add_argument("--lower-rho", type=float, default=0.1)
    parser.add_argument("--lower-max-iter", type=int, default=5000)
    parser.add_argument("--lower-abs-tol", type=float, default=1e-8)
    parser.add_argument("--lower-rel-tol", type=float, default=1e-7)
    parser.add_argument("--skip-heatmap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown = sorted(set(methods) - set(METHOD_LABELS))
    if unknown:
        raise ValueError(f"unsupported methods: {unknown}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_preprocess_school(args.data, args.seed)
    _save_split_and_preprocessing(args.output_dir, data)
    training = TaskLossOperator(data.train_a, data.train_b)
    validation = TaskLossOperator(data.validation_a, data.validation_b)
    test = TaskLossOperator(data.test_a, data.test_b)
    problem = SchoolLDPMProblem(training, validation)
    config = LDPMConfig(
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        tol=args.tol,
        beta0=args.beta0,
        beta_power=args.beta_power,
        beta_max=args.beta_max,
        gamma=args.gamma,
        initial_step=args.initial_step,
        max_step=args.initial_step if args.max_step is None else args.max_step,
        line_search_growth=args.line_search_growth,
        beta_step_scale=args.beta_step_scale,
        consensus_tol=args.consensus_tol,
        convergence_window=args.convergence_window,
        max_time=None if args.max_time <= 0 else args.max_time,
        record_interval=args.record_interval,
    )
    largest_uncapped_beta = args.beta0 * float(args.max_iter**args.beta_power)
    first_cap_iteration = (
        int(math.ceil((args.beta_max / args.beta0) ** (1.0 / args.beta_power)))
        if args.beta_power > 0.0 and args.beta_max > args.beta0
        else 1
    )
    protocol = {
        "dataset": "MALSAR School",
        "source": "https://github.com/jiayuzhou/MALSAR/blob/master/data/school.mat",
        "source_sha256": data.source_sha256,
        "tasks": 139,
        "features": 27,
        "hyperparameters": 27 * 139 + 1,
        "seed": args.seed,
        "split": "task-wise 60/20/20; floor for train and validation, remainder test",
        "sample_counts": {
            "train": training.n,
            "validation": validation.n,
            "test": test.n,
            "total": training.n + validation.n + test.n,
        },
        "preprocessing": (
            "constant intercept removed; pooled train-only standardization; "
            "task-wise train-only feature/response centering"
        ),
        "loss": "pooled one-half MSE internally",
        "reported_error": "pooled MSE; RMSE aliases also saved to reconcile the design document",
        "validation_error": "pooled validation MSE at the postprocessed fixed-lambda lower solution",
        "test_error": "pooled test MSE at the postprocessed fixed-lambda lower solution",
        "test_error_infeasibility": "pooled test MSE at the raw LDPM terminal W",
        "feasibility": "max(phi(lambda,W_raw)-v(lambda),0)/N_validation",
        "runtime": "LDPM optimizer only; fixed-lambda lower postprocessing excluded",
        "test_blind_optimization": True,
        "error_time_curve": {
            "raw": "RMSE at each saved raw LDPM W iterate",
            "display": "running minimum (best-so-far) of the raw RMSE sequence",
            "test_usage": "post hoc visualization only; test labels do not affect optimization, stopping, or selection",
        },
        "lambda_bounds": {
            "entrywise_upper": problem.lambda_l1_max,
            "nuclear_upper": problem.lambda_nuclear_max,
        },
        "ldpm_config": asdict(config),
        "lower_solver": {
            "method": "two-copy consensus ADMM",
            "rho": args.lower_rho,
            "max_iter": args.lower_max_iter,
            "abs_tol": args.lower_abs_tol,
            "rel_tol": args.lower_rel_tol,
        },
        "cap_reachability_note": (
            f"largest scheduled uncapped beta is {largest_uncapped_beta:.12g}; "
            f"capped beta_max is {args.beta_max:.12g}; first binding iteration is "
            f"approximately {first_cap_iteration}; cap is expected to activate: "
            f"{largest_uncapped_beta >= args.beta_max}"
        ),
    }
    _save_json(args.output_dir / "protocol.json", protocol)

    summaries: List[Dict[str, object]] = []
    error_curve_rows: List[Dict[str, object]] = []
    feasible_matrices: Dict[str, Array] = {}
    cap_activation_time: Optional[float] = None
    for method in methods:
        label = METHOD_LABELS[method]
        print(f"Running {label}...", flush=True)
        history, state, summary, trajectory = run_ldpm(problem, method, config)
        if method == "ldpm-capped":
            first_binding_row = next(
                (row for row in history if bool(row["cap_reached"])), None
            )
            if first_binding_row is not None:
                cap_activation_time = float(first_binding_row["time"])
        slug = method.replace("-", "_")
        _save_csv(args.output_dir / f"{slug}_history.csv", history)
        np.savez_compressed(args.output_dir / f"{slug}_state.npz", **state)
        np.savez_compressed(args.output_dir / f"{slug}_trajectory.npz", **trajectory)
        error_curve_rows.extend(
            _trajectory_error_rows(method, trajectory, validation, test)
        )
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
            abs_tol=args.lower_abs_tol,
            rel_tol=args.lower_rel_tol,
        )
        np.savez_compressed(
            args.output_dir / f"{slug}_feasible_state.npz",
            W=feasible_w,
            lambda_l1=lambda_l1,
            lambda_nuclear=np.asarray(lambda_nuclear),
        )
        raw_lower_objective = lower_objective(training, raw_w, lambda_l1, lambda_nuclear)
        objective_gap = raw_lower_objective - float(lower["lower_objective"])
        validation_mse = validation.mse(feasible_w)
        test_mse = test.mse(feasible_w)
        test_mse_infeasible = test.mse(raw_w)
        summary.update(
            {
                "seed": args.seed,
                "validation_error": validation_mse,
                "test_error": test_mse,
                "test_error_infeasibility": test_mse_infeasible,
                "feasibility": max(objective_gap, 0.0) / validation.n,
                "raw_lower_objective": raw_lower_objective,
                "feasible_lower_objective": float(lower["lower_objective"]),
                "raw_feasibility_gap": objective_gap,
                "validation_rmse": math.sqrt(validation_mse),
                "test_rmse": math.sqrt(test_mse),
                "test_rmse_infeasibility": math.sqrt(test_mse_infeasible),
                "postprocess_time": float(lower["time"]),
                "lower_status": lower["status"],
                "lower_iterations": lower["iterations"],
                "lower_primal_residual": lower["primal_residual"],
                "lower_dual_residual": lower["dual_residual"],
                "lower_primal_tolerance": lower["primal_tolerance"],
                "lower_dual_tolerance": lower["dual_tolerance"],
            }
        )
        _save_json(args.output_dir / f"{slug}_summary.json", summary)
        summaries.append(summary)
        feasible_matrices[label] = feasible_w
        print(
            f"{label}: status={summary['status']} time={summary['time']:.3f}s "
            f"val_mse={validation_mse:.6g} test_mse={test_mse:.6g} "
            f"test_mse_infeas={test_mse_infeasible:.6g} "
            f"feasibility={summary['feasibility']:.3e} cap={summary['cap_reached']}",
            flush=True,
        )

    _save_csv(args.output_dir / "summary.csv", summaries)
    _save_json(args.output_dir / "summary.json", summaries)
    _save_csv(args.output_dir / "error_time_curves.csv", error_curve_rows)
    _plot_error_time_curves(args.output_dir, error_curve_rows, cap_activation_time)
    if not args.skip_heatmap:
        _plot_heatmaps(args.output_dir, feasible_matrices, data.task_response_mean)


if __name__ == "__main__":
    main()
