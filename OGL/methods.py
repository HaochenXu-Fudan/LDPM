"""Methods shared by the splice and Tecator OGL experiments."""

from __future__ import annotations

import argparse
import contextlib
import csv
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - optional baseline dependency
    cp = None


Array = np.ndarray


@dataclass
class OGLData:
    x_train: Array
    y_train: Array
    x_val: Array
    y_val: Array
    x_test: Array
    y_test: Array
    groups: List[Array]


class MethodTimeout(RuntimeError):
    pass


@contextlib.contextmanager
def time_limit(seconds: Optional[float], label: str):
    if seconds is None or seconds <= 0:
        yield
        return

    def handler(signum, frame):  # noqa: ARG001
        raise MethodTimeout("%s timed out after %.1f seconds" % (label, seconds))

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous)


def project_l2_epigraph(v: Array, t: float) -> Tuple[Array, float]:
    v = np.asarray(v, dtype=float).copy()
    t = float(t)
    norm_v = float(np.linalg.norm(v))
    if norm_v <= t and t >= 0.0:
        return v, t
    if norm_v <= -t:
        return np.zeros_like(v), 0.0
    if norm_v <= 1e-14:
        return np.zeros_like(v), max(t, 0.0)
    alpha = 0.5 * (norm_v + t)
    return (alpha / norm_v) * v, alpha


def x_lambda_stop(x_new: Array, lam_new: Array, x_old: Array, lam_old: Array) -> float:
    return float(
        np.linalg.norm(x_new - x_old) / max(np.linalg.norm(x_old), 1.0)
        + np.linalg.norm(lam_new - lam_old) / max(np.linalg.norm(lam_old), 1.0)
    )


def _snapshot_values(value: Array) -> str:
    """Serialize a floating-point vector losslessly enough for CSV replay."""

    return ";".join(
        format(float(item), ".17g")
        for item in np.asarray(value, dtype=float).reshape(-1)
    )

def make_overlapping_groups(num_features, num_groups, group_size=None, stride=None):
    """Build a simple sliding-window family of overlapping feature groups."""
    num_features = int(num_features)
    num_groups = int(num_groups)
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    if group_size is None:
        group_size = int(np.ceil(2.0 * num_features / (num_groups + 1.0)))
    group_size = int(max(1, min(num_features, group_size)))

    if stride is None:
        if num_groups == 1:
            starts = np.array([0], dtype=int)
        else:
            starts = np.rint(np.linspace(0, max(num_features - group_size, 0), num_groups)).astype(int)
    else:
        stride = int(max(1, stride))
        starts = np.arange(num_groups, dtype=int) * stride
        starts = np.minimum(starts, max(num_features - 1, 0))

    groups = []
    seen = set()
    for start in starts:
        stop = min(num_features, int(start) + group_size)
        group = tuple(range(int(start), stop))
        if group and group not in seen:
            groups.append(np.array(group, dtype=int))
            seen.add(group)

    covered = set(np.concatenate(groups).tolist()) if groups else set()
    for index in range(num_features):
        if index not in covered:
            groups.append(np.array([index], dtype=int))
    return groups


def normalize_overlap_groups(
    groups,
    num_features,
    one_based=False,
    add_singletons=True,
    require_coverage=True,
):
    normalized = []
    covered = np.zeros(num_features, dtype=bool)
    for group in groups:
        arr = np.asarray(group, dtype=int).reshape(-1)
        if one_based:
            arr = arr - 1
        arr = np.unique(arr)
        if arr.size == 0:
            raise ValueError("overlap groups cannot contain empty groups")
        if np.any(arr < 0) or np.any(arr >= num_features):
            raise ValueError("overlap group index is out of range")
        normalized.append(arr)
        covered[arr] = True
    if add_singletons:
        for index in np.flatnonzero(~covered):
            normalized.append(np.array([index], dtype=int))
    elif require_coverage and not np.all(covered):
        raise ValueError("each feature must belong to at least one overlap group")
    return normalized

class DirectOGLProblem:
    def __init__(
        self,
        data: OGLData,
        loss_scale: str = "mean",
        rms_scale_operators: bool = False,
    ):
        self.data = data
        self.a_tr_eval = np.asarray(data.x_train, dtype=float)
        self.b_tr_eval = np.asarray(data.y_train, dtype=float)
        self.a_val_eval = np.asarray(data.x_val, dtype=float)
        self.b_val_eval = np.asarray(data.y_val, dtype=float)
        self.a_test_eval = np.asarray(data.x_test, dtype=float)
        self.b_test_eval = np.asarray(data.y_test, dtype=float)
        self.a_tr = self.a_tr_eval
        self.b_tr = self.b_tr_eval
        self.a_val = self.a_val_eval
        self.b_val = self.b_val_eval
        self.a_test = self.a_test_eval
        self.b_test = self.b_test_eval
        self.groups = [np.asarray(group, dtype=int) for group in data.groups]
        self.p = self.a_tr.shape[1]
        self.m = self.a_tr.shape[0]
        self.group_count = len(self.groups)
        self.group_lengths = np.asarray([len(group) for group in self.groups], dtype=int)
        self.rho_coordinates = np.concatenate(self.groups).astype(int, copy=False)
        self.rho_slices: List[slice] = []
        start = 0
        for size in self.group_lengths:
            self.rho_slices.append(slice(start, start + int(size)))
            start += int(size)
        self.rho_dim = start
        self.projection_buckets = []
        for size in np.unique(self.group_lengths):
            ids = np.flatnonzero(self.group_lengths == size)
            feature_columns = np.stack([self.groups[index] for index in ids])
            rho_columns = np.stack(
                [
                    np.arange(
                        self.rho_slices[index].start,
                        self.rho_slices[index].stop,
                        dtype=int,
                    )
                    for index in ids
                ]
            )
            self.projection_buckets.append((ids, feature_columns, rho_columns))
        if rms_scale_operators:
            if loss_scale != "mean":
                raise ValueError("RMS operator scaling requires loss_scale='mean'")
            # This is the same mean-squared objective under the equivalent
            # parameterization A <- A/sqrt(n), b <- b/sqrt(n).  It avoids an
            # unnecessarily ill-conditioned Fenchel-dual block while leaving
            # predictions and all reported metrics on the unscaled arrays.
            self.a_tr = self.a_tr_eval / np.sqrt(float(len(self.b_tr_eval)))
            self.b_tr = self.b_tr_eval / np.sqrt(float(len(self.b_tr_eval)))
            self.a_val = self.a_val_eval / np.sqrt(float(len(self.b_val_eval)))
            self.b_val = self.b_val_eval / np.sqrt(float(len(self.b_val_eval)))
            self.train_scale = 1.0
            self.val_scale = 1.0
        elif loss_scale == "sum":
            self.train_scale = 1.0
            self.val_scale = 1.0
        elif loss_scale == "mean":
            self.train_scale = 1.0 / len(self.b_tr)
            self.val_scale = 1.0 / len(self.b_val)
        else:
            raise ValueError("--loss-scale must be 'mean' or 'sum'")
        self.dual_scale = 1.0 / self.train_scale
        self.z_dim = self.p + self.group_count + self.rho_dim + self.group_count

    def beta(self, iteration: int, beta0: float, beta_power: float, beta_max: Optional[float]) -> float:
        value = float(beta0) * (1.0 + iteration) ** float(beta_power)
        if beta_max is not None:
            value = min(value, float(beta_max))
        return value

    def pack(self, x: Array, lam: Array, rho: Array, r: Array) -> Array:
        return np.concatenate([x, lam, rho, r])

    def unpack(self, z: Array) -> Tuple[Array, Array, Array, Array]:
        pos = 0
        x = z[pos : pos + self.p].copy()
        pos += self.p
        lam = z[pos : pos + self.group_count].copy()
        pos += self.group_count
        rho = z[pos : pos + self.rho_dim].copy()
        pos += self.rho_dim
        r = z[pos : pos + self.group_count].copy()
        return x, lam, rho, r

    def scatter_rho(self, rho: Array) -> Array:
        return np.bincount(
            self.rho_coordinates,
            weights=np.asarray(rho, dtype=float),
            minlength=self.p,
        ).astype(float, copy=False)

    def gather_groups(self, value: Array) -> Array:
        return np.asarray(value, dtype=float)[self.rho_coordinates]

    def group_norms_x(self, x: Array) -> Array:
        return np.asarray([np.linalg.norm(x[group]) for group in self.groups], dtype=float)

    def group_norms_rho(self, rho: Array) -> Array:
        return np.asarray([np.linalg.norm(rho[sl]) for sl in self.rho_slices], dtype=float)

    def h_value(self, xi: Array, rho: Array) -> Array:
        return self.a_tr.T @ xi + self.scatter_rho(rho)

    def psi_value(self, x: Array, lam: Array, rho: Array, r: Array, xi: Array) -> float:
        residual = self.a_tr @ x - self.b_tr
        h = self.h_value(xi, rho)
        return float(
            0.5 * self.train_scale * np.dot(residual, residual)
            + np.dot(lam, r)
            + 0.5 * self.dual_scale * np.dot(xi, xi)
            + np.dot(xi, self.b_tr)
            - np.dot(x, h)
            + 0.5 * np.dot(h, h)
        )

    def smooth_value(
        self,
        z: Array,
        xi: Array,
        u: Array,
        mu: Array,
        beta: float,
        gamma: float,
    ) -> float:
        x, lam, rho, r = self.unpack(z)
        val_res = self.a_val @ x - self.b_val
        psi = self.psi_value(x, lam, rho, r, xi)
        diff = u - z[None, :]
        consensus = float(np.sum(mu * diff) + 0.5 * gamma * np.sum(diff * diff))
        xi_quad = 0.5 * self.dual_scale * float(np.dot(xi, xi))
        # LDPM.pdf, LDP-ADMM subproblem: (1 / beta_k) * F_k =
        # L(x) / beta_k + psi(z), plus the *unscaled* consensus augmented
        # Lagrangian.  Writing L + beta*psi without also multiplying the
        # consensus term by beta changes the algorithm.
        return (
            0.5 * self.val_scale / beta * float(np.dot(val_res, val_res))
            + psi
            - xi_quad
            + consensus
        )

    def full_augmented_value(
        self,
        z: Array,
        xi: Array,
        u: Array,
        mu: Array,
        beta: float,
        gamma: float,
    ) -> float:
        x, lam, rho, r = self.unpack(z)
        val_res = self.a_val @ x - self.b_val
        diff = u - z[None, :]
        consensus = float(np.sum(mu * diff) + 0.5 * gamma * np.sum(diff * diff))
        return (
            0.5 * self.val_scale / beta * float(np.dot(val_res, val_res))
            + self.psi_value(x, lam, rho, r, xi)
            + consensus
        )

    def smooth_gradients(
        self,
        z: Array,
        xi: Array,
        u: Array,
        mu: Array,
        beta: float,
        gamma: float,
    ) -> Tuple[Array, Array, Array, Array, Array]:
        x, lam, rho, r = self.unpack(z)
        h = self.h_value(xi, rho)
        train_res = self.a_tr @ x - self.b_tr
        val_res = self.a_val @ x - self.b_val
        grad_x = self.val_scale / beta * (self.a_val.T @ val_res) + (
            self.train_scale * (self.a_tr.T @ train_res) - h
        )
        grad_lam = r.copy()
        grad_r = lam.copy()
        grad_rho = self.gather_groups(h - x)
        grad_xi_smooth = self.b_tr - self.a_tr @ x + self.a_tr @ h

        grad_z = self.pack(grad_x, grad_lam, grad_rho, grad_r)
        grad_z += -np.sum(mu, axis=0) + gamma * (self.group_count * z - np.sum(u, axis=0))
        grad_x, grad_lam, grad_rho, grad_r = self.unpack(grad_z)
        return grad_x, grad_lam, grad_rho, grad_r, grad_xi_smooth

    def prox_xi(self, xi: Array, grad_xi_smooth: Array, step: float, beta: float) -> Array:
        del beta  # kept in the signature so existing callers remain compatible
        return (xi - step * grad_xi_smooth) / (1.0 + step * self.dual_scale)

    def project_copy(self, value: Array, group_index: int) -> Array:
        x, lam, rho, r = self.unpack(value)
        group = self.groups[group_index]
        x[group], r[group_index] = project_l2_epigraph(x[group], r[group_index])
        for sl_index, sl in enumerate(self.rho_slices):
            rho[sl], lam[sl_index] = project_l2_epigraph(rho[sl], lam[sl_index])
        return self.pack(x, lam, rho, r)

    @staticmethod
    def project_l2_epigraph_batch(values: Array, radii: Array) -> Tuple[Array, Array]:
        values = np.asarray(values, dtype=float)
        radii = np.asarray(radii, dtype=float)
        norms = np.linalg.norm(values, axis=-1)
        projected_values = values.copy()
        projected_radii = radii.copy()
        inside = norms <= radii
        polar = (~inside) & (norms <= -radii)
        boundary = ~(inside | polar)
        if np.any(polar):
            projected_values[polar] = 0.0
            projected_radii[polar] = 0.0
        if np.any(boundary):
            boundary_norms = norms[boundary]
            alpha = 0.5 * (boundary_norms + radii[boundary])
            projected_values[boundary] *= (alpha / boundary_norms)[..., None]
            projected_radii[boundary] = alpha
        return projected_values, projected_radii

    def project_all_copies(self, z: Array, mu: Array, gamma: float) -> Array:
        # Every consensus copy has one primal group cone and all dual group
        # cones.  Batch the copies for each cone instead of invoking the scalar
        # projector group_count**2 times.  This is algebraically identical to
        # the original loop and materially reduces Python overhead.
        u_next = z[None, :] - mu / gamma
        lam_start = self.p
        lam_end = lam_start + self.group_count
        rho_start = lam_end
        r_start = rho_start + self.rho_dim
        for ids, feature_columns, rho_columns in self.projection_buckets:
            primal_values = u_next[ids[:, None], feature_columns]
            primal_radii = u_next[ids, r_start + ids]
            primal_values, primal_radii = self.project_l2_epigraph_batch(
                primal_values, primal_radii
            )
            u_next[ids[:, None], feature_columns] = primal_values
            u_next[ids, r_start + ids] = primal_radii

            dual_values = u_next[:, rho_start + rho_columns]
            dual_radii = u_next[:, lam_start + ids]
            dual_values, dual_radii = self.project_l2_epigraph_batch(
                dual_values, dual_radii
            )
            u_next[:, rho_start + rho_columns] = dual_values
            u_next[:, lam_start + ids] = dual_radii
        return u_next

    def metrics(
        self,
        x: Array,
        lam: Array,
        rho: Array,
        r: Array,
        xi: Array,
        iteration: int,
        elapsed: float,
        beta: float,
        step: float,
        stop_value: float,
        line_search_trials: int,
        include_test: bool = True,
    ) -> Dict[str, float]:
        train_res = self.a_tr_eval @ x - self.b_tr_eval
        val_res = self.a_val_eval @ x - self.b_val_eval
        x_norms = self.group_norms_x(x)
        rho_norms = self.group_norms_rho(rho)
        h = self.h_value(xi, rho)
        primal_violation = max(0.0, float(np.max(x_norms - r)), float(np.max(-r)))
        dual_violation = max(0.0, float(np.max(rho_norms - lam)), float(np.max(-lam)))
        result = {
            "iteration": iteration,
            "time": elapsed,
            "train_loss": 0.5 / len(self.b_tr) * float(np.dot(train_res, train_res)),
            "val_loss": 0.5 / len(self.b_val) * float(np.dot(val_res, val_res)),
            "x_lambda_stop": stop_value,
            "accepted_step": step,
            "line_search_trials": line_search_trials,
            "beta": beta,
            "psi": self.psi_value(x, lam, rho, r, xi),
            "h_norm": float(np.linalg.norm(h)),
            "primal_violation": primal_violation,
            "dual_violation": dual_violation,
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "lambda_mean": float(np.mean(lam)),
            "active_groups": int(np.sum(x_norms > 1e-6)),
            "sparsity": float(np.mean(np.abs(x) > 1e-6)),
        }
        if include_test:
            test_res = self.a_test_eval @ x - self.b_test_eval
            scores = self.a_test_eval @ x
            predictions = np.where(scores >= 0.0, 1.0, -1.0)
            result.update(
                {
                    "test_loss": 0.5 / len(self.b_test) * float(np.dot(test_res, test_res)),
                    "test_err": float(np.mean(predictions != self.b_test_eval)),
                }
            )
        return result

    def run_ldpm_cs(
        self,
        *,
        max_iter: int,
        tol: float,
        beta0: float,
        beta_power: float,
        beta_max: Optional[float],
        gamma: float,
        initial_lambda: float,
        initial_r: float,
        init_mode: str,
        init_ridge: float,
        init_dual: str,
        initial_step: float,
        max_step: float,
        min_step: float,
        line_search_decay: float,
        line_search_growth: float,
        max_line_search_iter: int,
        record_interval: int,
        psi_target: Optional[float],
        include_test: bool = True,
        stop_patience: int = 1,
        initial_state: Optional[Dict[str, Array]] = None,
        stop_mode: str = "x_lambda",
        record_snapshots: bool = False,
        max_time: Optional[float] = None,
    ) -> Tuple[List[Dict[str, float]], Dict[str, Array]]:
        if initial_state is not None:
            required = {"x", "lambda", "rho", "r", "xi"}
            missing = required.difference(initial_state)
            if missing:
                raise ValueError("initial_state is missing %s" % sorted(missing))
            x = np.asarray(initial_state["x"], dtype=float).reshape(self.p).copy()
            lam = np.asarray(initial_state["lambda"], dtype=float).reshape(self.group_count).copy()
            rho = np.asarray(initial_state["rho"], dtype=float).reshape(self.rho_dim).copy()
            r = np.asarray(initial_state["r"], dtype=float).reshape(self.group_count).copy()
            xi = np.asarray(initial_state["xi"], dtype=float).reshape(self.m).copy()
        else:
            if init_mode == "ridge":
                hessian = self.train_scale * (self.a_tr.T @ self.a_tr) + float(init_ridge) * np.eye(self.p)
                rhs = self.train_scale * (self.a_tr.T @ self.b_tr)
                try:
                    x = np.linalg.solve(hessian, rhs)
                except np.linalg.LinAlgError:
                    x = np.linalg.lstsq(hessian, rhs, rcond=None)[0]
            elif init_mode == "zero":
                x = np.zeros(self.p, dtype=float)
            else:
                raise ValueError("unknown LDPM init mode %r" % init_mode)
            lam = np.full(self.group_count, float(initial_lambda), dtype=float)
            rho = np.zeros(self.rho_dim, dtype=float)
            r = np.maximum(self.group_norms_x(x), float(initial_r))
            if init_dual == "zero":
                xi = np.zeros(self.m, dtype=float)
            elif init_dual == "fenchel":
                xi = self.train_scale * (self.a_tr @ x - self.b_tr)
                for i, (group, sl) in enumerate(zip(self.groups, self.rho_slices)):
                    norm_g = np.linalg.norm(x[group])
                    if norm_g > 1e-12:
                        rho[sl] = lam[i] * x[group] / norm_g
            else:
                raise ValueError("unknown LDPM dual init mode %r" % init_dual)
        z = self.pack(x, lam, rho, r)
        mu = np.zeros((self.group_count, self.z_dim), dtype=float)
        u = self.project_all_copies(z, mu, gamma)
        records: List[Dict[str, float]] = []
        start = time.time()
        step = float(initial_step)
        record_interval = max(1, int(record_interval))
        stop_patience = max(1, int(stop_patience))
        consecutive_stop_hits = 0
        first_hit_iteration: Optional[int] = None

        for k in range(max_iter):
            beta = self.beta(k, beta0, beta_power, beta_max)
            x_old = x.copy()
            lam_old = lam.copy()
            # The updates below rebind rather than mutate these arrays, so the
            # references preserve the previous state without a large concat or
            # copy on every iteration.
            z_before = z
            xi_before = xi
            u_before = u
            mu_before = mu
            grads = self.smooth_gradients(z, xi, u, mu, beta, gamma)
            grad_z = self.pack(grads[0], grads[1], grads[2], grads[3])
            grad_xi = grads[4]
            current_smooth = self.smooth_value(z, xi, u, mu, beta, gamma)
            trial_step = min(float(max_step), max(float(min_step), step * float(line_search_growth)))
            accepted = None
            trials = 0
            for trials in range(1, int(max_line_search_iter) + 1):
                z_trial = z - trial_step * grad_z
                xi_trial = self.prox_xi(xi, grad_xi, trial_step, beta)
                finite = np.all(np.isfinite(z_trial)) and np.all(np.isfinite(xi_trial))
                if finite:
                    trial_smooth = self.smooth_value(z_trial, xi_trial, u, mu, beta, gamma)
                    delta_z = z_trial - z
                    delta_xi = xi_trial - xi
                    rhs = (
                        current_smooth
                        + float(np.dot(grad_z, delta_z) + np.dot(grad_xi, delta_xi))
                        + 0.5 / trial_step * float(np.dot(delta_z, delta_z) + np.dot(delta_xi, delta_xi))
                        + 1e-12
                    )
                    if np.isfinite(trial_smooth) and trial_smooth <= rhs:
                        accepted = (z_trial, xi_trial, trial_step)
                        break
                trial_step *= float(line_search_decay)
                if trial_step < float(min_step):
                    break
            if accepted is None:
                z_trial = z - float(min_step) * grad_z
                xi_trial = self.prox_xi(xi, grad_xi, float(min_step), beta)
                accepted = (z_trial, xi_trial, float(min_step))
            z, xi, step = accepted
            x, lam, rho, r = self.unpack(z)
            u = self.project_all_copies(z, mu, gamma)
            mu = mu + gamma * (u - z[None, :])
            x_lambda_value = x_lambda_stop(x, lam, x_old, lam_old)
            x_lambda_mapping = x_lambda_value / max(float(step), float(min_step))
            def full_state_relative_change() -> float:
                numerator_sq = (
                    np.dot(z - z_before, z - z_before)
                    + np.dot(xi - xi_before, xi - xi_before)
                    + np.sum((u - u_before) ** 2)
                    + np.sum((mu - mu_before) ** 2)
                )
                denominator_sq = (
                    np.dot(z_before, z_before)
                    + np.dot(xi_before, xi_before)
                    + np.sum(u_before**2)
                    + np.sum(mu_before**2)
                )
                return float(np.sqrt(numerator_sq) / max(np.sqrt(denominator_sq), 1.0))

            full_state_value = (
                full_state_relative_change()
                if stop_mode == "full_state"
                else np.nan
            )
            consensus_value = max(
                (float(np.linalg.norm(copy - z)) for copy in u),
                default=0.0,
            ) / max(
                1.0,
                float(np.linalg.norm(z)),
                max((float(np.linalg.norm(copy)) for copy in u), default=0.0),
            )
            if stop_mode == "x_lambda":
                stop_value = x_lambda_value
            elif stop_mode == "x_lambda_mapping":
                stop_value = x_lambda_mapping
            elif stop_mode == "full_state":
                stop_value = full_state_value
            else:
                raise ValueError("unknown stop_mode %r" % stop_mode)
            psi_value = (
                self.psi_value(x, lam, rho, r, xi)
                if psi_target is not None
                else np.nan
            )
            psi_stop = psi_target is not None and abs(psi_value) <= float(psi_target)
            residual_hit = bool(np.isfinite(stop_value) and stop_value <= tol)
            if stop_mode == "full_state":
                residual_hit = residual_hit and consensus_value <= tol
            if residual_hit:
                if first_hit_iteration is None:
                    first_hit_iteration = k + 1
                consecutive_stop_hits += 1
            else:
                consecutive_stop_hits = 0
            confirmed_residual_stop = consecutive_stop_hits >= stop_patience
            elapsed = time.time() - start
            budget_reached = bool(
                max_time is not None and elapsed >= float(max_time)
            )
            should_stop = (
                (not np.isfinite(stop_value))
                or confirmed_residual_stop
                or psi_stop
                or budget_reached
            )
            should_record = should_stop or ((k + 1) % record_interval == 0) or (k + 1 == max_iter)
            if should_record:
                if not np.isfinite(full_state_value):
                    full_state_value = full_state_relative_change()
                record = self.metrics(
                    x,
                    lam,
                    rho,
                    r,
                    xi,
                    k + 1,
                    elapsed,
                    beta,
                    step,
                    stop_value,
                    trials,
                    include_test=include_test,
                )
                record["x_lambda_stop"] = x_lambda_value
                record["x_lambda_mapping"] = x_lambda_mapping
                record["z_stop"] = full_state_value
                record["consensus_residual"] = consensus_value
                record["stop_value"] = stop_value
                record["stop_metric"] = stop_mode
                record.update(
                    {
                        "first_hit_iteration": first_hit_iteration,
                        "stop_consecutive_hits": consecutive_stop_hits,
                        "stop_patience": stop_patience,
                        "confirmed_residual_stop": confirmed_residual_stop,
                        "time_budget_reached": budget_reached,
                    }
                )
                if record_snapshots:
                    record["x_values"] = _snapshot_values(x)
                    record["lambda_values"] = _snapshot_values(lam)
                records.append(record)
            if should_stop:
                break

        state = {
            "x": x.copy(),
            "lambda": lam.copy(),
            "rho": rho.copy(),
            "r": r.copy(),
            "xi": xi.copy(),
            "u": u.copy(),
            "mu": mu.copy(),
        }
        return records, state

    def evaluate_x(self, x: Array) -> Dict[str, float]:
        train_res = self.a_tr_eval @ x - self.b_tr_eval
        val_res = self.a_val_eval @ x - self.b_val_eval
        test_res = self.a_test_eval @ x - self.b_test_eval
        predictions = np.where((self.a_test_eval @ x) >= 0.0, 1.0, -1.0)
        return {
            "train_loss": 0.5 / len(self.b_tr) * float(np.dot(train_res, train_res)),
            "val_loss": 0.5 / len(self.b_val) * float(np.dot(val_res, val_res)),
            "test_loss": 0.5 / len(self.b_test) * float(np.dot(test_res, test_res)),
            "test_err": float(np.mean(predictions != self.b_test_eval)),
        }


def require_cvxpy() -> None:
    if cp is None:
        raise ImportError("cvxpy is required for VF-iDCA and LDMMA baselines")


def solve_cvxpy(problem, args: argparse.Namespace):
    solver = str(args.solver).upper()
    if solver == "ECOS" and cp is not None:
        try:
            return problem.solve(
                solver=cp.ECOS,
                abstol=args.solver_tol,
                reltol=args.solver_tol,
                max_iters=args.solver_max_iters,
                verbose=args.cvxpy_verbose,
            )
        except Exception:
            solver = "SCS"
    if solver == "SCS":
        return problem.solve(
            solver=cp.SCS,
            eps=args.solver_tol,
            max_iters=args.solver_max_iters,
            verbose=args.cvxpy_verbose,
        )
    return problem.solve(solver=solver, verbose=args.cvxpy_verbose)

def _make_dual_incidence(problem: DirectOGLProblem) -> Array:
    incidence = np.zeros((problem.p, problem.rho_dim), dtype=float)
    for group, sl in zip(problem.groups, problem.rho_slices):
        for local_pos, feature in enumerate(group):
            incidence[int(feature), sl.start + local_pos] = 1.0
    return incidence


def run_ldmma(
    problem: DirectOGLProblem,
    args: argparse.Namespace,
    record_snapshots: bool = False,
) -> Tuple[List[Dict[str, float]], Dict[str, Array]]:
    require_cvxpy()
    p = problem.p
    n = problem.m
    m = problem.group_count
    incidence = _make_dual_incidence(problem)

    x_var = cp.Variable(p)
    lam_var = cp.Variable(m, nonneg=True)
    r_var = cp.Variable(m, nonneg=True)
    rho_var = cp.Variable(problem.rho_dim)
    w_var = cp.Variable(n)
    coff_r = cp.Parameter(m, nonneg=True)
    coff_lam = cp.Parameter(m, nonneg=True)
    x_k = cp.Parameter(p)
    lam_k = cp.Parameter(m, nonneg=True)
    r_k = cp.Parameter(m, nonneg=True)
    rho_k = cp.Parameter(problem.rho_dim)

    val_loss = 0.5 * problem.val_scale * cp.sum_squares(problem.a_val @ x_var - problem.b_val)
    prox = (
        cp.sum_squares(x_var - x_k)
        + cp.sum_squares(lam_var - lam_k)
        + cp.sum_squares(r_var - r_k)
        + cp.sum_squares(rho_var - rho_k)
    )
    constraints = [
        problem.a_tr.T @ w_var + incidence @ rho_var == 0,
    ]
    constraints += [cp.norm(x_var[group], 2) <= r_var[i] for i, group in enumerate(problem.groups)]
    constraints += [
        cp.norm(rho_var[sl], 2) <= lam_var[i] for i, sl in enumerate(problem.rho_slices)
    ]
    train_term = 0.5 * problem.train_scale * cp.sum_squares(problem.a_tr @ x_var - problem.b_tr)
    dual_term = 0.5 * problem.dual_scale * cp.sum_squares(w_var + problem.train_scale * problem.b_tr)
    const_term = 0.5 * problem.train_scale * float(np.dot(problem.b_tr, problem.b_tr))
    majorizer = 0.5 * cp.sum_squares(cp.multiply(coff_r, r_var)) + 0.5 * cp.sum_squares(
        cp.multiply(coff_lam, lam_var)
    )
    constraints.append(train_term + dual_term - const_term + majorizer <= args.ldmma_epsilon)
    objective = val_loss + 0.5 * args.ldmma_eta * prox
    approx_problem = cp.Problem(cp.Minimize(objective), constraints)

    x = np.zeros(p, dtype=float)
    lam = np.full(m, float(args.initial_lambda), dtype=float)
    r = np.full(m, float(args.initial_r), dtype=float)
    rho = np.zeros(problem.rho_dim, dtype=float)
    records: List[Dict[str, float]] = []
    start = time.time()

    for k in range(args.ldmma_max_iter):
        if getattr(args, "max_runtime", None) is not None:
            elapsed = time.time() - start
            if elapsed >= float(args.max_runtime):
                if records:
                    records[-1]["time_budget_reached"] = True
                    break
                raise MethodTimeout("LDMMA reached the total time limit before one iteration")
        x_old = x.copy()
        lam_old = lam.copy()
        lam_floor = np.maximum(lam, 1e-8)
        r_floor = np.maximum(r, 1e-8)
        coff_r.value = np.sqrt(lam_floor / r_floor)
        coff_lam.value = np.sqrt(r_floor / lam_floor)
        x_k.value = x
        lam_k.value = lam_floor
        r_k.value = r_floor
        rho_k.value = rho
        per_iteration_limit = getattr(args, "baseline_timeout", None)
        if per_iteration_limit is None:
            solve_cvxpy(approx_problem, args)
        else:
            with time_limit(float(per_iteration_limit), "LDMMA outer iteration"):
                solve_cvxpy(approx_problem, args)
        if x_var.value is None or lam_var.value is None or r_var.value is None or rho_var.value is None:
            raise RuntimeError("LDMMA problem failed with status %s" % approx_problem.status)
        x = np.asarray(x_var.value, dtype=float).reshape(-1)
        lam = np.maximum(np.asarray(lam_var.value, dtype=float).reshape(-1), 0.0)
        r = np.maximum(np.asarray(r_var.value, dtype=float).reshape(-1), 0.0)
        rho = np.asarray(rho_var.value, dtype=float).reshape(-1)
        stop_value = x_lambda_stop(x, lam, x_old, lam_old)
        metrics = problem.evaluate_x(x)
        record = {
            "iteration": k + 1,
            "time": time.time() - start,
            **metrics,
            "x_lambda_stop": stop_value,
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "lambda_mean": float(np.mean(lam)),
            "time_budget_reached": False,
        }
        if record_snapshots:
            record["x_values"] = _snapshot_values(x)
            record["lambda_values"] = _snapshot_values(lam)
        records.append(record)
        if not np.isfinite(stop_value) or stop_value <= args.tol:
            break

    return records, {"x": x.copy(), "lambda": lam.copy(), "r": r.copy(), "rho": rho.copy()}


def write_history(path: Path, records: List[Dict[str, float]]) -> None:
    if not records:
        return
    keys = list(records[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key, "") for key in keys})


def save_state(path: Path, state: Dict[str, Array]) -> None:
    np.savez(path, **{key: np.asarray(value) for key, value in state.items()})

UPSTREAM_REPOSITORY = "https://github.com/SUSTech-Optimization/VF-iDCA"
UPSTREAM_COMMIT = "4abfed955443140eca7833d72b864707867ff880"
UPSTREAM_SOURCE_FILES = (
    "VF_iDCA.py",
    "wLasso.py",
    "experiments/SGL_Experiments.py",
    "utils/SGL_Algorithms.py",
)

# Literal defaults from the repository-root VF_iDCA.py and the rho used by its
# wLasso.py model example.  This is kept separate from the more comparable SGL
# experiment configuration below.
UPSTREAM_GENERIC_FIXED_CONFIG = {
    "initial_guess": 1.0,
    "alpha": 1.0,
    "c_alpha": 1.0,
    "delta": 5.0,
    "rho": 1.0,
    "MAX_ITERATION": 10,
    "lower_ecos_options": "ECOS defaults",
    "approx_ecos_options": "ECOS defaults",
}

# Exact SGL_Experiments.py settings, except that TOL is supplied by the splice
# command line because this experiment must be run at both 1e-4 and 1e-5.
UPSTREAM_SGL_FIXED_CONFIG = {
    "initial_guess": 10.0,
    "epsilon": 0.0,
    "beta_0": 1.0,
    "rho": 0.1,
    "MAX_ITERATION": 50,
    "c": 0.01,
    "delta": 5.0,
    "violation_weight": 100.0,
    "lower_ecos_tol": 1e-4,
    "lower_ecos_max_iters": 100,
    "approx_ecos_options": "ECOS defaults",
}


def _losses(problem, x: np.ndarray) -> Dict[str, float]:
    train_residual = problem.a_tr @ x - problem.b_tr
    val_residual = problem.a_val @ x - problem.b_val
    test_residual = problem.a_test @ x - problem.b_test
    prediction = np.where(problem.a_test @ x >= 0.0, 1.0, -1.0)
    return {
        "train_loss": 0.5 / len(problem.b_tr) * float(train_residual @ train_residual),
        "val_loss": 0.5 / len(problem.b_val) * float(val_residual @ val_residual),
        "test_loss": 0.5 / len(problem.b_test) * float(test_residual @ test_residual),
        "test_err": float(np.mean(prediction != problem.b_test)),
    }


def _pair_step_upstream(
    x: np.ndarray,
    r: np.ndarray,
    x_next: np.ndarray,
    r_next: np.ndarray,
) -> float:
    """Copy the SGL iP_DCA ``iteration_err`` formula exactly.

    In particular, the denominator uses the *current* pair and has no added
    constant.  The official SGL initialization has r=10, so it is nonzero.
    """

    numerator = np.sqrt(np.sum((x - x_next) ** 2) + np.sum((r - r_next) ** 2))
    denominator = np.sqrt(np.sum(x**2) + np.sum(r**2))
    return float(numerator / denominator)


def _maximum(expressions):
    result = expressions[0]
    for expression in expressions[1:]:
        result = cp.maximum(result, expression)
    return result


def run_vfidca_upstream_sgl_adapter(
    problem,
    tol: float,
    record_snapshots: bool = False,
    initial_r=None,
    initial_x=None,
    max_iter=None,
    max_time=None,
) -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray]]:
    """Run the pinned upstream SGL iP-DCA logic on direct OGL constraints.

    ``initial_r=None`` preserves the upstream synthetic SGL example's
    ``initial_guess=10``.
    An explicit override may be either a positive scalar, applied to every
    group, or a positive vector with one entry per group.  All other upstream
    hyperparameters and the native stopping rule remain unchanged.
    """

    if cp is None:
        raise ImportError("cvxpy is required for the upstream VF-iDCA baseline")
    if "ECOS" not in cp.installed_solvers():
        raise RuntimeError(
            "The pinned VF-iDCA code uses ECOS, but ECOS is unavailable. "
            "Install it in an isolated path and add that path to PYTHONPATH."
        )

    p = problem.p
    group_count = problem.group_count
    config = UPSTREAM_SGL_FIXED_CONFIG

    # DC_lower from utils/SGL_Algorithms.py, with direct overlapping groups.
    x_lower = cp.Variable(p)
    r_lower = cp.Parameter(group_count, nonneg=True)
    lower_constraints = [
        cp.norm(x_lower[group], 2) <= r_lower[i]
        for i, group in enumerate(problem.groups)
    ]
    lower_loss = 0.5 / len(problem.b_tr) * cp.sum_squares(
        problem.b_tr - problem.a_tr @ x_lower
    )
    lower_problem = cp.Problem(cp.Minimize(lower_loss), lower_constraints)

    # DC_approximated from utils/SGL_Algorithms.py.  The max-penalty, beta
    # update, proximal coefficient, and solve options are kept unchanged.
    x_upper = cp.Variable(p)
    r_upper = cp.Variable(group_count)
    x_k = cp.Parameter(p)
    r_k = cp.Parameter(group_count, pos=True)
    gamma_k = cp.Parameter(group_count)
    bias_k = cp.Parameter()
    beta_k = cp.Parameter(pos=True)

    upper_loss = 0.5 / len(problem.b_val) * cp.sum_squares(
        problem.b_val - problem.a_val @ x_upper
    )
    prox = cp.sum_squares(x_upper - x_k) + cp.sum_squares(r_upper - r_k)
    train_loss = 0.5 / len(problem.b_tr) * cp.sum_squares(
        problem.b_tr - problem.a_tr @ x_upper
    )
    value_violation = (
        beta_k * train_loss
        + gamma_k @ r_upper
        - bias_k
        - beta_k * config["epsilon"]
    )
    primal_violation = _maximum(
        [
            cp.norm(x_upper[group], 2) - r_upper[i]
            for i, group in enumerate(problem.groups)
        ]
    )
    raw_penalty = cp.maximum(
        0.0,
        cp.maximum(
            value_violation,
            config["violation_weight"] * beta_k * primal_violation,
        ),
    )
    approx_objective = upper_loss + 0.5 * config["rho"] * prox + raw_penalty
    approx_problem = cp.Problem(cp.Minimize(approx_objective), [r_upper >= 0])

    if initial_x is None:
        x = np.zeros(p, dtype=float)
    else:
        x = np.asarray(initial_x, dtype=float).reshape(-1).copy()
        if x.size != p:
            raise ValueError("initial_x must have length %d; got %d" % (p, x.size))
        if not np.all(np.isfinite(x)):
            raise ValueError("initial_x entries must be finite")
    if initial_r is None:
        r = np.full(group_count, config["initial_guess"], dtype=float)
    else:
        initial_r_array = np.asarray(initial_r, dtype=float)
        if initial_r_array.ndim == 0:
            r = np.full(group_count, float(initial_r_array), dtype=float)
        else:
            r = initial_r_array.reshape(-1).copy()
            if r.size != group_count:
                raise ValueError(
                    "initial_r must be a scalar or have length %d; got %d"
                    % (group_count, r.size)
                )
        if not np.all(np.isfinite(r)) or np.any(r <= 0.0):
            raise ValueError("initial_r entries must be finite and strictly positive")
    beta = float(config["beta_0"])
    last_gamma = np.zeros(group_count, dtype=float)
    last_lower_x = x.copy()
    records: List[Dict[str, float]] = []
    start = time.time()

    iteration_limit = int(config["MAX_ITERATION"] if max_iter is None else max_iter)
    if iteration_limit <= 0:
        raise ValueError("max_iter must be positive")
    for iteration in range(iteration_limit):
        r_lower.value = r
        lower_objective = lower_problem.solve(
            solver=cp.ECOS,
            abstol=config["lower_ecos_tol"],
            reltol=config["lower_ecos_tol"],
            abstol_inacc=config["lower_ecos_tol"],
            reltol_inacc=config["lower_ecos_tol"],
            max_iters=config["lower_ecos_max_iters"],
        )
        if x_lower.value is None:
            raise RuntimeError(
                "VF-iDCA lower problem failed with status %s" % lower_problem.status
            )
        last_lower_x = np.asarray(x_lower.value, dtype=float).reshape(-1)
        last_gamma = np.asarray(
            [float(constraint.dual_value) for constraint in lower_constraints],
            dtype=float,
        )

        x_k.value = x
        r_k.value = r
        beta_k.value = beta
        gamma_k.value = beta * last_gamma
        bias_k.value = beta * float(lower_objective) + gamma_k.value @ r_k.value
        approx_problem.solve(solver=cp.ECOS, verbose=False)
        if x_upper.value is None or r_upper.value is None:
            raise RuntimeError(
                "VF-iDCA approximate problem failed with status %s"
                % approx_problem.status
            )

        x_next = np.asarray(x_upper.value, dtype=float).reshape(-1)
        r_next = np.maximum(
            np.asarray(r_upper.value, dtype=float).reshape(-1), 0.0
        )
        step_err = _pair_step_upstream(x, r, x_next, r_next)
        raw_penalty_value = float(np.asarray(raw_penalty.value))
        penalty = raw_penalty_value / beta
        converged = bool(step_err < tol and penalty < tol)

        metrics = _losses(problem, x_next)
        lower_metrics = _losses(problem, last_lower_x)
        elapsed = time.time() - start
        budget_reached = bool(max_time is not None and elapsed >= float(max_time))
        record = {
            "iteration": iteration + 1,
            "time": elapsed,
            **metrics,
            "lower_train_loss": lower_metrics["train_loss"],
            "lower_val_loss": lower_metrics["val_loss"],
            "lower_test_loss": lower_metrics["test_loss"],
            "diff_xk_xtilde": float(np.linalg.norm(x - last_lower_x)),
            "diff_xkp_xtilde": float(np.linalg.norm(x_next - last_lower_x)),
            "step_err": step_err,
            "penalty": penalty,
            "raw_penalty": raw_penalty_value,
            "upstream_stop_value": max(step_err, penalty),
            "native_converged": converged,
            "time_budget_reached": budget_reached,
            "beta": beta,
            "lambda_min": float(np.min(last_gamma)),
            "lambda_max": float(np.max(last_gamma)),
            "lambda_mean": float(np.mean(last_gamma)),
        }
        if record_snapshots:
            record["x_values"] = _snapshot_values(x_next)
            record["lambda_values"] = _snapshot_values(last_gamma)
        records.append(record)

        # Exact stopping and penalty update order from the upstream SGL code.
        if converged or budget_reached:
            x, r = x_next, r_next
            break
        if step_err * beta <= config["c"] * min(1.0, raw_penalty_value):
            beta += config["delta"]
        x, r = x_next, r_next

    state = {
        "x": x.copy(),
        "lambda": last_gamma.copy(),
        "r": r.copy(),
        "lower_x": last_lower_x.copy(),
        "beta": np.asarray([beta], dtype=float),
    }
    return records, state


def run_vfidca_upstream_generic_adapter(problem, tol: float) -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray]]:
    """Use the literal VF_iDCA.py/wLasso.py defaults with OGL constraints.

    The upstream generic driver has only ten iterations and is intentionally
    not extended here.  This is a literal-code baseline; the SGL adapter above
    remains the closer group-regularization comparison.
    """

    if cp is None:
        raise ImportError("cvxpy is required for the upstream VF-iDCA baseline")
    if "ECOS" not in cp.installed_solvers():
        raise RuntimeError("The pinned VF-iDCA code uses ECOS, but ECOS is unavailable")

    p = problem.p
    group_count = problem.group_count
    config = UPSTREAM_GENERIC_FIXED_CONFIG

    x_lower = cp.Variable(p)
    r_lower = cp.Parameter(group_count, nonneg=True)
    lower_constraints = [
        cp.norm(x_lower[group], 2) <= r_lower[i]
        for i, group in enumerate(problem.groups)
    ]
    lower_loss = 0.5 / len(problem.b_tr) * cp.sum_squares(
        problem.b_tr - problem.a_tr @ x_lower
    )
    lower_problem = cp.Problem(cp.Minimize(lower_loss), lower_constraints)

    x_upper = cp.Variable(p)
    r_upper = cp.Variable(group_count)
    x_k = cp.Parameter(p)
    r_k = cp.Parameter(group_count, nonneg=True)
    lower_value_k = cp.Parameter()
    gamma_k = cp.Parameter(group_count)
    alpha_k = cp.Parameter(nonneg=True)
    upper_loss = 0.5 / len(problem.b_val) * cp.sum_squares(
        problem.b_val - problem.a_val @ x_upper
    )
    train_loss = 0.5 / len(problem.b_tr) * cp.sum_squares(
        problem.b_tr - problem.a_tr @ x_upper
    )
    value_violation = train_loss - lower_value_k + gamma_k @ (r_upper - r_k)
    violation = _maximum(
        [0.0, value_violation]
        + [
            cp.norm(x_upper[group], 2) - r_upper[i]
            for i, group in enumerate(problem.groups)
        ]
    )
    prox = cp.sum_squares(x_upper - x_k) + cp.sum_squares(r_upper - r_k)
    approx_problem = cp.Problem(
        cp.Minimize(upper_loss + 0.5 * config["rho"] * prox + alpha_k * violation),
        [r_upper >= 0],
    )

    x = np.zeros(p, dtype=float)
    r = np.full(group_count, config["initial_guess"], dtype=float)
    alpha = float(config["alpha"])
    last_gamma = np.zeros(group_count, dtype=float)
    records: List[Dict[str, float]] = []
    start = time.time()

    for iteration in range(int(config["MAX_ITERATION"])):
        r_lower.value = r
        lower_objective = lower_problem.solve(solver=cp.ECOS)
        if x_lower.value is None:
            raise RuntimeError(
                "VF-iDCA generic lower problem failed with status %s"
                % lower_problem.status
            )
        last_gamma = np.asarray(
            [float(constraint.dual_value) for constraint in lower_constraints],
            dtype=float,
        )

        x_k.value = x
        r_k.value = r
        lower_value_k.value = float(lower_objective)
        gamma_k.value = last_gamma
        alpha_k.value = alpha
        approx_problem.solve(solver=cp.ECOS)
        if x_upper.value is None or r_upper.value is None:
            raise RuntimeError(
                "VF-iDCA generic approximation failed with status %s"
                % approx_problem.status
            )

        x_next = np.asarray(x_upper.value, dtype=float).reshape(-1)
        r_next = np.maximum(np.asarray(r_upper.value, dtype=float).reshape(-1), 0.0)
        numerator = np.sqrt(
            np.sum((x - x_next) ** 2) + np.sum((r - r_next) ** 2)
        )
        denominator = np.sqrt(1.0 + np.sum(x**2) + np.sum(r**2))
        step_err = float(numerator / denominator)
        penalty = float(np.asarray(violation.value))
        converged = bool(step_err < tol and penalty < tol)
        metrics = _losses(problem, x_next)
        records.append(
            {
                "iteration": iteration + 1,
                "time": time.time() - start,
                **metrics,
                "step_err": step_err,
                "penalty": penalty,
                "upstream_stop_value": max(step_err, penalty),
                "native_converged": converged,
                "alpha": alpha,
                "lambda_min": float(np.min(last_gamma)),
                "lambda_max": float(np.max(last_gamma)),
                "lambda_mean": float(np.mean(last_gamma)),
            }
        )
        if converged:
            x, r = x_next, r_next
            break
        if step_err * alpha <= config["c_alpha"] * min(1.0, alpha * penalty):
            alpha += config["delta"]
        x, r = x_next, r_next

    return records, {
        "x": x.copy(),
        "lambda": last_gamma.copy(),
        "r": r.copy(),
        "alpha": np.asarray([alpha], dtype=float),
    }
