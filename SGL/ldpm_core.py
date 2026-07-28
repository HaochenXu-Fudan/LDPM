import time

import numpy as np
import pandas as pd


def train_error(settings, data, x):
    return 0.5 / settings.num_train * np.sum((data.y_train - data.X_train @ x) ** 2)


def validation_error(settings, data, x):
    return 0.5 / settings.num_validate * np.sum((data.y_validate - data.X_validate @ x) ** 2)


def test_error(settings, data, x):
    return 0.5 / settings.num_test * np.sum((data.y_test - data.X_test @ x) ** 2)


def _soft_threshold(x, tau):
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def _relative_step(new, old):
    """r_stat = ||z^{k+1}-z^k|| / max(1, ||z^k||)."""

    return float(np.linalg.norm(new - old) / max(1.0, np.linalg.norm(old)))


def _x_lambda_max_relative_stop(x_new, lam_new, x_old, lam_old):
    x_den = max(np.linalg.norm(x_old), 1.0)
    lam_den = max(np.linalg.norm(lam_old), 1.0)
    return float(np.linalg.norm(x_new - x_old) / x_den + np.linalg.norm(lam_new - lam_old) / lam_den)


def _blockwise_max_relative_stop(new_blocks, old_blocks):
    """Maximum relative change across corresponding state blocks."""

    new_blocks = tuple(new_blocks)
    old_blocks = tuple(old_blocks)
    if len(new_blocks) != len(old_blocks):
        raise ValueError("new_blocks and old_blocks must have the same length")
    relative_stops = []
    for new_block, old_block in zip(new_blocks, old_blocks):
        new_array = np.asarray(new_block, dtype=float)
        old_array = np.asarray(old_block, dtype=float)
        if new_array.shape != old_array.shape:
            raise ValueError("corresponding state blocks must have the same shape")
        relative_stop = float(
            np.linalg.norm(new_array - old_array)
            / max(1.0, np.linalg.norm(old_array))
        )
        if not np.isfinite(relative_stop):
            return np.nan
        relative_stops.append(relative_stop)
    return float(np.max(np.asarray(relative_stops))) if relative_stops else 0.0


def _select_stationarity_stop(
    metric,
    full_z_stop,
    tilde_z_stop,
    x_lambda_stop,
    blockwise_stop=None,
):
    """Select an audited LDPM stationarity residual.

    ``full_z`` is the paper variable z=(tilde_z, xi), while ``tilde_z`` keeps
    the historical projected-block residual for backward-compatible audits.
    ``blockwise`` is the largest separately normalized state-block change.
    """

    aliases = {
        "full_z": "full_z",
        "full_relative_step": "full_z",
        "r_stat": "full_z",
        "x_lambda": "x_lambda",
        "tilde_z": "tilde_z",
        "packed_relative_step": "tilde_z",
        "blockwise": "blockwise",
    }
    key = aliases.get(str(metric))
    if key is None:
        raise ValueError("unknown LDPM stop metric %r" % metric)
    if key == "full_z":
        return float(full_z_stop)
    if key == "x_lambda":
        return float(x_lambda_stop)
    if key == "blockwise":
        if blockwise_stop is None:
            raise ValueError("blockwise stop metric requires blockwise_stop")
        return float(blockwise_stop)
    return float(tilde_z_stop)


def _spectral_norm_squared(a, num_iter=30):
    n = a.shape[1]
    rng = np.random.default_rng(0)
    v = rng.normal(size=n)
    v /= max(np.linalg.norm(v), 1e-12)
    for _ in range(num_iter):
        v = a.T @ (a @ v)
        v_norm = np.linalg.norm(v)
        if v_norm <= 1e-12:
            return 0.0
        v /= v_norm
    av = a @ v
    return float(np.dot(av, av))


def project_l2_epigraph(v, t):
    v = np.asarray(v, dtype=float).copy()
    t = float(np.asarray(t))
    norm_v = np.linalg.norm(v)
    if norm_v <= t and t >= 0.0:
        return v, t
    if norm_v <= -t:
        return np.zeros_like(v), 0.0
    if norm_v <= 1e-12:
        return np.zeros_like(v), max(t, 0.0)
    alpha = 0.5 * (norm_v + t)
    return alpha / norm_v * v, alpha


def project_l1_epigraph(v, t):
    v = np.asarray(v, dtype=float).copy()
    t = float(np.asarray(t))
    abs_v = np.abs(v)
    if np.sum(abs_v) <= t and t >= 0.0:
        return v, t
    high = max(float(np.max(abs_v)) if abs_v.size else 0.0, -t, 1.0)
    while np.sum(np.maximum(abs_v - high, 0.0)) - (t + high) > 0.0:
        high *= 2.0
    low = 0.0
    for _ in range(70):
        mid = 0.5 * (low + high)
        value = np.sum(np.maximum(abs_v - mid, 0.0)) - (t + mid)
        if value > 0.0:
            low = mid
        else:
            high = mid
    gamma = high
    return _soft_threshold(v, gamma), max(t + gamma, 0.0)


def project_linf_epigraph(v, t):
    v = np.asarray(v, dtype=float).copy()
    t = float(np.asarray(t))
    abs_v = np.abs(v)
    max_v = float(np.max(abs_v)) if abs_v.size else 0.0
    if max_v <= t and t >= 0.0:
        return v, t

    def derivative(tau):
        active = abs_v > tau
        return tau - t + np.sum(tau - abs_v[active])

    if derivative(0.0) >= 0.0:
        tau = 0.0
    else:
        low = 0.0
        high = max(max_v, t, 1.0)
        while derivative(high) < 0.0:
            high *= 2.0
        for _ in range(70):
            mid = 0.5 * (low + high)
            if derivative(mid) < 0.0:
                low = mid
            else:
                high = mid
        tau = high
    return np.clip(v, -tau, tau), tau


def project_squared_l2_epigraph(v, t):
    v = np.asarray(v, dtype=float).copy()
    t = float(np.asarray(t))
    norm2 = float(np.dot(v, v))
    if norm2 <= 2.0 * t and t >= 0.0:
        return v, t
    low = max(0.0, -t)

    def residual(mu):
        return 0.5 * norm2 / (1.0 + mu) ** 2 - t - mu

    high = max(1.0, low)
    while residual(high) > 0.0:
        high *= 2.0
    for _ in range(70):
        mid = 0.5 * (low + high)
        if residual(mid) > 0.0:
            low = mid
        else:
            high = mid
    mu = high
    return v / (1.0 + mu), t + mu


def project_rotated_soc(rho, lam, s):
    rho = np.asarray(rho, dtype=float).copy()
    lam = float(np.asarray(lam))
    s = float(np.asarray(s))
    if lam >= 0.0 and s >= 0.0 and np.dot(rho, rho) <= 2.0 * lam * s:
        return rho, lam, s

    root2 = np.sqrt(2.0)
    a = (lam - s) / root2
    t = (lam + s) / root2
    z = np.concatenate([rho, np.array([a])])
    norm_z = np.linalg.norm(z)
    if norm_z <= t:
        zp, tp = z, t
    elif norm_z <= -t:
        zp, tp = np.zeros_like(z), 0.0
    else:
        alpha = 0.5 * (norm_z + t)
        zp, tp = alpha / max(norm_z, 1e-12) * z, alpha
    rho_p = zp[:-1]
    a_p = zp[-1]
    lam_p = (tp + a_p) / root2
    s_p = (tp - a_p) / root2
    return rho_p, max(lam_p, 0.0), max(s_p, 0.0)


class LeastSquaresLDPM:
    def __init__(self, data_info, regularizers, setting):
        self.data_info = data_info
        self.settings = data_info.settings
        self.data = data_info.data
        self.regularizers = regularizers
        self.setting = setting
        self.a_tr = np.asarray(self.data.X_train, dtype=float)
        self.b_tr = np.asarray(self.data.y_train, dtype=float).reshape(-1)
        self.a_val = np.asarray(self.data.X_validate, dtype=float)
        self.b_val = np.asarray(self.data.y_validate, dtype=float).reshape(-1)
        self.n = self.a_tr.shape[1]
        self.m = self.a_tr.shape[0]
        if setting.get("normalize_loss", True):
            if setting.get("sqrt_loss_scaling", False):
                train_root = np.sqrt(float(self.settings.num_train))
                val_root = np.sqrt(float(self.settings.num_validate))
                self.a_tr = self.a_tr / train_root
                self.b_tr = self.b_tr / train_root
                self.a_val = self.a_val / val_root
                self.b_val = self.b_val / val_root
                self.train_scale = 1.0
                self.val_scale = 1.0
            else:
                self.train_scale = 1.0 / self.settings.num_train
                self.val_scale = 1.0 / self.settings.num_validate
        else:
            self.train_scale = 1.0
            self.val_scale = 1.0
        self.dual_scale = 1.0 / self.train_scale
        self.reduced_dual = bool(setting.get("reduced_dual", False))
        if self.reduced_dual:
            self.train_gram = self.a_tr.T @ self.a_tr
            self.train_atb = self.a_tr.T @ self.b_tr
            self.train_btb = float(np.dot(self.b_tr, self.b_tr))
            self.val_gram = self.a_val.T @ self.a_val
            self.val_atb = self.a_val.T @ self.b_val
            self.val_btb = float(np.dot(self.b_val, self.b_val))
        self.r_count = len(regularizers)
        self.quad_index = next(
            (i for i, reg in enumerate(regularizers) if reg["type"] == "squared_l2"),
            None,
        )
        self.has_quad = self.quad_index is not None

    def beta(self, k):
        beta0 = self.setting.get("beta0", 1.0)
        power = self.setting.get("beta_power", self.setting.get("p", 0.3))
        beta = beta0 * (1.0 + k) ** power
        beta_max = self.setting.get("beta_max", None)
        if beta_max is not None:
            beta = min(beta, float(beta_max))
        return beta

    def initial_lambda(self):
        if "initial_lambda" in self.setting:
            lam = np.asarray(self.setting["initial_lambda"], dtype=float)
        elif "lambda0" in self.setting:
            lam = np.asarray(self.setting["lambda0"], dtype=float)
        else:
            lam = np.full(self.r_count, 0.1)
            if self.quad_index is not None:
                lam[self.quad_index] = 0.05
            for i, reg in enumerate(self.regularizers):
                if reg["type"] == "l1" and self.r_count > 1:
                    lam[i] = 0.05
        return np.maximum(lam, 1e-8)

    def lower_solve(self, lam):
        max_iter = self.setting.get("init_max_iter", 300)
        tol = self.setting.get("init_tol", 1e-7)
        if self.reduced_dual:
            lipschitz = self.train_scale * float(
                np.max(np.linalg.eigvalsh(self.train_gram))
            )
        else:
            lipschitz = self.train_scale * _spectral_norm_squared(self.a_tr)
        quad = 0.0
        if self.quad_index is not None:
            quad = lam[self.quad_index]
        step = 1.0 / max(lipschitz + quad, 1e-8)
        x = np.zeros(self.n)
        for _ in range(max_iter):
            old = x.copy()
            if self.reduced_dual:
                loss_gradient = self.train_gram @ x - self.train_atb
            else:
                loss_gradient = self.a_tr.T @ (self.a_tr @ x - self.b_tr)
            x = x - step * self.train_scale * loss_gradient
            l1_index = next(
                (i for i, reg in enumerate(self.regularizers) if reg["type"] == "l1"),
                None,
            )
            if l1_index is not None:
                x = _soft_threshold(x, step * lam[l1_index])
            for i, reg in enumerate(self.regularizers):
                if reg["type"] == "group_l2":
                    sl = reg["slice"]
                    norm_g = np.linalg.norm(x[sl])
                    shrink = max(0.0, 1.0 - step * lam[i] / max(norm_g, 1e-12))
                    x[sl] *= shrink
            if self.quad_index is not None:
                x /= (1.0 + step * lam[self.quad_index])
            if _relative_step(x, old) < tol:
                break
        return x

    def initial_state(self):
        lam = self.initial_lambda()
        if "initial_coef" in self.setting:
            x = np.asarray(self.setting["initial_coef"], dtype=float).reshape(self.n).copy()
        else:
            x = self.lower_solve(lam)
        r = np.array([self.reg_value(reg, x) for reg in self.regularizers], dtype=float)
        rho = np.zeros((self.r_count, self.n))
        init_dual = self.setting.get("init_dual", "zero")
        if init_dual == "zero":
            xi = np.zeros(self.n + 1 if self.reduced_dual else self.m)
        elif init_dual in {"fenchel", "kkt"}:
            if self.reduced_dual:
                xi = self.train_scale * np.concatenate(([-1.0], x))
            else:
                xi = self.train_scale * (self.a_tr @ x - self.b_tr)
            loss_dual_gradient = self.dual_image(xi)
            target = -loss_dual_gradient
            if init_dual == "kkt":
                l1_indices = [
                    i
                    for i, reg in enumerate(self.regularizers)
                    if reg["type"] == "l1"
                ]
                if len(l1_indices) > 1:
                    raise ValueError(
                        "kkt dual initialization supports at most one l1 regularizer"
                    )
                if l1_indices:
                    l1_index = l1_indices[0]
                    rho[l1_index] = np.clip(
                        target, -lam[l1_index], lam[l1_index]
                    )
                residual = target - np.sum(rho, axis=0)
                for i, reg in enumerate(self.regularizers):
                    if reg["type"] != "group_l2":
                        continue
                    sl = reg["slice"]
                    candidate = residual[sl].copy()
                    candidate_norm = np.linalg.norm(candidate)
                    if candidate_norm > lam[i]:
                        candidate *= lam[i] / max(candidate_norm, 1e-12)
                    rho[i, sl] = candidate
                    residual[sl] -= candidate
            else:
                for i, reg in enumerate(self.regularizers):
                    if reg["type"] != "group_l2":
                        continue
                    sl = reg["slice"]
                    candidate = target[sl]
                    candidate_norm = np.linalg.norm(candidate)
                    if candidate_norm > lam[i]:
                        candidate = candidate * (
                            lam[i] / max(candidate_norm, 1e-12)
                        )
                    rho[i, sl] = candidate
        else:
            raise ValueError("unknown dual initialization %r" % init_dual)
        s = 0.0
        if self.quad_index is not None:
            s = np.dot(rho[self.quad_index], rho[self.quad_index]) / (
                2.0 * max(lam[self.quad_index], 1e-8)
            )
        return x, r, lam, rho, xi, s

    def reg_value(self, reg, x):
        if reg["type"] == "group_l2":
            return np.linalg.norm(x[reg["slice"]])
        if reg["type"] == "l1":
            return np.linalg.norm(x, 1)
        if reg["type"] == "squared_l2":
            return 0.5 * np.dot(x, x)
        raise ValueError("unknown regularizer %r" % reg["type"])

    def dual_image(self, xi):
        if not self.reduced_dual:
            return self.a_tr.T @ xi
        return xi[0] * self.train_atb + self.train_gram @ xi[1:]

    def dual_inner(self, left, right):
        if not self.reduced_dual:
            return float(np.dot(left, right))
        left_b, left_x = float(left[0]), left[1:]
        right_b, right_x = float(right[0]), right[1:]
        return float(
            left_b * right_b * self.train_btb
            + left_b * np.dot(self.train_atb, right_x)
            + right_b * np.dot(self.train_atb, left_x)
            + np.dot(left_x, self.train_gram @ right_x)
        )

    def dual_b_inner(self, xi):
        if not self.reduced_dual:
            return float(np.dot(xi, self.b_tr))
        return float(xi[0] * self.train_btb + np.dot(self.train_atb, xi[1:]))

    def materialize_dual(self, xi):
        if not self.reduced_dual:
            return np.asarray(xi, dtype=float).copy()
        return xi[0] * self.b_tr + self.a_tr @ xi[1:]

    @staticmethod
    def quadratic_residual(x, gram, atb, btb):
        value = float(np.dot(x, gram @ x) - 2.0 * np.dot(x, atb) + btb)
        return max(value, 0.0)

    def h_value(self, xi, rho):
        return self.dual_image(xi) + np.sum(rho, axis=0)

    def q_value(self, x, r, lam, rho, xi, s):
        if self.reduced_dual:
            residual_sq = self.quadratic_residual(
                x, self.train_gram, self.train_atb, self.train_btb
            )
        else:
            residual = self.a_tr @ x - self.b_tr
            residual_sq = float(np.dot(residual, residual))
        h = self.h_value(xi, rho)
        value = (
            0.5 * self.train_scale * residual_sq
            + np.dot(lam, r)
            + 0.5 * self.dual_scale * self.dual_inner(xi, xi)
            + self.dual_b_inner(xi)
            - np.dot(x, h)
            + 0.5 * np.dot(h, h)
        )
        if self.has_quad:
            value += s
        return float(value)

    def gradients(self, x, r, lam, rho, xi, s, beta):
        h = self.h_value(xi, rho)
        if self.reduced_dual:
            grad_upper = self.val_scale * (self.val_gram @ x - self.val_atb)
            grad_lower = self.train_scale * (self.train_gram @ x - self.train_atb)
        else:
            grad_upper = self.val_scale * (self.a_val.T @ (self.a_val @ x - self.b_val))
            grad_lower = self.train_scale * (self.a_tr.T @ (self.a_tr @ x - self.b_tr))
        grad_x = grad_upper / beta + grad_lower - h
        grad_r = lam.copy()
        grad_lam = r.copy()
        grad_rho = np.tile((-x + h), (self.r_count, 1))
        grad_s = 1.0 if self.has_quad else 0.0
        if self.reduced_dual:
            grad_xi_smooth = np.concatenate(([1.0], -x + h))
        else:
            grad_xi_smooth = self.b_tr - self.a_tr @ x + self.a_tr @ h
        return grad_x, grad_r, grad_lam, grad_rho, grad_s, grad_xi_smooth

    def prox_xi(self, xi, grad_xi_smooth, step):
        return (xi - step * grad_xi_smooth) / (1.0 + step * self.dual_scale)

    def line_search_smooth_value(self, x, r, lam, rho, xi, s, beta):
        if self.reduced_dual:
            upper = 0.5 * self.val_scale * self.quadratic_residual(
                x, self.val_gram, self.val_atb, self.val_btb
            )
        else:
            val_residual = self.a_val @ x - self.b_val
            upper = 0.5 * self.val_scale * float(np.dot(val_residual, val_residual))
        return (
            upper / max(beta, 1e-12)
            + self.q_value(x, r, lam, rho, xi, s)
            - 0.5 * self.dual_scale * self.dual_inner(xi, xi)
        )

    def line_search_update(
        self,
        x,
        r,
        lam,
        rho,
        xi,
        s,
        beta,
        grads,
        current_step,
    ):
        grad_x, grad_r, grad_lam, grad_rho, grad_s, grad_xi_smooth = grads
        decay = float(self.setting.get("line_search_decay", 0.5))
        growth = float(self.setting.get("line_search_growth", 1.25))
        min_step = float(self.setting.get("line_search_min_step", 1e-12))
        max_step = float(
            self.setting.get("line_search_max_step", max(current_step, 1e-12))
        )
        max_trials = int(self.setting.get("max_line_search_iter", 50))
        current_value = self.line_search_smooth_value(x, r, lam, rho, xi, s, beta)
        trial_step = min(max_step, max(min_step, current_step * growth))
        fallback = None
        self._last_line_search_trials = 0
        self._last_line_search_outcome = "no_finite_trial"
        for trial_number in range(1, max_trials + 1):
            self._last_line_search_trials = trial_number
            x_trial = x - trial_step * grad_x
            r_trial = r - trial_step * grad_r
            lam_trial = lam - trial_step * grad_lam
            rho_trial = rho - trial_step * grad_rho
            s_trial = s - trial_step * grad_s if self.has_quad else s
            xi_trial = self.prox_xi(xi, grad_xi_smooth, trial_step)
            x_trial, r_trial, lam_trial, rho_trial, s_trial = self.project_pgm(
                x_trial, r_trial, lam_trial, rho_trial, s_trial
            )
            trial_value = self.line_search_smooth_value(
                x_trial, r_trial, lam_trial, rho_trial, xi_trial, s_trial, beta
            )
            finite_trial = (
                np.isfinite(trial_value)
                and np.all(np.isfinite(x_trial))
                and np.all(np.isfinite(r_trial))
                and np.all(np.isfinite(lam_trial))
                and np.all(np.isfinite(rho_trial))
                and np.isfinite(s_trial)
                and np.all(np.isfinite(xi_trial))
            )
            if finite_trial:
                old_vec = self.pack(x, r, lam, rho, s)
                trial_vec = self.pack(x_trial, r_trial, lam_trial, rho_trial, s_trial)
                grad_vec = self.pack(grad_x, grad_r, grad_lam, grad_rho, grad_s)
                delta_vec = trial_vec - old_vec
                delta_xi = xi_trial - xi
                rhs = (
                    current_value
                    + float(np.dot(grad_vec, delta_vec))
                    + self.dual_inner(grad_xi_smooth, delta_xi)
                    + 0.5
                    / trial_step
                    * float(
                        np.dot(delta_vec, delta_vec)
                        + self.dual_inner(delta_xi, delta_xi)
                    )
                    + 1e-12
                )
                fallback = (
                    x_trial,
                    r_trial,
                    lam_trial,
                    rho_trial,
                    xi_trial,
                    s_trial,
                    trial_step,
                )
                if (not np.isfinite(current_value)) or trial_value <= rhs:
                    self._last_line_search_outcome = "accepted"
                    return fallback
            trial_step *= decay
            if trial_step < min_step:
                break
        if fallback is not None:
            self._last_line_search_outcome = "finite_fallback"
            return fallback
        return x, r, lam, rho, xi, s, min_step

    def admm_smooth_value(self, z, xi, u, mu, beta, gamma):
        """Smooth augmented-Lagrangian value with phi*(xi) split to prox."""

        x, r, lam, rho, s = self.unpack(z)
        value = self.line_search_smooth_value(
            x, r, lam, rho, xi, s, beta
        )
        for local, multiplier in zip(u, mu):
            delta = local - z
            value += float(np.dot(multiplier, delta))
            value += 0.5 * gamma * float(np.dot(delta, delta))
        return float(value)

    def line_search_admm_update(
        self,
        z,
        xi,
        u,
        mu,
        beta,
        gamma,
        direction,
        grad_xi_smooth,
        current_step,
    ):
        """Backtracking proximal-gradient update for the LDPM-CS z block."""

        decay = float(self.setting.get("line_search_decay", 0.5))
        growth = float(self.setting.get("line_search_growth", 1.25))
        min_step = float(self.setting.get("line_search_min_step", 1e-12))
        max_step = float(
            self.setting.get("line_search_max_step", max(current_step, 1e-12))
        )
        max_trials = int(self.setting.get("max_line_search_iter", 50))
        current_value = self.admm_smooth_value(
            z, xi, u, mu, beta, gamma
        )
        trial_step = min(max_step, max(min_step, current_step * growth))
        fallback = None
        self._last_line_search_trials = 0
        self._last_line_search_outcome = "no_finite_trial"
        for trial_number in range(1, max_trials + 1):
            self._last_line_search_trials = trial_number
            z_trial = z - trial_step * direction
            xi_trial = self.prox_xi(xi, grad_xi_smooth, trial_step)
            trial_value = self.admm_smooth_value(
                z_trial, xi_trial, u, mu, beta, gamma
            )
            finite_trial = (
                np.isfinite(trial_value)
                and np.all(np.isfinite(z_trial))
                and np.all(np.isfinite(xi_trial))
            )
            if finite_trial:
                delta_z = z_trial - z
                delta_xi = xi_trial - xi
                rhs = (
                    current_value
                    + float(np.dot(direction, delta_z))
                    + self.dual_inner(grad_xi_smooth, delta_xi)
                    + 0.5
                    / trial_step
                    * float(
                        np.dot(delta_z, delta_z)
                        + self.dual_inner(delta_xi, delta_xi)
                    )
                    + 1e-12
                )
                fallback = z_trial, xi_trial, trial_step
                if (not np.isfinite(current_value)) or trial_value <= rhs:
                    self._last_line_search_outcome = "accepted"
                    return fallback
            trial_step *= decay
            if trial_step < min_step:
                break
        if fallback is not None:
            self._last_line_search_outcome = "finite_fallback"
            return fallback
        return z, xi, min_step

    def pack(self, x, r, lam, rho, s):
        parts = [x, r, lam, rho.reshape(-1)]
        if self.has_quad:
            parts.append(np.array([s], dtype=float))
        return np.concatenate(parts)

    def pack_full_state(self, x, r, lam, rho, xi, s):
        """Pack the full paper variable z=(tilde_z, xi)."""

        return np.concatenate(
            [self.pack(x, r, lam, rho, s), self.materialize_dual(xi)]
        )

    def unpack(self, vec):
        pos = 0
        x = vec[pos : pos + self.n].copy()
        pos += self.n
        r = vec[pos : pos + self.r_count].copy()
        pos += self.r_count
        lam = vec[pos : pos + self.r_count].copy()
        pos += self.r_count
        rho = vec[pos : pos + self.r_count * self.n].reshape(self.r_count, self.n).copy()
        pos += self.r_count * self.n
        s = float(vec[pos]) if self.has_quad else 0.0
        return x, r, lam, rho, s

    def project_primal_one(self, x, r, index):
        reg = self.regularizers[index]
        if reg["type"] == "group_l2":
            sl = reg["slice"]
            x[sl], r[index] = project_l2_epigraph(x[sl], r[index])
        elif reg["type"] == "l1":
            x[:], r[index] = project_l1_epigraph(x, r[index])
        elif reg["type"] == "squared_l2":
            x[:], r[index] = project_squared_l2_epigraph(x, r[index])
        return x, r

    def project_dual_all(self, rho, lam, s):
        for i, reg in enumerate(self.regularizers):
            if reg["type"] == "group_l2":
                sl = reg["slice"]
                projected = np.zeros(self.n)
                projected[sl], lam[i] = project_l2_epigraph(rho[i, sl], lam[i])
                rho[i] = projected
            elif reg["type"] == "l1":
                rho[i], lam[i] = project_linf_epigraph(rho[i], lam[i])
            elif reg["type"] == "squared_l2":
                rho[i], lam[i], s = project_rotated_soc(rho[i], lam[i], s)
        return rho, lam, s

    def project_pgm(self, x, r, lam, rho, s):
        # The group epigraphs and the global l1 epigraph share x.  A single
        # cyclic pass is not the projection onto their intersection.  Use
        # Dykstra's algorithm, matching the original sparse-group projection.
        current = np.concatenate([np.asarray(x, dtype=float), np.asarray(r, dtype=float)])
        # The group-l2 blocks are disjoint, so their Cartesian-product
        # projection is one projector.  Treating every block as a separate
        # Dykstra set is correct but needlessly expensive for Table 3.
        group_indices = [
            i for i, reg in enumerate(self.regularizers) if reg["type"] == "group_l2"
        ]
        projector_indices = [group_indices] if group_indices else []
        projector_indices.extend(
            [i]
            for i, reg in enumerate(self.regularizers)
            if reg["type"] != "group_l2"
        )
        corrections = [np.zeros_like(current) for _ in projector_indices]
        max_sweeps = int(self.setting.get("projection_max_sweeps", 100))
        projection_tol = float(self.setting.get("projection_tol", 1e-7))
        for _ in range(max_sweeps):
            sweep_start = current.copy()
            for projector, indices in enumerate(projector_indices):
                shifted = current + corrections[projector]
                x_shifted = shifted[: self.n].copy()
                r_shifted = shifted[self.n :].copy()
                for i in indices:
                    x_shifted, r_shifted = self.project_primal_one(
                        x_shifted, r_shifted, i
                    )
                x_projected, r_projected = x_shifted, r_shifted
                projected = np.concatenate([x_projected, r_projected])
                corrections[projector] = shifted - projected
                current = projected
            if np.linalg.norm(current - sweep_start) <= projection_tol * max(
                1.0, np.linalg.norm(sweep_start)
            ):
                break
        x = current[: self.n].copy()
        r = current[self.n :].copy()
        rho, lam, s = self.project_dual_all(rho, lam, s)
        return x, r, lam, rho, s

    def project_admm_constraint(self, vec, index):
        x, r, lam, rho, s = self.unpack(vec)
        x, r = self.project_primal_one(x, r, index)
        rho, lam, s = self.project_dual_all(rho, lam, s)
        return self.pack(x, r, lam, rho, s)

    def make_record(self, elapsed, x, r, lam, rho, xi, s, iteration):
        h = self.h_value(xi, rho)
        p_value = self.q_value(x, r, lam, rho, xi, s) + np.dot(x, h) - 0.5 * np.dot(h, h)
        return {
            "iteration": iteration,
            "time": elapsed,
            "train_error": train_error(self.settings, self.data, x),
            "validation_error": validation_error(self.settings, self.data, x),
            "test_error": test_error(self.settings, self.data, x),
            "ll_duality_gap": self.q_value(x, r, lam, rho, xi, s),
            "ll_feasibility": max(abs(float(p_value)), float(np.linalg.norm(h))),
            "sparsity": 100.0 * np.mean(np.abs(x) > 1e-6),
            "lambda_values": ";".join("%.17g" % value for value in np.asarray(lam, dtype=float)),
        }

    def prepare_pgm_state(self):
        """Construct the standard LDPM-PG initialization outside timed work."""

        return self.initial_state()

    def prepare_admm_state(self):
        """Construct the standard LDPM-CS initialization outside timed work."""

        x, r, lam, rho, xi, s = self.initial_state()
        z = self.pack(x, r, lam, rho, s)
        gamma = float(self.setting.get("gamma", self.setting.get("prox", 10.0)))
        u = [self.project_admm_constraint(z, i) for i in range(self.r_count)]
        mu = [np.zeros_like(z) for _ in range(self.r_count)]
        return x, r, lam, rho, xi, s, z, u, mu

    @staticmethod
    def _copy_pgm_state(state):
        x, r, lam, rho, xi, s = state
        return (
            np.asarray(x, dtype=float).copy(),
            np.asarray(r, dtype=float).copy(),
            np.asarray(lam, dtype=float).copy(),
            np.asarray(rho, dtype=float).copy(),
            np.asarray(xi, dtype=float).copy(),
            float(s),
        )

    @classmethod
    def _copy_admm_state(cls, state):
        x, r, lam, rho, xi, s, z, u, mu = state
        pgm_state = cls._copy_pgm_state((x, r, lam, rho, xi, s))
        return (
            *pgm_state,
            np.asarray(z, dtype=float).copy(),
            [np.asarray(local, dtype=float).copy() for local in u],
            [np.asarray(multiplier, dtype=float).copy() for multiplier in mu],
        )

    def run_pgm(self, prepared_state=None):
        if prepared_state is None:
            x, r, lam, rho, xi, s = self.prepare_pgm_state()
        elif self.setting.get("copy_prepared_state", True):
            x, r, lam, rho, xi, s = self._copy_pgm_state(prepared_state)
        else:
            x, r, lam, rho, xi, s = prepared_state
        step = self.setting.get("step_size", self.setting.get("gd_step", 0.01))
        line_search = bool(self.setting.get("line_search", False))
        tol = self.setting.get("TOL", 1e-2)
        max_iter = self.setting.get("MAX_ITERATION", 100)
        time_limit_sec = float(self.setting.get("time_limit_sec", np.inf))
        stop_metric = self.setting.get("stop_metric", "full_z")
        use_blockwise_stop = str(stop_metric) == "blockwise"
        record_interval = max(1, int(self.setting.get("record_interval", 1)))
        stop_patience = max(1, int(self.setting.get("stop_patience", 1)))
        consecutive_stop_hits = 0
        first_hit_iteration = None
        initial_record = self.make_record(0.0, x, r, lam, rho, xi, s, 0)
        initial_record["beta"] = self.beta(0)
        initial_record["accepted_step_size"] = 0.0
        initial_record["packed_relative_step"] = 0.0
        initial_record["full_z_relative_step"] = 0.0
        initial_record["r_stat"] = 0.0
        initial_record["stop_metric"] = 0.0
        initial_record["x_lambda_stop"] = 0.0
        initial_record["blockwise_stop"] = 0.0
        initial_record["first_hit_iteration"] = None
        initial_record["stop_consecutive_hits"] = 0
        initial_record["stop_patience"] = stop_patience
        initial_record["confirmed_residual_stop"] = False
        initial_record["common_initial_point"] = True
        initial_record["backtracking_trials"] = 0
        initial_record["total_backtracking_trials"] = 0
        initial_record["line_search_outcome"] = "initial"
        records = [initial_record]
        termination_status = "max_iter"
        total_backtracking_trials = 0
        accepted_step_sizes = []
        backtracking_trials_history = []
        line_search_outcomes = []
        last_iteration = 0
        start = time.perf_counter()
        for k in range(max_iter):
            x_old = x.copy()
            lam_old = lam.copy()
            old_blocks = None
            if use_blockwise_stop:
                old_blocks = [
                    x.copy(),
                    r.copy(),
                    lam.copy(),
                    rho.copy(),
                    self.materialize_dual(xi),
                ]
                if self.has_quad:
                    old_blocks.append(np.asarray([s], dtype=float))
            old = self.pack(x, r, lam, rho, s)
            old_full = self.pack_full_state(x, r, lam, rho, xi, s)
            beta = self.beta(k)
            grad_x, grad_r, grad_lam, grad_rho, grad_s, grad_xi_smooth = self.gradients(
                x, r, lam, rho, xi, s, beta
            )
            grads = (grad_x, grad_r, grad_lam, grad_rho, grad_s, grad_xi_smooth)
            if line_search:
                x, r, lam, rho, xi, s, step = self.line_search_update(
                    x, r, lam, rho, xi, s, beta, grads, step
                )
                backtracking_trials = int(
                    getattr(self, "_last_line_search_trials", 0)
                )
                line_search_outcome = str(
                    getattr(self, "_last_line_search_outcome", "unknown")
                )
            else:
                x = x - step * grad_x
                r = r - step * grad_r
                lam = lam - step * grad_lam
                rho = rho - step * grad_rho
                if self.has_quad:
                    s = s - step * grad_s
                xi = self.prox_xi(xi, grad_xi_smooth, step)
                x, r, lam, rho, s = self.project_pgm(x, r, lam, rho, s)
                backtracking_trials = 0
                line_search_outcome = "disabled"
            total_backtracking_trials += backtracking_trials
            backtracking_trials_history.append(int(backtracking_trials))
            line_search_outcomes.append(str(line_search_outcome))
            if line_search_outcome == "no_finite_trial":
                elapsed = time.perf_counter() - start
                record = self.make_record(
                    elapsed, x, r, lam, rho, xi, s, last_iteration
                )
                record["attempted_iteration"] = k + 1
                record["beta"] = beta
                record["accepted_step_size"] = np.nan
                record["backtracking_trials"] = backtracking_trials
                record["total_backtracking_trials"] = total_backtracking_trials
                record["line_search_outcome"] = line_search_outcome
                record["packed_relative_step"] = np.nan
                record["full_z_relative_step"] = np.nan
                record["r_stat"] = np.nan
                record["stop_metric"] = np.nan
                record["x_lambda_stop"] = np.nan
                record["blockwise_stop"] = np.nan
                record["first_hit_iteration"] = first_hit_iteration
                record["stop_consecutive_hits"] = 0
                record["stop_patience"] = stop_patience
                record["confirmed_residual_stop"] = False
                records.append(record)
                termination_status = "nonfinite"
                break
            accepted_step_sizes.append(float(step))
            last_iteration = k + 1
            new = self.pack(x, r, lam, rho, s)
            packed_stop = _relative_step(new, old)
            full_z_stop = _relative_step(
                self.pack_full_state(x, r, lam, rho, xi, s), old_full
            )
            x_lambda_stop = _x_lambda_max_relative_stop(x, lam, x_old, lam_old)
            if use_blockwise_stop:
                new_blocks = [
                    x,
                    r,
                    lam,
                    rho,
                    self.materialize_dual(xi),
                ]
                if self.has_quad:
                    new_blocks.append(np.asarray([s], dtype=float))
                blockwise_stop = _blockwise_max_relative_stop(
                    new_blocks,
                    old_blocks,
                )
            else:
                blockwise_stop = np.nan
            selected_stop = _select_stationarity_stop(
                stop_metric,
                full_z_stop,
                packed_stop,
                x_lambda_stop,
                blockwise_stop=blockwise_stop,
            )
            nonfinite_stop = (
                not np.isfinite(selected_stop)
            )
            raw_stop_hit = np.isfinite(selected_stop) and selected_stop <= tol
            if raw_stop_hit:
                if first_hit_iteration is None:
                    first_hit_iteration = k + 1
                consecutive_stop_hits += 1
            else:
                consecutive_stop_hits = 0
            converged_stop = consecutive_stop_hits >= stop_patience
            elapsed = time.perf_counter() - start
            time_limit_reached = elapsed >= time_limit_sec
            should_record = (
                nonfinite_stop
                or converged_stop
                or time_limit_reached
                or (k + 1) % record_interval == 0
                or k + 1 == max_iter
            )
            if should_record:
                record = self.make_record(elapsed, x, r, lam, rho, xi, s, k + 1)
                record["beta"] = beta
                record["accepted_step_size"] = step
                record["backtracking_trials"] = backtracking_trials
                record["total_backtracking_trials"] = total_backtracking_trials
                record["line_search_outcome"] = line_search_outcome
                record["packed_relative_step"] = packed_stop
                record["full_z_relative_step"] = full_z_stop
                record["r_stat"] = full_z_stop
                record["stop_metric"] = selected_stop
                record["x_lambda_stop"] = x_lambda_stop
                record["blockwise_stop"] = blockwise_stop
                record["first_hit_iteration"] = first_hit_iteration
                record["stop_consecutive_hits"] = consecutive_stop_hits
                record["stop_patience"] = stop_patience
                record["confirmed_residual_stop"] = converged_stop
                records.append(record)
            if nonfinite_stop:
                termination_status = "nonfinite"
                break
            if time_limit_reached:
                termination_status = "time_limit"
                break
            if converged_stop:
                termination_status = "converged"
                break
        algorithm_runtime_sec = time.perf_counter() - start
        self.coef_ = x.copy()
        self.lambda_ = lam.copy()
        self.radius_ = r.copy()
        self.rho_ = rho.copy()
        self.xi_reduced_ = xi.copy() if self.reduced_dual else None
        self.xi_ = self.materialize_dual(xi)
        self.termination_status_ = termination_status
        frame = pd.DataFrame(records)
        frame.attrs["coef"] = self.coef_.copy()
        frame.attrs["lambda"] = self.lambda_.copy()
        frame.attrs["radius"] = self.radius_.copy()
        frame.attrs["rho"] = self.rho_.copy()
        frame.attrs["xi"] = self.xi_.copy()
        frame.attrs["xi_reduced"] = (
            None if self.xi_reduced_ is None else self.xi_reduced_.copy()
        )
        frame.attrs["termination_status"] = termination_status
        frame.attrs["outer_iterations"] = int(last_iteration)
        frame.attrs["algorithm_runtime_sec"] = float(algorithm_runtime_sec)
        frame.attrs["total_backtracking_trials"] = int(total_backtracking_trials)
        frame.attrs["accepted_step_sizes"] = np.asarray(
            accepted_step_sizes, dtype=float
        )
        frame.attrs["backtracking_trials"] = np.asarray(
            backtracking_trials_history, dtype=int
        )
        frame.attrs["line_search_outcomes"] = tuple(line_search_outcomes)
        return frame

    def run_admm(self, prepared_state=None):
        if prepared_state is None:
            x, r, lam, rho, xi, s, z, u, mu = self.prepare_admm_state()
        elif self.setting.get("copy_prepared_state", True):
            x, r, lam, rho, xi, s, z, u, mu = self._copy_admm_state(
                prepared_state
            )
        else:
            x, r, lam, rho, xi, s, z, u, mu = prepared_state
        gamma = float(self.setting.get("gamma", self.setting.get("prox", 10.0)))
        step = float(self.setting.get("step_size", self.setting.get("gd_step", 0.01)))
        line_search = bool(self.setting.get("line_search", False))
        tol = float(self.setting.get("TOL", 1e-1))
        max_iter = int(self.setting.get("MAX_ITERATION", 100))
        time_limit_sec = float(self.setting.get("time_limit_sec", np.inf))
        stop_metric = self.setting.get("stop_metric", "full_z")
        record_interval = max(1, int(self.setting.get("record_interval", 1)))
        stop_patience = max(1, int(self.setting.get("stop_patience", 1)))
        num_constraints = self.r_count
        initial_cons = max(
            float(np.linalg.norm(local - z)) for local in u
        ) / max(1.0, float(np.linalg.norm(z)))
        initial_record = self.make_record(0.0, x, r, lam, rho, xi, s, 0)
        initial_record.update(
            beta=self.beta(0),
            accepted_step_size=0.0,
            packed_relative_step=0.0,
            full_z_relative_step=0.0,
            r_stat=0.0,
            r_cons=initial_cons,
            stop_metric=initial_cons,
            x_lambda_stop=0.0,
            stop_consecutive_hits=0,
            stop_patience=stop_patience,
            confirmed_residual_stop=False,
            common_initial_point=True,
            backtracking_trials=0,
            total_backtracking_trials=0,
            line_search_outcome="initial",
        )
        records = [initial_record]
        consecutive_stop_hits = 0
        termination_status = "max_iter"
        total_backtracking_trials = 0
        accepted_step_sizes = []
        backtracking_trials_history = []
        line_search_outcomes = []
        last_iteration = 0
        start = time.perf_counter()
        for k in range(max_iter):
            x_old = x.copy()
            lam_old = lam.copy()
            old_z = z.copy()
            old_full = self.pack_full_state(x, r, lam, rho, xi, s)
            beta = self.beta(k)
            grad_x, grad_r, grad_lam, grad_rho, grad_s, grad_xi_smooth = self.gradients(
                x, r, lam, rho, xi, s, beta
            )
            grad = self.pack(grad_x, grad_r, grad_lam, grad_rho, grad_s)
            sum_mu = np.zeros_like(z)
            sum_u = np.zeros_like(z)
            for local, multiplier in zip(u, mu):
                sum_u += local
                sum_mu += multiplier
            direction = grad - sum_mu + gamma * (num_constraints * z - sum_u)
            if line_search:
                z, xi, step = self.line_search_admm_update(
                    z,
                    xi,
                    u,
                    mu,
                    beta,
                    gamma,
                    direction,
                    grad_xi_smooth,
                    step,
                )
                backtracking_trials = int(
                    getattr(self, "_last_line_search_trials", 0)
                )
                line_search_outcome = str(
                    getattr(self, "_last_line_search_outcome", "unknown")
                )
            else:
                z = z - step * direction
                xi = self.prox_xi(xi, grad_xi_smooth, step)
                backtracking_trials = 0
                line_search_outcome = "disabled"
            total_backtracking_trials += backtracking_trials
            backtracking_trials_history.append(int(backtracking_trials))
            line_search_outcomes.append(str(line_search_outcome))
            if line_search_outcome == "no_finite_trial":
                elapsed = time.perf_counter() - start
                current_consensus = max(
                    float(np.linalg.norm(local - z)) for local in u
                ) / max(1.0, float(np.linalg.norm(z)))
                record = self.make_record(
                    elapsed, x, r, lam, rho, xi, s, last_iteration
                )
                record.update(
                    attempted_iteration=k + 1,
                    beta=beta,
                    accepted_step_size=np.nan,
                    backtracking_trials=backtracking_trials,
                    total_backtracking_trials=total_backtracking_trials,
                    line_search_outcome=line_search_outcome,
                    packed_relative_step=np.nan,
                    full_z_relative_step=np.nan,
                    r_stat=np.nan,
                    r_cons=current_consensus,
                    stop_metric=np.nan,
                    x_lambda_stop=np.nan,
                    stop_consecutive_hits=0,
                    stop_patience=stop_patience,
                    confirmed_residual_stop=False,
                )
                records.append(record)
                termination_status = "nonfinite"
                break
            accepted_step_sizes.append(float(step))
            last_iteration = k + 1
            x, r, lam, rho, s = self.unpack(z)
            for i in range(num_constraints):
                u[i] = self.project_admm_constraint(z - mu[i] / gamma, i)
                mu[i] = mu[i] + gamma * (u[i] - z)
            tilde_z_stop = _relative_step(z, old_z)
            full_z_stop = _relative_step(
                self.pack_full_state(x, r, lam, rho, xi, s), old_full
            )
            x_lambda_stop = _x_lambda_max_relative_stop(
                x, lam, x_old, lam_old
            )
            r_stat = full_z_stop
            r_cons = max(
                float(np.linalg.norm(local - z)) for local in u
            ) / max(1.0, float(np.linalg.norm(z)))
            stationarity_stop = _select_stationarity_stop(
                stop_metric, full_z_stop, tilde_z_stop, x_lambda_stop
            )
            selected_stop = max(stationarity_stop, r_cons)
            nonfinite_stop = (
                not np.isfinite(selected_stop)
            )
            if np.isfinite(selected_stop) and selected_stop <= tol:
                consecutive_stop_hits += 1
            else:
                consecutive_stop_hits = 0
            converged_stop = consecutive_stop_hits >= stop_patience
            elapsed = time.perf_counter() - start
            time_limit_reached = elapsed >= time_limit_sec
            should_record = (
                nonfinite_stop
                or converged_stop
                or time_limit_reached
                or (k + 1) % record_interval == 0
                or k + 1 == max_iter
            )
            if should_record:
                record = self.make_record(
                    elapsed, x, r, lam, rho, xi, s, k + 1
                )
                record.update(
                    beta=beta,
                    accepted_step_size=step,
                    backtracking_trials=backtracking_trials,
                    total_backtracking_trials=total_backtracking_trials,
                    line_search_outcome=line_search_outcome,
                    packed_relative_step=tilde_z_stop,
                    full_z_relative_step=full_z_stop,
                    r_stat=r_stat,
                    r_cons=r_cons,
                    stop_metric=selected_stop,
                    x_lambda_stop=x_lambda_stop,
                    stop_consecutive_hits=consecutive_stop_hits,
                    stop_patience=stop_patience,
                    confirmed_residual_stop=converged_stop,
                )
                records.append(record)
            if nonfinite_stop:
                termination_status = "nonfinite"
                break
            if time_limit_reached:
                termination_status = "time_limit"
                break
            if converged_stop:
                termination_status = "converged"
                break
        algorithm_runtime_sec = time.perf_counter() - start
        self.coef_ = x.copy()
        self.lambda_ = lam.copy()
        self.radius_ = r.copy()
        self.rho_ = rho.copy()
        self.xi_reduced_ = xi.copy() if self.reduced_dual else None
        self.xi_ = self.materialize_dual(xi)
        self.termination_status_ = termination_status
        frame = pd.DataFrame(records)
        frame.attrs["coef"] = self.coef_.copy()
        frame.attrs["lambda"] = self.lambda_.copy()
        frame.attrs["radius"] = self.radius_.copy()
        frame.attrs["rho"] = self.rho_.copy()
        frame.attrs["xi"] = self.xi_.copy()
        frame.attrs["xi_reduced"] = (
            None if self.xi_reduced_ is None else self.xi_reduced_.copy()
        )
        frame.attrs["termination_status"] = termination_status
        frame.attrs["local_copies"] = (
            [local.copy() for local in u]
            if self.setting.get("save_local_copies", True)
            else None
        )
        frame.attrs["outer_iterations"] = int(last_iteration)
        frame.attrs["algorithm_runtime_sec"] = float(algorithm_runtime_sec)
        frame.attrs["total_backtracking_trials"] = int(total_backtracking_trials)
        frame.attrs["accepted_step_sizes"] = np.asarray(
            accepted_step_sizes, dtype=float
        )
        frame.attrs["backtracking_trials"] = np.asarray(
            backtracking_trials_history, dtype=int
        )
        frame.attrs["line_search_outcomes"] = tuple(line_search_outcomes)
        return frame


def group_regularizers(num_features, num_groups):
    num_features = int(num_features)
    num_groups = int(num_groups)
    if num_features <= 0:
        raise ValueError("num_features must be positive")
    if not 1 <= num_groups <= num_features:
        raise ValueError("num_groups must lie in [1, num_features]")
    blocks = np.array_split(np.arange(num_features, dtype=int), num_groups)
    return [
        {
            "type": "group_l2",
            "slice": slice(int(block[0]), int(block[-1]) + 1),
        }
        for block in blocks
    ]


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


def _lift_matrix(a, groups):
    return np.hstack([a[:, group] for group in groups])


def _overlap_group_slices(groups):
    slices = []
    start = 0
    for group in groups:
        stop = start + len(group)
        slices.append(slice(start, stop))
        start = stop
    return slices


def collapse_lifted_coefficients(x, groups, num_features):
    w = np.zeros(num_features, dtype=float)
    for sl, group in zip(_overlap_group_slices(groups), groups):
        np.add.at(w, group, x[sl])
    return w


class OverlappingGroupLassoLDPM:
    """LDPM-PG for least-squares latent overlapping group Lasso.

    The latent lift writes w = Bx, where each block x_g lives on one
    overlapping group. The lower-level penalty is sum_g lambda_g ||x_g||_2.
    """

    def __init__(self, data_info, groups, setting):
        self.data_info = data_info
        self.settings = data_info.settings
        self.data = data_info.data
        self.setting = setting
        self.num_features = int(self.settings.num_features)
        self.groups = normalize_overlap_groups(
            groups,
            self.num_features,
            one_based=setting.get("one_based_groups", setting.get("groups_are_one_based", False)),
            add_singletons=setting.get("add_singleton_groups", True),
            require_coverage=setting.get(
                "require_group_coverage",
                setting.get("add_singleton_groups", True),
            ),
        )
        self.group_slices = _overlap_group_slices(self.groups)
        self.group_count = len(self.groups)
        self.a_tr = np.asarray(_lift_matrix(np.asarray(self.data.X_train, dtype=float), self.groups), dtype=float)
        self.b_tr = np.asarray(self.data.y_train, dtype=float).reshape(-1)
        self.a_val = np.asarray(_lift_matrix(np.asarray(self.data.X_validate, dtype=float), self.groups), dtype=float)
        self.b_val = np.asarray(self.data.y_validate, dtype=float).reshape(-1)
        self.a_test = np.asarray(_lift_matrix(np.asarray(self.data.X_test, dtype=float), self.groups), dtype=float)
        self.b_test = np.asarray(self.data.y_test, dtype=float).reshape(-1)
        self.q = self.a_tr.shape[1]
        self.m = self.a_tr.shape[0]
        if setting.get("normalize_loss", True):
            self.train_scale = 1.0 / self.settings.num_train
            self.val_scale = 1.0 / self.settings.num_validate
            self.test_scale = 1.0 / self.settings.num_test
        else:
            self.train_scale = 1.0
            self.val_scale = 1.0
            self.test_scale = 1.0
        self.dual_scale = 1.0 / self.train_scale

    def beta(self, k):
        beta0 = self.setting.get("beta0", 1.0)
        power = self.setting.get("beta_power", self.setting.get("p", 0.3))
        beta = beta0 * (1.0 + k) ** power
        beta_max = self.setting.get("beta_max", None)
        if beta_max is not None:
            beta = min(beta, float(beta_max))
        return beta

    def initial_lambda(self):
        if "initial_lambda" in self.setting:
            lam = np.asarray(self.setting["initial_lambda"], dtype=float).reshape(-1)
        elif "lambda0" in self.setting:
            lam = np.asarray(self.setting["lambda0"], dtype=float).reshape(-1)
        else:
            lam = np.asarray(self.setting.get("initial_lam", 0.1), dtype=float).reshape(-1)
        if lam.size == 1:
            lam = np.full(self.group_count, float(lam[0]))
        if lam.size != self.group_count:
            raise ValueError("initial_lambda must be scalar or have one value per overlap group")
        return np.maximum(lam, 1e-8)

    def block_norms(self, x):
        return np.array([np.linalg.norm(x[sl]) for sl in self.group_slices], dtype=float)

    def regularizer_value(self, x):
        return self.block_norms(x)

    def collapse(self, x):
        return collapse_lifted_coefficients(x, self.groups, self.num_features)

    def lower_solve(self, lam):
        max_iter = self.setting.get("init_max_iter", 300)
        tol = self.setting.get("init_tol", 1e-7)
        lipschitz = self.train_scale * _spectral_norm_squared(self.a_tr)
        step = 1.0 / max(lipschitz, 1e-8)
        x = np.zeros(self.q)
        for _ in range(max_iter):
            old = x.copy()
            x = x - step * self.train_scale * (self.a_tr.T @ (self.a_tr @ x - self.b_tr))
            tau = step * lam
            for i, sl in enumerate(self.group_slices):
                norm_g = np.linalg.norm(x[sl])
                shrink = max(0.0, 1.0 - tau[i] / max(norm_g, 1e-12))
                x[sl] *= shrink
            if _relative_step(x, old) < tol:
                break
        return x

    def initial_state(self):
        lam = self.initial_lambda()
        x = self.lower_solve(lam)
        r = self.regularizer_value(x)
        rho = np.zeros(self.q)
        xi = np.zeros(self.m)
        return x, r, lam, rho, xi

    def h_value(self, xi, rho):
        return self.a_tr.T @ xi + rho

    def q_value(self, x, r, lam, rho, xi):
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

    def merit_value(self, x, r, lam, rho, xi, beta):
        val_res = self.a_val @ x - self.b_val
        upper = 0.5 * self.val_scale * float(np.dot(val_res, val_res)) / max(beta, 1e-12)
        return upper + self.q_value(x, r, lam, rho, xi)

    def line_search_smooth_value(self, x, r, lam, rho, xi, beta):
        return self.merit_value(x, r, lam, rho, xi, beta) - 0.5 * self.dual_scale * float(
            np.dot(xi, xi)
        )

    def gradients(self, x, r, lam, rho, xi, beta):
        h = self.h_value(xi, rho)
        grad_upper = self.val_scale * (self.a_val.T @ (self.a_val @ x - self.b_val))
        grad_x = grad_upper / beta + self.train_scale * (self.a_tr.T @ (self.a_tr @ x - self.b_tr)) - h
        grad_r = lam.copy()
        grad_lam = r.copy()
        grad_rho = -x + h
        grad_xi_smooth = self.b_tr - self.a_tr @ x + self.a_tr @ h
        return grad_x, grad_r, grad_lam, grad_rho, grad_xi_smooth

    def prox_xi(self, xi, grad_xi_smooth, step):
        return (xi - step * grad_xi_smooth) / (1.0 + step * self.dual_scale)

    def pack(self, x, r, lam, rho):
        return np.concatenate([x, r, lam, rho])

    def unpack(self, vec):
        pos = 0
        x = vec[pos : pos + self.q].copy()
        pos += self.q
        r = vec[pos : pos + self.group_count].copy()
        pos += self.group_count
        lam = vec[pos : pos + self.group_count].copy()
        pos += self.group_count
        rho = vec[pos : pos + self.q].copy()
        return x, r, lam, rho

    def project_primal(self, x, r):
        for i, sl in enumerate(self.group_slices):
            x[sl], r[i] = project_l2_epigraph(x[sl], r[i])
        return x, r

    def project_dual(self, rho, lam):
        for i, sl in enumerate(self.group_slices):
            rho[sl], lam[i] = project_l2_epigraph(rho[sl], lam[i])
        return rho, lam

    def project(self, x, r, lam, rho):
        x, r = self.project_primal(x, r)
        rho, lam = self.project_dual(rho, lam)
        return x, r, lam, rho

    def line_search_update(
        self,
        x,
        r,
        lam,
        rho,
        xi,
        beta,
        grads,
        current_step,
    ):
        grad_x, grad_r, grad_lam, grad_rho, grad_xi_smooth = grads
        decay = float(self.setting.get("line_search_decay", 0.5))
        growth = float(self.setting.get("line_search_growth", 1.25))
        min_step = float(self.setting.get("line_search_min_step", 1e-12))
        max_step = float(self.setting.get("line_search_max_step", max(current_step, 1e-12)))
        max_trials = int(self.setting.get("max_line_search_iter", 50))
        current_value = self.line_search_smooth_value(x, r, lam, rho, xi, beta)
        trial_step = min(max_step, current_step * growth)
        fallback = None
        for _ in range(max_trials):
            x_trial = x - trial_step * grad_x
            r_trial = r - trial_step * grad_r
            lam_trial = lam - trial_step * grad_lam
            rho_trial = rho - trial_step * grad_rho
            xi_trial = self.prox_xi(xi, grad_xi_smooth, trial_step)
            x_trial, r_trial, lam_trial, rho_trial = self.project(
                x_trial, r_trial, lam_trial, rho_trial
            )
            trial_value = self.line_search_smooth_value(
                x_trial, r_trial, lam_trial, rho_trial, xi_trial, beta
            )
            finite_trial = (
                np.isfinite(trial_value)
                and np.all(np.isfinite(x_trial))
                and np.all(np.isfinite(r_trial))
                and np.all(np.isfinite(lam_trial))
                and np.all(np.isfinite(rho_trial))
                and np.all(np.isfinite(xi_trial))
            )
            if finite_trial:
                gx = (x - x_trial) / trial_step
                gr = (r - r_trial) / trial_step
                glam = (lam - lam_trial) / trial_step
                grho = (rho - rho_trial) / trial_step
                gxi = (xi - xi_trial) / trial_step
                grad_dot_mapping = float(
                    np.dot(grad_x, gx)
                    + np.dot(grad_r, gr)
                    + np.dot(grad_lam, glam)
                    + np.dot(grad_rho, grho)
                    + np.dot(grad_xi_smooth, gxi)
                )
                mapping_norm_sq = float(
                    np.dot(gx, gx)
                    + np.dot(gr, gr)
                    + np.dot(glam, glam)
                    + np.dot(grho, grho)
                    + np.dot(gxi, gxi)
                )
                rhs = current_value - trial_step * grad_dot_mapping + 0.5 * trial_step * mapping_norm_sq
                fallback = (x_trial, r_trial, lam_trial, rho_trial, xi_trial, trial_step, trial_value)
                if (not np.isfinite(current_value)) or trial_value <= rhs:
                    return fallback
            trial_step *= decay
            if trial_step < min_step:
                break
        if fallback is not None:
            return fallback
        return x, r, lam, rho, xi, current_step, current_value

    def make_record(self, elapsed, x, r, lam, rho, xi, iteration):
        h = self.h_value(xi, rho)
        p_value = self.q_value(x, r, lam, rho, xi) + np.dot(x, h) - 0.5 * np.dot(h, h)
        w = self.collapse(x)
        train_res = self.data.X_train @ w - self.data.y_train
        val_res = self.data.X_validate @ w - self.data.y_validate
        test_res = self.data.X_test @ w - self.data.y_test
        block_norms = self.block_norms(x)
        rho_norms = self.block_norms(rho)
        primal_violation = max(
            0.0,
            float(np.max(block_norms - r)),
            float(np.max(-r)),
        )
        dual_violation = max(
            0.0,
            float(np.max(rho_norms - lam)),
            float(np.max(-lam)),
        )
        return {
            "iteration": iteration,
            "time": elapsed,
            "train_error": 0.5 * self.train_scale * float(np.dot(train_res, train_res)),
            "validation_error": 0.5 * self.val_scale * float(np.dot(val_res, val_res)),
            "test_error": 0.5 * self.test_scale * float(np.dot(test_res, test_res)),
            "ll_duality_gap": self.q_value(x, r, lam, rho, xi),
            "ll_feasibility": max(abs(float(p_value)), float(np.linalg.norm(h))),
            "primal_violation": primal_violation,
            "dual_violation": dual_violation,
            "lambda": lam.copy(),
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "lambda_mean": float(np.mean(lam)),
            "lambda_l2": float(np.linalg.norm(lam)),
            "lambda_values": ";".join("%.17g" % value for value in np.asarray(lam, dtype=float)),
            "r": r.copy(),
            "r_sum": float(np.sum(r)),
            "overlap_penalty": float(np.sum(block_norms)),
            "weighted_overlap_penalty": float(np.dot(lam, block_norms)),
            "active_groups": int(np.sum(block_norms > 1e-6)),
            "latent_sparsity": 100.0 * np.mean(np.abs(x) > 1e-6),
            "sparsity": 100.0 * np.mean(np.abs(w) > 1e-6),
        }

    def run_pgm(self):
        x, r, lam, rho, xi = self.initial_state()
        step = self.setting.get("step_size", self.setting.get("gd_step", 0.01))
        line_search = bool(self.setting.get("line_search", False))
        tol = self.setting.get("TOL", 1e-2)
        max_iter = self.setting.get("MAX_ITERATION", 100)
        record_interval = int(self.setting.get("record_interval", 1))
        record_interval = max(1, record_interval)
        records = []
        start = time.time()
        for k in range(max_iter):
            x_old = x.copy()
            lam_old = lam.copy()
            old = self.pack(x, r, lam, rho)
            beta = self.beta(k)
            grad_x, grad_r, grad_lam, grad_rho, grad_xi_smooth = self.gradients(
                x, r, lam, rho, xi, beta
            )
            if line_search:
                x, r, lam, rho, xi, step, merit = self.line_search_update(
                    x,
                    r,
                    lam,
                    rho,
                    xi,
                    beta,
                    (grad_x, grad_r, grad_lam, grad_rho, grad_xi_smooth),
                    step,
                )
            else:
                x = x - step * grad_x
                r = r - step * grad_r
                lam = lam - step * grad_lam
                rho = rho - step * grad_rho
                xi = self.prox_xi(xi, grad_xi_smooth, step)
                x, r, lam, rho = self.project(x, r, lam, rho)
                merit = np.nan
            new = self.pack(x, r, lam, rho)
            step_err = _relative_step(new, old)
            x_lambda_step_sq = float(
                np.linalg.norm(x - x_old) ** 2 + np.linalg.norm(lam - lam_old) ** 2
            )
            x_lambda_max_relative_stop = _x_lambda_max_relative_stop(x, lam, x_old, lam_old)
            should_stop = (not np.isfinite(x_lambda_max_relative_stop)) or (
                x_lambda_max_relative_stop <= tol
            )
            should_record = should_stop or ((k + 1) % record_interval == 0) or (k + 1 == max_iter)
            if should_record:
                record = self.make_record(time.time() - start, x, r, lam, rho, xi, k + 1)
                record["step_err"] = step_err
                record["x_lambda_step_sq"] = x_lambda_step_sq
                record["x_lambda_max_relative_stop"] = x_lambda_max_relative_stop
                record["accepted_step_size"] = step
                record["merit_value"] = merit
                records.append(record)
            if should_stop:
                break
        self.latent_coef_ = x.copy()
        self.coef_ = self.collapse(x)
        self.lambda_ = lam.copy()
        self.r_ = r.copy()
        self.rho_ = rho.copy()
        self.xi_ = xi.copy()
        df = pd.DataFrame(records)
        df.attrs["latent_coef"] = self.latent_coef_.copy()
        df.attrs["coef"] = self.coef_.copy()
        df.attrs["lambda"] = self.lambda_.copy()
        df.attrs["r"] = self.r_.copy()
        df.attrs["groups"] = [group.copy() for group in self.groups]
        return df


def run_group_lasso_ldp_pgm(data_info, setting):
    regs = group_regularizers(data_info.settings.num_features, data_info.settings.num_experiment_groups)
    solver = LeastSquaresLDPM(data_info, regs, setting)
    return solver.run_pgm()


def run_overlapping_group_lasso_ldp_pgm(data_info, setting):
    setting = dict(setting or {})
    groups = setting.get(
        "groups",
        setting.get("overlap_groups", getattr(data_info.settings, "overlap_groups", None)),
    )
    if groups is None:
        groups = make_overlapping_groups(
            data_info.settings.num_features,
            setting.get("num_groups", getattr(data_info.settings, "num_experiment_groups", 5)),
            group_size=setting.get("group_size", setting.get("overlap_group_size", None)),
            stride=setting.get("stride", setting.get("overlap_stride", None)),
        )
    solver = OverlappingGroupLassoLDPM(data_info, groups, setting)
    return solver.run_pgm()


def run_sparse_group_lasso_ldp_admm(data_info, setting):
    regs = group_regularizers(data_info.settings.num_features, data_info.settings.num_experiment_groups)
    regs.append({"type": "l1", "slice": slice(None)})
    solver = LeastSquaresLDPM(data_info, regs, setting)
    return solver.run_admm()


def run_elastic_net_ldp_admm(data_info, setting):
    regs = [{"type": "l1", "slice": slice(None)}, {"type": "squared_l2", "slice": slice(None)}]
    solver = LeastSquaresLDPM(data_info, regs, setting)
    return solver.run_admm()


def run_elastic_net_ifdm(data_info, setting):
    regs = [{"type": "l1", "slice": slice(None)}, {"type": "squared_l2", "slice": slice(None)}]
    solver = LeastSquaresLDPM(data_info, regs, setting)
    n_outer = setting.get("n_outer", setting.get("MAX_ITERATION", 50))
    step_size = setting.get("step_size", 0.1)
    fd_eps = setting.get("fd_eps", 1e-3)
    bounds = setting.get("log_bounds", (-9.0, -2.0))
    if "alpha0" in setting:
        log_lam = np.log10(np.maximum(np.asarray(setting["alpha0"], dtype=float), 1e-12))
    else:
        log_lam = np.array([-2.0, -2.0])
    log_lam = np.clip(log_lam, bounds[0], bounds[1])
    records = []
    start = time.time()

    def objective(log_params):
        lam = np.power(10.0, log_params)
        x = solver.lower_solve(lam)
        return validation_error(solver.settings, solver.data, x), x

    for k in range(n_outer):
        val, x = objective(log_lam)
        records.append(
            {
                "iteration": k + 1,
                "time": time.time() - start,
                "train_error": train_error(solver.settings, solver.data, x),
                "validation_error": val,
                "test_error": test_error(solver.settings, solver.data, x),
            }
        )
        grad = np.zeros_like(log_lam)
        for j in range(log_lam.size):
            plus = log_lam.copy()
            minus = log_lam.copy()
            plus[j] = min(bounds[1], plus[j] + fd_eps)
            minus[j] = max(bounds[0], minus[j] - fd_eps)
            val_plus, _ = objective(plus)
            val_minus, _ = objective(minus)
            denom = max(plus[j] - minus[j], 1e-12)
            grad[j] = (val_plus - val_minus) / denom
        grad_norm = np.linalg.norm(grad)
        if grad_norm < setting.get("TOL", 1e-4):
            break
        log_lam = np.clip(log_lam - step_size * grad / max(1.0, grad_norm), bounds[0], bounds[1])
    return pd.DataFrame(records)
