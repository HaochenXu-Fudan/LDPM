"""Reduced-statistics baselines for the real-data Group Lasso experiment.

The problem is

    min_lambda  1/(2 n_val) ||X_val x(lambda) - y_val||^2
    s.t.        x(lambda) in argmin_x
                1/(2 n_tr) ||X_tr x - y_tr||^2
                + sum_g lambda_g ||x_g||_2.

The iterative baselines use the method-specific stopping rules stated on page
29 of the AGILS paper; LDPM keeps the separately requested x/lambda residual.
The sufficient-statistics representation keeps the conic baselines practical
on covtype without changing the quadratic losses.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - handled by the conic methods
    cp = None


Array = np.ndarray


def x_lambda_stop(x_new: Array, lam_new: Array, x_old: Array, lam_old: Array) -> float:
    """The exact stopping criterion requested for the comparison."""

    x_den = max(float(np.linalg.norm(x_old)), 1.0)
    lam_den = max(float(np.linalg.norm(lam_old)), 1.0)
    return float(
        np.linalg.norm(x_new - x_old) / x_den
        + np.linalg.norm(lam_new - lam_old) / lam_den
    )


def paper_relative_step(
    x_new: Array, lam_new: Array, x_old: Array, lam_old: Array
) -> float:
    """Relative z=(lambda, coefficient) change used on page 29 of AGILS."""

    numerator = np.sqrt(
        np.linalg.norm(x_new - x_old) ** 2 + np.linalg.norm(lam_new - lam_old) ** 2
    )
    denominator = np.sqrt(
        1.0 + np.linalg.norm(x_old) ** 2 + np.linalg.norm(lam_old) ** 2
    )
    return float(numerator / denominator)


def _require_cvxpy():
    if cp is None:
        raise ImportError("cvxpy is required for VF-iDCA and LDMMA")
    return cp


def _solver_kwargs(name: str, setting: Dict[str, object]) -> Dict[str, object]:
    name = name.upper()
    kwargs: Dict[str, object] = {
        "verbose": bool(setting.get("solver_verbose", False)),
        "warm_start": True,
    }
    tol = float(setting.get("solver_tol", 1e-7))
    max_iter = int(setting.get("solver_max_iter", 10000))
    if name == "CLARABEL":
        kwargs.update(
            tol_gap_abs=tol,
            tol_gap_rel=tol,
            tol_feas=tol,
            max_iter=max_iter,
        )
    elif name == "SCS":
        kwargs.update(eps=tol, max_iters=max_iter)
    return kwargs


def _solve(problem, setting: Dict[str, object], label: str) -> float:
    cp_mod = _require_cvxpy()
    requested = setting.get("solver", None)
    candidates = [str(requested), "CLARABEL", "SCS"] if requested else ["CLARABEL", "SCS"]
    candidates = list(dict.fromkeys(candidates))
    installed = {name.upper(): name for name in cp_mod.installed_solvers()}
    errors: List[str] = []
    for candidate in candidates:
        name = candidate.upper()
        if name not in installed:
            continue
        try:
            value = problem.solve(solver=installed[name], **_solver_kwargs(name, setting))
            if problem.status in {cp_mod.OPTIMAL, cp_mod.OPTIMAL_INACCURATE}:
                return float(value)
            errors.append("%s status=%s" % (name, problem.status))
        except Exception as exc:  # pragma: no cover - solver-specific fallback
            errors.append("%s: %s" % (name, exc))
    raise RuntimeError("%s failed (%s)" % (label, "; ".join(errors) or "no solver"))


class ReducedGroupLassoProblem:
    """Exact quadratic sufficient statistics plus group-Lasso primitives."""

    def __init__(self, data_info, groups: Sequence[Dict[str, object]], setting=None):
        self.data_info = data_info
        self.data = data_info.data
        self.settings = data_info.settings
        self.setting = dict(setting or {})
        self.regularizers = [dict(item) for item in groups]
        self.slices = [
            item["slice"] for item in self.regularizers if item["type"] == "group_l2"
        ]
        self.p = int(self.settings.num_features)
        self.group_count = len(self.regularizers)
        self.time_origin = float(self.setting.get("time_origin", time.perf_counter()))

        self.train_stats = self._stats(self.data.X_train, self.data.y_train)
        self.val_stats = self._stats(self.data.X_validate, self.data.y_validate)
        self.test_stats = self._stats(self.data.X_test, self.data.y_test)
        self.train_lipschitz = max(float(np.linalg.eigvalsh(self.train_stats[0])[-1]), 1e-12)
        self.val_lipschitz = max(float(np.linalg.eigvalsh(self.val_stats[0])[-1]), 1e-12)

    @staticmethod
    def _stats(x: Array, y: Array) -> Tuple[Array, Array, float]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = max(1, y.size)
        gram = np.asarray(x.T @ x, dtype=float) / n
        gram = 0.5 * (gram + gram.T)
        cross = np.asarray(x.T @ y, dtype=float).reshape(-1) / n
        yy = float(np.dot(y, y) / n)
        return gram, cross, yy

    def elapsed(self) -> float:
        return float(time.perf_counter() - self.time_origin)

    @staticmethod
    def loss_from_stats(stats: Tuple[Array, Array, float], x: Array) -> float:
        gram, cross, yy = stats
        return float(0.5 * (np.dot(x, gram @ x) - 2.0 * np.dot(cross, x) + yy))

    def train_loss(self, x: Array) -> float:
        return self.loss_from_stats(self.train_stats, x)

    def validation_loss(self, x: Array) -> float:
        return self.loss_from_stats(self.val_stats, x)

    def test_loss(self, x: Array) -> float:
        return self.loss_from_stats(self.test_stats, x)

    def train_grad(self, x: Array) -> Array:
        gram, cross, _ = self.train_stats
        return gram @ x - cross

    def val_grad(self, x: Array) -> Array:
        gram, cross, _ = self.val_stats
        return gram @ x - cross

    def group_norms(self, x: Array) -> Array:
        values = []
        for regularizer in self.regularizers:
            if regularizer["type"] == "group_l2":
                values.append(np.linalg.norm(x[regularizer["slice"]]))
            elif regularizer["type"] == "l1":
                values.append(np.linalg.norm(x, 1))
            else:
                raise ValueError("unsupported regularizer %r" % regularizer["type"])
        return np.asarray(values, dtype=float)

    def lower_objective(self, lam: Array, x: Array) -> float:
        return float(self.train_loss(x) + np.dot(lam, self.group_norms(x)))

    def prox(self, value: Array, step: float, lam: Array) -> Array:
        out = np.asarray(value, dtype=float).copy()
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] == "l1":
                threshold = step * float(lam[index])
                out = np.sign(out) * np.maximum(np.abs(out) - threshold, 0.0)
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] != "group_l2":
                continue
            sl = regularizer["slice"]
            norm = float(np.linalg.norm(out[sl]))
            shrink = max(0.0, 1.0 - step * float(lam[index]) / max(norm, 1e-15))
            out[sl] *= shrink
        return out

    def cvx_train_loss(self, x):
        cp_mod = _require_cvxpy()
        gram, cross, yy = self.train_stats
        return 0.5 * cp_mod.quad_form(x, cp_mod.psd_wrap(gram)) - cross @ x + 0.5 * yy

    def cvx_validation_loss(self, x):
        cp_mod = _require_cvxpy()
        gram, cross, yy = self.val_stats
        return 0.5 * cp_mod.quad_form(x, cp_mod.psd_wrap(gram)) - cross @ x + 0.5 * yy

    def cvx_epigraph_constraints(self, x, r):
        cp_mod = _require_cvxpy()
        constraints = []
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] == "group_l2":
                constraints.append(cp_mod.norm(x[regularizer["slice"]], 2) <= r[index])
            elif regularizer["type"] == "l1":
                constraints.append(cp_mod.norm1(x) <= r[index])
            else:
                raise ValueError("unsupported regularizer %r" % regularizer["type"])
        return constraints

    def cvx_epigraph_residuals(self, x, r):
        cp_mod = _require_cvxpy()
        residuals = []
        for index, regularizer in enumerate(self.regularizers):
            if regularizer["type"] == "group_l2":
                residuals.append(cp_mod.norm(x[regularizer["slice"]], 2) - r[index])
            elif regularizer["type"] == "l1":
                residuals.append(cp_mod.norm1(x) - r[index])
            else:
                raise ValueError("unsupported regularizer %r" % regularizer["type"])
        return residuals

    def lower_solve(
        self,
        lam: Array,
        x0: Optional[Array] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Tuple[Array, int]:
        lam = np.asarray(lam, dtype=float).reshape(self.group_count)
        max_iter = int(max_iter or self.setting.get("lower_max_iter", 5000))
        tol = float(tol or self.setting.get("lower_tol", 1e-10))
        step = 1.0 / self.train_lipschitz
        x = np.zeros(self.p) if x0 is None else np.asarray(x0, dtype=float).copy()
        extrapolated = x.copy()
        momentum = 1.0
        for iteration in range(1, max_iter + 1):
            old = x.copy()
            x = self.prox(extrapolated - step * self.train_grad(extrapolated), step, lam)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = x + ((momentum - 1.0) / next_momentum) * (x - old)
            momentum = next_momentum
            if np.linalg.norm(x - old) / max(1.0, np.linalg.norm(old)) <= tol:
                return x, iteration
        return x, max_iter

    def proximal_lower_solve(
        self,
        lam: Array,
        center: Array,
        gamma: float,
        x0: Optional[Array] = None,
        max_iter: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Tuple[Array, int]:
        """Solve the Moreau proximal lower problem for metric post-processing."""

        lam = np.asarray(lam, dtype=float).reshape(self.group_count)
        center = np.asarray(center, dtype=float).reshape(self.p)
        max_iter = int(max_iter or self.setting.get("lower_max_iter", 5000))
        tol = float(tol or self.setting.get("lower_tol", 1e-10))
        step = 1.0 / (self.train_lipschitz + 1.0 / gamma)
        x = center.copy() if x0 is None else np.asarray(x0, dtype=float).copy()
        extrapolated = x.copy()
        momentum = 1.0
        for iteration in range(1, max_iter + 1):
            old = x.copy()
            gradient = self.train_grad(extrapolated) + (extrapolated - center) / gamma
            x = self.prox(extrapolated - step * gradient, step, lam)
            next_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            extrapolated = x + ((momentum - 1.0) / next_momentum) * (x - old)
            momentum = next_momentum
            if np.linalg.norm(x - old) / max(1.0, np.linalg.norm(old)) <= tol:
                return x, iteration
        return x, max_iter

    def record(self, iteration: int, x: Array, lam: Array, stop: float, **extra) -> Dict[str, object]:
        row: Dict[str, object] = {
            "iteration": int(iteration),
            "time": self.elapsed(),
            "train_error": self.train_loss(x),
            "validation_error": self.validation_loss(x),
            "test_error": self.test_loss(x),
            "x_lambda_stop": float(stop),
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "lambda_l2": float(np.linalg.norm(lam)),
            "lambda_values": ";".join("%.17g" % value for value in np.asarray(lam, dtype=float)),
        }
        row.update(extra)
        return row


def _finish(records, x: Array, lam: Array, status: str, method: str, **state):
    frame = pd.DataFrame(records)
    frame.attrs["coef"] = np.asarray(x, dtype=float).copy()
    frame.attrs["lambda"] = np.asarray(lam, dtype=float).copy()
    frame.attrs["termination_status"] = status
    frame.attrs["method"] = method
    for key, value in state.items():
        frame.attrs[key] = value
    return frame


def _initial_record(problem: ReducedGroupLassoProblem, x: Array, lam: Array, **extra):
    """Record the shared physical BLP start exactly at t=0 for every method."""

    row = problem.record(0, x, lam, 0.0, common_initial_point=True, **extra)
    row["time"] = 0.0
    return row


def _make_search_lower_solver(problem, setting):
    """Build the fixed-lambda lower solver used by Table-3 search methods."""

    if str(setting.get("search_lower_solver", "fista")).lower() != "cvxpy":
        return lambda lam, warm: problem.lower_solve(lam, x0=warm)

    cp_mod = _require_cvxpy()
    lower_x = cp_mod.Variable(problem.p)
    lower_lam = cp_mod.Parameter(problem.group_count, nonneg=True)
    penalties = []
    for index, regularizer in enumerate(problem.regularizers):
        if regularizer["type"] == "group_l2":
            penalty = cp_mod.norm(lower_x[regularizer["slice"]], 2)
        elif regularizer["type"] == "l1":
            penalty = cp_mod.norm1(lower_x)
        else:
            raise ValueError("unsupported Table-3 regularizer %r" % regularizer["type"])
        penalties.append(lower_lam[index] * penalty)
    lower_problem = cp_mod.Problem(
        cp_mod.Minimize(problem.cvx_train_loss(lower_x) + cp_mod.sum(penalties))
    )

    def solve(lam, warm):
        lower_lam.value = np.asarray(lam, dtype=float)
        if warm is not None:
            lower_x.value = np.asarray(warm, dtype=float)
        _solve(lower_problem, setting, "Table-3 fixed-lambda lower problem")
        if lower_x.value is None:
            raise RuntimeError("Table-3 fixed-lambda lower problem returned no coefficient")
        stats = lower_problem.solver_stats
        solve.last_solver_name = str(stats.solver_name)
        solve.last_status = str(lower_problem.status)
        solve.last_solve_time = (
            None if stats.solve_time is None else float(stats.solve_time)
        )
        iterations = int(stats.num_iters or 0)
        return np.asarray(lower_x.value, dtype=float).reshape(-1), iterations

    solve.last_solver_name = None
    solve.last_status = None
    solve.last_solve_time = None
    return solve


def _finish_search(problem, candidates, method: str):
    """Evaluate fixed paper-budget candidates and retain incumbent history."""

    records = []
    best_value = np.inf
    best_coef = None
    best_lam = None
    warm = None
    solve_lower = _make_search_lower_solver(problem, problem.setting)
    for evaluation, lam in enumerate(candidates, start=1):
        lam = np.asarray(lam, dtype=float).reshape(problem.group_count)
        coef, lower_iterations = solve_lower(lam, warm)
        warm = coef.copy()
        value = problem.validation_loss(coef)
        if value < best_value:
            best_value = value
            best_coef = coef.copy()
            best_lam = lam.copy()
            records.append(
                problem.record(
                    evaluation,
                    best_coef,
                    best_lam,
                    0.0,
                    search_evaluations=evaluation,
                    search_validation_error=best_value,
                    feasible_lower_iterations=lower_iterations,
                )
            )
    if best_coef is None or best_lam is None:
        raise RuntimeError("%s evaluated no candidates" % method)
    return _finish(records, best_coef, best_lam, "budget_complete", method)


def run_grid_search(problem: ReducedGroupLassoProblem, setting=None):
    """Page-29 grid over one common group weight and the optional L1 weight."""

    setting = dict(setting or {})
    points = int(setting.get("grid_points", 20))
    grid = np.linspace(-9.0, 2.0, points)
    l1_indices = [
        index
        for index, regularizer in enumerate(problem.regularizers)
        if regularizer["type"] == "l1"
    ]
    if l1_indices:
        if len(l1_indices) != 1:
            raise ValueError("page-29 grid expects exactly one L1 coordinate")
        l1_index = l1_indices[0]
        candidates = []
        for group_rho in grid:
            for l1_rho in grid:
                lam = np.full(problem.group_count, 10.0**group_rho)
                lam[l1_index] = 10.0**l1_rho
                candidates.append(lam)
    else:
        candidates = [
            np.full(problem.group_count, 10.0**rho)
            for rho in grid
        ]
    return _finish_search(problem, candidates, "Grid")


def run_random_search(problem: ReducedGroupLassoProblem, setting=None):
    """Page-29 400-point independent uniform search in log10 space."""

    setting = dict(setting or {})
    budget = int(setting.get("search_budget", 400))
    rng = np.random.default_rng(int(setting.get("seed", 0)))
    rhos = rng.uniform(-9.0, 2.0, size=(budget, problem.group_count))
    candidates = [np.power(10.0, rho) for rho in rhos]
    return _finish_search(problem, candidates, "Random")


def run_tpe_search(problem: ReducedGroupLassoProblem, setting=None):
    """Page-29 TPE with independent uniform log10 priors and 400 trials."""

    setting = dict(setting or {})
    try:
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
    except ImportError:
        import os
        import sys

        dependency_path = os.environ.get(
            "SGL_HYPEROPT_PATH", "/private/tmp/ldpm-hyperopt-deps"
        )
        if dependency_path not in sys.path:
            sys.path.insert(0, dependency_path)
        from hyperopt import STATUS_OK, Trials, fmin, hp, tpe

    budget = int(setting.get("search_budget", 400))
    seed = int(setting.get("seed", 0))
    records = []
    best_value = np.inf
    best_coef = None
    best_lam = None
    evaluation = 0
    warm = None
    solve_lower = _make_search_lower_solver(problem, setting)

    def objective(values):
        nonlocal evaluation, best_value, best_coef, best_lam, warm
        evaluation += 1
        rho = np.asarray([values["rho_%d" % i] for i in range(problem.group_count)])
        lam = np.power(10.0, rho)
        coef, lower_iterations = solve_lower(lam, warm)
        warm = coef.copy()
        value = problem.validation_loss(coef)
        if value < best_value:
            best_value = value
            best_coef = coef.copy()
            best_lam = lam.copy()
            records.append(
                problem.record(
                    evaluation,
                    best_coef,
                    best_lam,
                    0.0,
                    search_evaluations=evaluation,
                    search_validation_error=best_value,
                    feasible_lower_iterations=lower_iterations,
                )
            )
        return {"loss": value, "status": STATUS_OK}

    space = {
        "rho_%d" % i: hp.uniform("rho_%d" % i, -9.0, 2.0)
        for i in range(problem.group_count)
    }
    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=budget,
        trials=Trials(),
        rstate=np.random.default_rng(seed),
        show_progressbar=False,
        verbose=False,
    )
    if best_coef is None or best_lam is None:
        raise RuntimeError("TPE evaluated no candidates")
    return _finish(records, best_coef, best_lam, "budget_complete", "TPE")


def run_vfidca(problem: ReducedGroupLassoProblem, setting=None):
    """Author-code VF-iDCA update, adapted only from SGL to pure Group Lasso.

    Source: SUSTech-Optimization/VF-iDCA, ``utils/SGL_Algorithms.py``.
    The common initialization and requested stopping rule are experiment-level
    overrides; rho, beta, c, delta and the majorized subproblem follow the
    authors' SGL experiment implementation.
    """

    cp_mod = _require_cvxpy()
    setting = dict(setting or {})
    setting["solver_tol"] = float(setting.get("vfidca_solver_tol", 1e-4))
    setting["solver_max_iter"] = int(setting.get("vfidca_solver_max_iter", 100))
    max_iter = int(setting.get("max_iter", 200))
    tol = float(setting.get("tol", 1e-5))
    rho_prox = float(setting.get("vfidca_rho", 0.1))
    beta = float(setting.get("vfidca_beta0", 1.0))
    beta_delta = float(setting.get("vfidca_beta_delta", 5.0))
    beta_update_c = float(setting.get("vfidca_c", 0.01))
    epsilon = float(setting.get("vfidca_epsilon", 0.0))
    violation_weight = float(setting.get("vfidca_violation_weight", 100.0))
    lam0 = np.full(problem.group_count, float(setting.get("initial_lambda", 0.1)))
    x = np.asarray(setting.get("initial_coef", np.ones(problem.p)), dtype=float).copy()
    r = np.asarray(
        setting.get("initial_radius", np.full(problem.group_count, 0.1)),
        dtype=float,
    ).reshape(problem.group_count)
    r = np.maximum(r, 1e-8)
    previous_lam = lam0.copy()

    lower_x = cp_mod.Variable(problem.p)
    lower_r = cp_mod.Parameter(problem.group_count, nonneg=True)
    lower_constraints = problem.cvx_epigraph_constraints(lower_x, lower_r)
    lower_loss = problem.cvx_train_loss(lower_x)
    lower_problem = cp_mod.Problem(cp_mod.Minimize(lower_loss), lower_constraints)

    upper_x = cp_mod.Variable(problem.p)
    upper_r = cp_mod.Variable(problem.group_count)
    x_ref = cp_mod.Parameter(problem.p)
    r_ref = cp_mod.Parameter(problem.group_count, nonneg=True)
    gamma_ref = cp_mod.Parameter(problem.group_count, nonneg=True)
    lower_value_ref = cp_mod.Parameter()
    beta_ref = cp_mod.Parameter(nonneg=True)
    train_loss = problem.cvx_train_loss(upper_x)
    val_loss = problem.cvx_validation_loss(upper_x)
    value_violation = beta_ref * (
        train_loss - lower_value_ref + gamma_ref @ (upper_r - r_ref) - epsilon
    )
    primal_violations = problem.cvx_epigraph_residuals(upper_x, upper_r)
    primal_max = primal_violations[0]
    for expr in primal_violations[1:]:
        primal_max = cp_mod.maximum(primal_max, expr)
    penalty = cp_mod.maximum(
        0.0,
        cp_mod.maximum(value_violation, violation_weight * beta_ref * primal_max),
    )
    prox = cp_mod.sum_squares(upper_x - x_ref) + cp_mod.sum_squares(upper_r - r_ref)
    upper_problem = cp_mod.Problem(
        cp_mod.Minimize(val_loss + 0.5 * rho_prox * prox + penalty),
        [upper_r >= 0.0],
    )

    records = [_initial_record(problem, x, lam0, beta=float(beta), native_violation=np.nan)]
    status = "max_iter"
    lam = previous_lam.copy()
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        r_old = r.copy()
        lam_old = previous_lam.copy()
        lower_r.value = np.maximum(r, 1e-10)
        lower_x.value = x
        lower_value = _solve(lower_problem, setting, "VF-iDCA lower problem")
        lower_solution = np.asarray(lower_x.value, dtype=float).reshape(-1)
        # The author implementation uses the conic constraints' dual values
        # directly.  Removing the SGL L1 constraint leaves one dual per group.
        lam = np.clip(
            np.nan_to_num(
                np.asarray(
                    [float(constraint.dual_value) for constraint in lower_constraints],
                    dtype=float,
                ),
                nan=0.0,
                posinf=10.0,
                neginf=0.0,
            ),
            0.0,
            float(setting.get("lambda_ceiling", 10.0)),
        )

        x_ref.value = x
        r_ref.value = np.maximum(r, 1e-10)
        gamma_ref.value = lam
        lower_value_ref.value = lower_value
        beta_ref.value = beta
        upper_x.value = x
        upper_r.value = r
        _solve(upper_problem, setting, "VF-iDCA majorization problem")
        if upper_x.value is None or upper_r.value is None:
            raise RuntimeError("VF-iDCA returned an empty iterate")
        x = np.asarray(upper_x.value, dtype=float).reshape(-1)
        r = np.maximum(np.asarray(upper_r.value, dtype=float).reshape(-1), 0.0)
        stop = x_lambda_stop(x, lam, x_old, lam_old)
        penalty_value = float(np.asarray(penalty.value)) / max(beta, 1e-12)
        vf_relative_step = float(
            np.sqrt(np.linalg.norm(x - x_old) ** 2 + np.linalg.norm(r - r_old) ** 2)
            / np.sqrt(1.0 + np.linalg.norm(x_old) ** 2 + np.linalg.norm(r_old) ** 2)
        )
        vf_paper_stop = max(vf_relative_step, penalty_value / problem.p)
        requested_stop = (
            vf_paper_stop < float(setting.get("vfidca_paper_tol", 0.1))
            if setting.get("paper29_protocol", False)
            else stop <= tol
        )
        should_record = requested_stop or iteration % int(setting.get("record_interval", 25)) == 0 or iteration == max_iter
        if should_record:
            records.append(
                problem.record(
                    iteration,
                    x,
                    lam,
                    stop,
                    beta=float(beta),
                    native_violation=penalty_value,
                    paper_relative_step=vf_relative_step,
                    paper_stop_metric=vf_paper_stop,
                    cvxpy_lower_status=lower_problem.status,
                    cvxpy_upper_status=upper_problem.status,
                )
            )
        if not np.isfinite(stop):
            status = "nonfinite"
            break
        if requested_stop:
            status = "converged"
            break
        pair_step = np.sqrt(np.linalg.norm(x - x_old) ** 2 + np.linalg.norm(r - r_ref.value) ** 2)
        if pair_step * beta <= beta_update_c * min(1.0, max(penalty_value, 0.0)):
            beta += beta_delta
        previous_lam = lam.copy()
    return _finish(records, x, lam, status, "VF-iDCA", radius=r.copy())


def run_ldmma(problem: ReducedGroupLassoProblem, setting=None):
    """Author-code LDMMA update with an exact reduced Gram representation.

    Source: HaochenXu-Fudan/LDMMA, ``LDMMA_py/SGL_Algorithms.py``.  Only the
    L1 block is removed for pure Group Lasso and the full data matrices are
    replaced by algebraically equivalent sufficient statistics.
    """

    setting = dict(setting or {})
    if getattr(problem, "direct_sgl", False):
        return _run_ldmma_direct_matrix(problem, setting)
    cp_mod = _require_cvxpy()
    setting["solver_tol"] = float(setting.get("ldmma_solver_tol", 1e-2))
    setting["solver_max_iter"] = int(setting.get("ldmma_solver_max_iter", 50))
    max_iter = int(setting.get("max_iter", 200))
    tol = float(setting.get("tol", 1e-5))
    epsilon = float(setting.get("ldmma_epsilon", 1e-3))
    eta = float(setting.get("ldmma_eta", 0.0))
    floor = float(setting.get("lambda_floor", 1e-8))
    lambda_ceiling = float(setting.get("lambda_ceiling", 10.0))
    lam = np.full(problem.group_count, float(setting.get("initial_lambda", 0.1)))
    x = np.asarray(setting.get("initial_coef", np.ones(problem.p)), dtype=float).copy()
    r = np.asarray(
        setting.get("initial_radius", np.full(problem.group_count, 0.1)),
        dtype=float,
    ).reshape(problem.group_count)
    r = np.maximum(r, floor)
    z = x.copy()

    gram_tr, cross_tr, _ = problem.train_stats
    gram_val, cross_val, yy_val = problem.val_stats
    psd_tr = cp_mod.psd_wrap(gram_tr)
    psd_val = cp_mod.psd_wrap(gram_val)

    x_var = cp_mod.Variable(problem.p)
    r_var = cp_mod.Variable(problem.group_count, nonneg=True)
    lam_var = cp_mod.Variable(problem.group_count, nonneg=True)
    z_var = cp_mod.Variable(problem.p)
    coeff_r = cp_mod.Parameter(problem.group_count, nonneg=True)
    coeff_lam = cp_mod.Parameter(problem.group_count, nonneg=True)
    x_ref = cp_mod.Parameter(problem.p)
    r_ref_parameter = cp_mod.Parameter(problem.group_count, nonneg=True)
    lam_ref_parameter = cp_mod.Parameter(problem.group_count, nonneg=True)
    z_ref = cp_mod.Parameter(problem.p)
    rho_expr = cross_tr - gram_tr @ z_var
    majorizer = 0.5 * cp_mod.sum(
        cp_mod.multiply(coeff_r, cp_mod.square(r_var))
        + cp_mod.multiply(coeff_lam, cp_mod.square(lam_var))
    )
    fenchel_majorizer = (
        0.5 * cp_mod.quad_form(x_var, psd_tr)
        - cross_tr @ x_var
        + majorizer
        + 0.5 * cp_mod.quad_form(z_var, psd_tr)
    )
    val_loss = 0.5 * cp_mod.quad_form(x_var, psd_val) - cross_val @ x_var + 0.5 * yy_val
    constraints = [lam_var >= floor, lam_var <= lambda_ceiling]
    constraints.extend(
        cp_mod.norm(x_var[sl], 2) <= r_var[index]
        for index, sl in enumerate(problem.slices)
    )
    constraints.extend(
        cp_mod.norm(rho_expr[sl], 2) <= lam_var[index]
        for index, sl in enumerate(problem.slices)
    )
    constraints.append(fenchel_majorizer <= epsilon)
    proximal_term = (
        cp_mod.sum_squares(x_var - x_ref)
        + cp_mod.sum_squares(r_var - r_ref_parameter)
        + cp_mod.sum_squares(lam_var - lam_ref_parameter)
        + cp_mod.sum_squares(z_var - z_ref)
    )
    conic_problem = cp_mod.Problem(
        cp_mod.Minimize(val_loss + 0.5 * eta * proximal_term), constraints
    )

    records = [
        _initial_record(
            problem,
            x,
            lam,
            native_fenchel_gap=np.nan,
            native_constraint_violation=np.nan,
        )
    ]
    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        r_old = r.copy()
        lam_old = lam.copy()
        lam_ref = np.maximum(lam, floor)
        r_ref = np.maximum(r, floor)
        coeff_r.value = lam_ref / r_ref
        coeff_lam.value = r_ref / lam_ref
        x_ref.value = x
        r_ref_parameter.value = r_ref
        lam_ref_parameter.value = lam_ref
        z_ref.value = z
        x_var.value = x
        r_var.value = r
        lam_var.value = lam
        z_var.value = z
        _solve(conic_problem, setting, "LDMMA reduced conic problem")
        if any(item.value is None for item in (x_var, r_var, lam_var, z_var)):
            raise RuntimeError("LDMMA returned an empty iterate")
        x = np.asarray(x_var.value, dtype=float).reshape(-1)
        r = np.maximum(np.asarray(r_var.value, dtype=float).reshape(-1), floor)
        lam = np.clip(np.asarray(lam_var.value, dtype=float).reshape(-1), floor, lambda_ceiling)
        z = np.asarray(z_var.value, dtype=float).reshape(-1)
        rho = cross_tr - gram_tr @ z
        true_gap = float(
            0.5 * np.dot(x, gram_tr @ x)
            - np.dot(cross_tr, x)
            + np.dot(lam, problem.group_norms(x))
            + 0.5 * np.dot(z, gram_tr @ z)
        )
        dual_violation = max(
            [max(0.0, np.linalg.norm(rho[sl]) - lam[index]) for index, sl in enumerate(problem.slices)]
            or [0.0]
        )
        stop = x_lambda_stop(x, lam, x_old, lam_old)
        ldmma_relative_step = float(
            np.sqrt(
                np.linalg.norm(x - x_old) ** 2
                + np.linalg.norm(r - r_old) ** 2
                + np.linalg.norm(lam - lam_old) ** 2
            )
            / max(
                np.sqrt(
                    np.linalg.norm(x) ** 2
                    + np.linalg.norm(r) ** 2
                    + np.linalg.norm(lam) ** 2
                ),
                1e-15,
            )
        )
        requested_stop = (
            ldmma_relative_step < float(setting.get("ldmma_paper_tol", 0.05))
            if setting.get("paper29_protocol", False)
            else stop <= tol
        )
        should_record = requested_stop or iteration % int(setting.get("record_interval", 25)) == 0 or iteration == max_iter
        if should_record:
            records.append(
                problem.record(
                    iteration,
                    x,
                    lam,
                    stop,
                    native_fenchel_gap=true_gap,
                    native_constraint_violation=float(max(dual_violation, true_gap - epsilon, 0.0)),
                    paper_relative_step=ldmma_relative_step,
                    paper_stop_metric=ldmma_relative_step,
                    cvxpy_status=conic_problem.status,
                )
            )
        if not np.isfinite(stop):
            status = "nonfinite"
            break
        if requested_stop:
            status = "converged"
            break
    return _finish(records, x, lam, status, "LDMMA", radius=r.copy(), dual_anchor=z.copy())


def _run_ldmma_direct_matrix(problem, setting):
    """Released LDMMA MM subproblem using direct matrices instead of a p-by-p Gram matrix."""

    cp_mod = _require_cvxpy()
    setting = dict(setting or {})
    setting["solver_tol"] = float(setting.get("ldmma_solver_tol", 1e-2))
    setting["solver_max_iter"] = int(setting.get("ldmma_solver_max_iter", 50))
    max_iter = int(setting.get("max_iter", 100))
    epsilon = float(setting.get("ldmma_epsilon", 1e-3))
    eta = float(setting.get("ldmma_eta", 0.0))
    floor = float(setting.get("lambda_floor", 1e-8))
    ceiling = float(setting.get("lambda_ceiling", 100.0))

    lam = np.full(problem.group_count, float(setting.get("initial_lambda", 5.0)))
    x = np.asarray(setting.get("initial_coef", np.zeros(problem.p)), dtype=float).copy()
    r = np.asarray(
        setting.get("initial_radius", np.full(problem.group_count, 0.1)), dtype=float
    ).reshape(problem.group_count)
    r = np.maximum(r, floor)

    a_train, b_train = problem.scaled_training_data()
    a_validate, b_validate = problem.scaled_validation_data()
    x_var = cp_mod.Variable(problem.p)
    r_var = cp_mod.Variable(problem.group_count, nonneg=True)
    lam_var = cp_mod.Variable(problem.group_count, nonneg=True)
    w_var = cp_mod.Variable(a_train.shape[0])
    rho_group = cp_mod.Variable(problem.p)
    rho_l1 = cp_mod.Variable(problem.p) if any(
        regularizer["type"] == "l1" for regularizer in problem.regularizers
    ) else None

    coeff_r = cp_mod.Parameter(problem.group_count, nonneg=True)
    coeff_lam = cp_mod.Parameter(problem.group_count, nonneg=True)
    x_ref = cp_mod.Parameter(problem.p)
    r_ref_parameter = cp_mod.Parameter(problem.group_count, nonneg=True)
    lam_ref_parameter = cp_mod.Parameter(problem.group_count, nonneg=True)
    w_ref = cp_mod.Parameter(a_train.shape[0])

    constraints = [lam_var >= floor, lam_var <= ceiling]
    constraints.extend(problem.cvx_epigraph_constraints(x_var, r_var))
    group_indices = []
    l1_indices = []
    for index, regularizer in enumerate(problem.regularizers):
        if regularizer["type"] == "group_l2":
            group_indices.append(index)
            constraints.append(
                cp_mod.norm(rho_group[regularizer["slice"]], 2) <= lam_var[index]
            )
        elif regularizer["type"] == "l1":
            l1_indices.append(index)
        else:
            raise ValueError("unsupported LDMMA regularizer %r" % regularizer["type"])
    if l1_indices:
        if len(l1_indices) != 1 or rho_l1 is None:
            raise ValueError("LDMMA expects at most one L1 regularizer")
        constraints.append(cp_mod.norm_inf(rho_l1) <= lam_var[l1_indices[0]])
    stationarity = a_train.T @ w_var + rho_group
    if rho_l1 is not None:
        stationarity = stationarity + rho_l1
    constraints.append(stationarity == 0.0)

    primal_loss = 0.5 * cp_mod.sum_squares(a_train @ x_var - b_train)
    majorizer = 0.5 * cp_mod.sum(
        cp_mod.multiply(coeff_r, cp_mod.square(r_var))
        + cp_mod.multiply(coeff_lam, cp_mod.square(lam_var))
    )
    dual_conjugate = 0.5 * cp_mod.sum_squares(w_var) + b_train @ w_var
    fenchel_majorizer = primal_loss + majorizer + dual_conjugate
    constraints.append(fenchel_majorizer <= epsilon)

    validation_loss = 0.5 * cp_mod.sum_squares(a_validate @ x_var - b_validate)
    proximal = (
        cp_mod.sum_squares(x_var - x_ref)
        + cp_mod.sum_squares(r_var - r_ref_parameter)
        + cp_mod.sum_squares(lam_var - lam_ref_parameter)
        + cp_mod.sum_squares(w_var - w_ref)
    )
    conic_problem = cp_mod.Problem(
        cp_mod.Minimize(validation_loss + 0.5 * eta * proximal), constraints
    )

    w = np.zeros(a_train.shape[0])
    records = [
        _initial_record(
            problem,
            x,
            lam,
            native_fenchel_gap=np.nan,
            native_constraint_violation=np.nan,
        )
    ]
    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        r_old = r.copy()
        lam_old = lam.copy()
        lam_ref = np.maximum(lam, floor)
        r_ref = np.maximum(r, floor)
        coeff_r.value = lam_ref / r_ref
        coeff_lam.value = r_ref / lam_ref
        x_ref.value = x
        r_ref_parameter.value = r_ref
        lam_ref_parameter.value = lam_ref
        w_ref.value = w
        x_var.value = x
        r_var.value = r
        lam_var.value = lam
        w_var.value = w
        rho_group.value = np.zeros(problem.p)
        if rho_l1 is not None:
            rho_l1.value = np.zeros(problem.p)
        _solve(conic_problem, setting, "LDMMA direct conic problem")
        if any(value.value is None for value in (x_var, r_var, lam_var, w_var)):
            raise RuntimeError("LDMMA returned an empty direct-matrix iterate")
        x = np.asarray(x_var.value, dtype=float).reshape(-1)
        r = np.maximum(np.asarray(r_var.value, dtype=float).reshape(-1), floor)
        lam = np.clip(np.asarray(lam_var.value, dtype=float).reshape(-1), floor, ceiling)
        w = np.asarray(w_var.value, dtype=float).reshape(-1)
        rho_group_value = np.asarray(rho_group.value, dtype=float).reshape(-1)
        rho_l1_value = (
            np.asarray(rho_l1.value, dtype=float).reshape(-1)
            if rho_l1 is not None
            else np.zeros(problem.p)
        )
        stationarity_value = a_train.T @ w + rho_group_value + rho_l1_value
        true_gap = float(
            problem.train_loss(x)
            + np.dot(lam, problem.group_norms(x))
            + 0.5 * np.dot(w, w)
            + np.dot(b_train, w)
        )
        dual_violations = []
        for index, regularizer in enumerate(problem.regularizers):
            if regularizer["type"] == "group_l2":
                dual_violations.append(
                    max(
                        0.0,
                        float(np.linalg.norm(rho_group_value[regularizer["slice"]]))
                        - lam[index],
                    )
                )
            elif regularizer["type"] == "l1":
                dual_violations.append(
                    max(0.0, float(np.linalg.norm(rho_l1_value, np.inf)) - lam[index])
                )
        native_violation = max(
            max(dual_violations or [0.0]),
            float(np.linalg.norm(stationarity_value)),
            true_gap - epsilon,
            0.0,
        )
        stop = x_lambda_stop(x, lam, x_old, lam_old)
        relative_step = float(
            np.sqrt(
                np.linalg.norm(x - x_old) ** 2
                + np.linalg.norm(r - r_old) ** 2
                + np.linalg.norm(lam - lam_old) ** 2
            )
            / max(
                np.sqrt(
                    np.linalg.norm(x) ** 2
                    + np.linalg.norm(r) ** 2
                    + np.linalg.norm(lam) ** 2
                ),
                1e-15,
            )
        )
        requested_stop = relative_step < float(setting.get("ldmma_paper_tol", 0.05))
        should_record = (
            requested_stop
            or iteration % int(setting.get("record_interval", 25)) == 0
            or iteration == max_iter
        )
        if should_record:
            records.append(
                problem.record(
                    iteration,
                    x,
                    lam,
                    stop,
                    native_fenchel_gap=true_gap,
                    native_constraint_violation=native_violation,
                    paper_relative_step=relative_step,
                    paper_stop_metric=relative_step,
                    cvxpy_status=conic_problem.status,
                )
            )
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(lam)):
            status = "nonfinite"
            break
        if requested_stop:
            status = "converged"
            break
    return _finish(
        records,
        x,
        lam,
        status,
        "LDMMA",
        radius=r.copy(),
        dual_multiplier=w.copy(),
    )


def _theta_pg_residual(
    problem: ReducedGroupLassoProblem,
    theta: Array,
    lam: Array,
    center: Array,
    gamma: float,
    eta: float,
) -> float:
    gradient = problem.train_grad(theta) + (theta - center) / gamma
    prox_point = problem.prox(theta - eta * gradient, eta, lam)
    return float(np.linalg.norm(theta - prox_point))


def _theta_inexact_solve(
    problem: ReducedGroupLassoProblem,
    lam: Array,
    center: Array,
    theta0: Array,
    gamma: float,
    eta: float,
    target: float,
    max_steps: int,
) -> Tuple[Array, int, float]:
    theta = np.asarray(theta0, dtype=float).copy()
    residual = _theta_pg_residual(problem, theta, lam, center, gamma, eta)
    for inner_iteration in range(1, max_steps + 1):
        gradient = problem.train_grad(theta) + (theta - center) / gamma
        theta = problem.prox(theta - eta * gradient, eta, lam)
        residual = _theta_pg_residual(problem, theta, lam, center, gamma, eta)
        if residual <= target:
            return theta, inner_iteration, residual
    return theta, max_steps, residual


def _moreau_violation(
    problem: ReducedGroupLassoProblem,
    lam: Array,
    lower: Array,
    theta: Array,
    gamma: float,
    epsilon: float = 0.0,
) -> float:
    phi_lower = problem.train_loss(lower) + float(np.dot(lam, problem.group_norms(lower)))
    phi_theta = problem.train_loss(theta) + float(np.dot(lam, problem.group_norms(theta)))
    value = phi_lower - phi_theta - 0.5 * float(np.dot(theta - lower, theta - lower)) / gamma
    return float(max(value - epsilon, 0.0))


def run_meha(problem: ReducedGroupLassoProblem, setting=None):
    """MEHA copied from the authors' ``MEHAHO/R/MEHA_SGL.R`` update.

    The sparse L1 block is removed for pure Group Lasso.  The common initial
    coefficient/hyperparameter point and the requested stopping residual are
    the only experiment-level overrides.
    """

    setting = dict(setting or {})
    max_iter = int(setting.get("max_iter", 100000))
    tol = float(setting.get("tol", 1e-5))
    record_interval = max(1, int(setting.get("record_interval", 25)))
    gamma = float(setting.get("moreau_gamma", 1.0))
    floor = float(setting.get("lambda_floor", 1e-8))
    ceiling = float(setting.get("lambda_ceiling", 10.0))
    c0 = float(setting.get("meha_c0", 1.0))
    c_power = float(setting.get("meha_c_power", 0.49))
    alpha = float(setting.get("meha_alpha", 1e-4))
    beta = float(setting.get("meha_beta", 1e-3))
    eta = float(setting.get("meha_eta", 1e-3))

    initial_lambda = np.asarray(setting.get("initial_lambda", 1.0), dtype=float)
    if initial_lambda.ndim == 0:
        lam = np.full(problem.group_count, float(initial_lambda))
    else:
        lam = initial_lambda.reshape(-1).copy()
        if lam.size != problem.group_count:
            raise ValueError(
                "MEHA initial_lambda needs %d values, got %d"
                % (problem.group_count, lam.size)
            )
    lower = np.asarray(setting.get("initial_coef", np.ones(problem.p)), dtype=float).copy()
    theta = lower.copy()
    records = [
        _initial_record(
            problem,
            lower,
            lam,
            moreau_residual=np.nan,
            theta_pg_residual=np.nan,
            penalty_parameter=c0,
            theta_iterations=0,
        )
    ]
    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        lower_old = lower.copy()
        lam_old = lam.copy()
        theta_old = theta.copy()
        theta_gradient = problem.train_grad(theta) + (theta - lower) / gamma
        theta = problem.prox(theta - eta * theta_gradient, eta, lam)

        upper_direction = problem.group_norms(lower) - problem.group_norms(theta)
        lam = np.clip(lam - alpha * upper_direction, floor, ceiling)
        ck = c0 * iteration**c_power
        lower_direction = (
            problem.val_grad(lower) / ck
            + problem.train_grad(lower)
            - (lower - theta) / gamma
        )
        lower = problem.prox(lower - beta * lower_direction, beta, lam)

        state_norms = np.asarray(
            [np.linalg.norm(lower), np.linalg.norm(lam), np.linalg.norm(theta)],
            dtype=float,
        )
        if not np.all(np.isfinite(state_norms)):
            # Preserve the last representable iterate.  Without this guard a
            # floating-point-frozen divergent state can falsely report zero
            # relative change and therefore "converged".
            lower = lower_old
            lam = lam_old
            theta = theta_old
            records.append(
                problem.record(
                    iteration - 1,
                    lower,
                    lam,
                    np.nan,
                    moreau_residual=np.nan,
                    paper_relative_step=np.nan,
                    paper_stop_metric=np.nan,
                    theta_pg_residual=np.nan,
                    penalty_parameter=c0 * max(iteration - 1, 1) ** c_power,
                    theta_iterations=1,
                    numerical_failure=True,
                )
            )
            status = "nonfinite"
            break

        stop = x_lambda_stop(lower, lam, lower_old, lam_old)
        meha_relative_step = paper_relative_step(lower, lam, lower_old, lam_old)
        theta_residual = _theta_pg_residual(problem, theta, lam, lower, gamma, eta)
        violation = _moreau_violation(problem, lam, lower, theta, gamma)
        requested_stop = (
            meha_relative_step < float(setting.get("paper_relative_tol", 0.005 / problem.p))
            if setting.get("paper29_protocol", False)
            else stop <= tol
        )
        feasibility_tol = setting.get("meha_feasibility_tol")
        if feasibility_tol is not None:
            requested_stop = requested_stop and (
                violation / problem.n_validate <= float(feasibility_tol)
            )
        should_record = requested_stop or iteration % record_interval == 0 or iteration == max_iter
        if should_record:
            records.append(
                problem.record(
                    iteration,
                    lower,
                    lam,
                    stop,
                    moreau_residual=violation,
                    paper_relative_step=meha_relative_step,
                    paper_stop_metric=meha_relative_step,
                    theta_pg_residual=theta_residual,
                    penalty_parameter=ck,
                    theta_iterations=1,
                )
            )
        if not np.isfinite(stop):
            status = "nonfinite"
            break
        if requested_stop:
            status = "converged"
            break
    return _finish(records, lower, lam, status, "MEHA", theta=theta.copy())


def run_agils(problem: ReducedGroupLassoProblem, setting=None):
    """Algorithm 1/2 and Section 6.2 settings of Bai et al. (2026).

    No public author repository was available as of the experiment date.  This
    is a direct transcription of arXiv:2412.18929v5; no parameter search is
    performed for AGILS.
    """

    setting = dict(setting or {})
    max_iter = int(setting.get("max_iter", 100000))
    tol = float(setting.get("tol", 1e-5))
    record_interval = max(1, int(setting.get("record_interval", 25)))
    gamma = float(setting.get("agils_gamma", 1.0 / problem.p))
    floor = float(setting.get("lambda_floor", 1e-8))
    ceiling = float(setting.get("lambda_ceiling", 10.0))
    epsilon = float(setting.get("agils_epsilon", 1e-6))
    penalty = float(setting.get("agils_penalty0", 6.0))
    penalty_increment = float(setting.get("agils_penalty_increment", 0.01))
    cp_value = float(setting.get("agils_cp", 1.0))
    cy_value = float(setting.get("agils_cy", 50.0 * np.sqrt(problem.p)))
    feasibility_tol = float(setting.get("agils_feasibility_tol", 0.1))
    inner_max = int(setting.get("agils_inner_max", 10000))

    lam = np.full(problem.group_count, float(setting.get("initial_lambda", 1.0)))
    lower_bar = np.asarray(setting.get("initial_coef", np.ones(problem.p)), dtype=float).copy()
    theta_bar = lower_bar.copy()
    lower = lower_bar.copy()
    theta = theta_bar.copy()
    # Algorithm 1 uses different reference iterates in (17) and (20).
    # Keep the uncorrected (x^k, y^k, theta^k) sequence separate from the
    # feasibility-corrected (x^k, y_tilde^k, theta_tilde^k) sequence.
    raw_lam = lam.copy()
    raw_lower = lower.copy()
    raw_theta = theta.copy()
    previous_raw_lam = raw_lam.copy()
    previous_raw_lower = raw_lower.copy()
    previous_raw_theta = raw_theta.copy()
    eta = 1.0 / (problem.train_lipschitz + 1.0 / gamma)
    alpha = 1.0 / 1.1
    records = [
        _initial_record(
            problem,
            lower_bar,
            lam,
            moreau_residual=np.nan,
            theta_pg_residual=np.nan,
            theta_half_residual=np.nan,
            theta_iterations=0,
            agils_delta=0.0,
            agils_s_k=np.nan,
            agils_tau_k=np.nan,
            penalty_parameter=penalty,
            feasibility_correction=False,
        )
    ]
    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        lam_old = lam.copy()
        lower_reference = lower_bar.copy()
        beta = 1.0 / (problem.val_lipschitz / penalty + problem.train_lipschitz + 0.1)
        lower_direction = (
            problem.val_grad(lower_bar) / penalty
            + problem.train_grad(lower_bar)
            - (lower_bar - theta_bar) / gamma
        )
        lower = problem.prox(lower_bar - beta * lower_direction, beta, lam)

        s0 = float(setting.get("agils_s0", 5.0))
        s_power = float(setting.get("agils_s_power", 1.05))
        tau0 = float(setting.get("agils_tau0", 10.0))
        tau_power = float(setting.get("agils_tau_power", 0.2))
        s_k = s0 / iteration**s_power
        tau_k = tau0 / iteration**tau_power
        reference_residual = _theta_pg_residual(
            problem,
            previous_raw_theta,
            previous_raw_lam,
            previous_raw_lower,
            gamma,
            eta,
        )
        half_target = max(s_k, tau_k * reference_residual)
        theta_half, inner_half, residual_half = _theta_inexact_solve(
            problem,
            lam_old,
            lower,
            theta_bar,
            gamma,
            eta,
            half_target,
            inner_max,
        )

        upper_direction = problem.group_norms(lower) - problem.group_norms(theta_half)
        lam = np.clip(lam_old - alpha * upper_direction, floor, ceiling)
        s_next = s0 / (iteration + 1) ** s_power
        tau_next = tau0 / (iteration + 1) ** tau_power
        full_reference_residual = _theta_pg_residual(
            problem, raw_theta, raw_lam, raw_lower, gamma, eta
        )
        full_target = max(s_next, tau_next * full_reference_residual)
        theta, inner_full, theta_residual = _theta_inexact_solve(
            problem,
            lam,
            lower,
            theta_half,
            gamma,
            eta,
            full_target,
            inner_max,
        )
        delta = float(
            np.sqrt(np.linalg.norm(lam - lam_old) ** 2 + np.linalg.norm(lower - lower_reference) ** 2)
        )
        violation = _moreau_violation(problem, lam, lower, theta, gamma, epsilon)
        stop = x_lambda_stop(lower, lam, lower_reference, lam_old)
        agils_relative_step = paper_relative_step(lower, lam, lower_reference, lam_old)

        threshold = cp_value * min(1.0 / penalty, violation)
        correction_used = False
        if delta < threshold:
            if np.linalg.norm(lower - theta) <= cy_value * gamma / penalty:
                penalty += penalty_increment
                lower_bar = lower.copy()
                theta_bar = theta.copy()
            else:
                lower_candidate, _ = problem.lower_solve(
                    lam, x0=lower, max_iter=inner_max, tol=float(setting.get("lower_tol", 1e-10))
                )
                candidate_reference = _theta_pg_residual(
                    problem, raw_theta, raw_lam, raw_lower, gamma, eta
                )
                candidate_target = max(s_next, tau_next * candidate_reference)
                theta_candidate, _, _ = _theta_inexact_solve(
                    problem,
                    lam,
                    lower_candidate,
                    theta,
                    gamma,
                    eta,
                    candidate_target,
                    inner_max,
                )
                merit_current = problem.validation_loss(lower) / penalty + _moreau_violation(
                    problem, lam, lower, theta, gamma, 0.0
                )
                merit_candidate = problem.validation_loss(lower_candidate) / penalty + _moreau_violation(
                    problem, lam, lower_candidate, theta_candidate, gamma, 0.0
                )
                if merit_candidate <= merit_current:
                    lower_bar = lower_candidate
                    theta_bar = theta_candidate
                    correction_used = True
                else:
                    lower_bar = lower.copy()
                    theta_bar = theta.copy()
                    penalty += penalty_increment
        else:
            lower_bar = lower.copy()
            theta_bar = theta.copy()

        previous_raw_lam = raw_lam.copy()
        previous_raw_lower = raw_lower.copy()
        previous_raw_theta = raw_theta.copy()
        raw_lam = lam.copy()
        raw_lower = lower.copy()
        raw_theta = theta.copy()

        requested_stop = (
            agils_relative_step < float(setting.get("paper_relative_tol", 0.005 / problem.p))
            and violation < float(setting.get("agils_feasibility_tol", 0.1))
            if setting.get("paper29_protocol", False)
            else stop <= tol
        )
        should_record = requested_stop or iteration % record_interval == 0 or iteration == max_iter
        if should_record:
            records.append(
                problem.record(
                    iteration,
                    lower,
                    lam,
                    stop,
                    moreau_residual=violation,
                    paper_relative_step=agils_relative_step,
                    paper_stop_metric=max(
                        agils_relative_step,
                        violation / max(float(setting.get("agils_feasibility_tol", 0.1)), 1e-15),
                    ),
                    theta_pg_residual=theta_residual,
                    theta_half_residual=residual_half,
                    theta_half_target=half_target,
                    theta_half_reference_residual=reference_residual,
                    theta_full_target=full_target,
                    theta_full_reference_residual=full_reference_residual,
                    theta_iterations=int(inner_half + inner_full),
                    agils_delta=delta,
                    agils_s_k=s_k,
                    agils_tau_k=tau_k,
                    penalty_parameter=penalty,
                    feasibility_correction=bool(correction_used),
                )
            )
        if not np.isfinite(stop):
            status = "nonfinite"
            break
        if requested_stop:
            status = "converged"
            break
    return _finish(records, lower, lam, status, "AGILS", theta=theta.copy())
 
def regularizer_hessian_and_directions(
    problem: ReducedGroupLassoProblem,
    x: Array,
    lam: Array,
    threshold: float = 1e-4,
) -> Tuple[Array, Array, Array]:
    """Build the active-manifold system used by IGJO."""

    active = np.flatnonzero(np.abs(x) > threshold)
    if active.size == 0:
        return active, np.zeros((0, 0)), np.zeros((0, problem.group_count))
    gram, _, _ = problem.train_stats
    hessian = gram[np.ix_(active, active)].copy()
    directions = np.zeros((active.size, problem.group_count), dtype=float)
    active_position = {int(index): position for position, index in enumerate(active)}

    for regularizer_index, regularizer in enumerate(problem.regularizers):
        if regularizer["type"] == "l1":
            directions[:, regularizer_index] = -np.sign(x[active])
            continue
        group_indices = np.arange(
            regularizer["slice"].start, regularizer["slice"].stop, dtype=int
        )
        selected = [
            int(index) for index in group_indices if int(index) in active_position
        ]
        if not selected:
            continue
        positions = np.asarray([active_position[index] for index in selected], dtype=int)
        coefficient = x[np.asarray(selected, dtype=int)]
        norm = float(np.linalg.norm(coefficient))
        if norm <= 0.0:
            continue
        hessian[np.ix_(positions, positions)] += float(lam[regularizer_index]) * (
            np.eye(coefficient.size) / norm
            - np.outer(coefficient, coefficient) / (norm**3)
        )
        directions[positions, regularizer_index] = -coefficient / norm
    return active, hessian, directions


def run_igjo(problem: ReducedGroupLassoProblem, setting=None):
    """Run IGJO on equal or unequal disjoint groups."""

    setting = dict(setting or {})
    max_iter = int(setting.get("igjo_max_iter", 50))
    step_size = float(setting.get("igjo_step_size", 1.0))
    step_size_min = float(setting.get("igjo_step_size_min", 1e-6))
    shrink_factor = float(setting.get("igjo_shrink_factor", 0.1))
    decrease_threshold = float(setting.get("igjo_decrease_threshold", 5e-4))
    armijo = float(setting.get("igjo_backtrack_alpha", 0.001))
    floor = float(setting.get("igjo_lambda_floor", 1e-6))
    lam = np.ones(problem.group_count, dtype=float)
    solve_lower = _make_search_lower_solver(problem, setting)
    x, lower_iterations = solve_lower(lam, None)
    value = problem.validation_loss(x)
    records = [
        problem.record(
            0,
            x,
            lam,
            0.0,
            feasible_lower_iterations=lower_iterations,
            igjo_validation_value=value,
        )
    ]
    status = "max_iter"
    for iteration in range(1, max_iter + 1):
        active, hessian, directions = regularizer_hessian_and_directions(
            problem, x, lam
        )
        if active.size == 0:
            gradient = np.zeros(problem.group_count, dtype=float)
        else:
            derivatives = np.linalg.lstsq(hessian, directions, rcond=None)[0]
            gradient = problem.val_grad(x)[active] @ derivatives
        if not np.all(np.isfinite(gradient)):
            status = "nonfinite"
            break

        candidate_step = step_size
        candidate_lam = np.maximum(lam - candidate_step * gradient, floor)
        candidate_x, candidate_lower_iterations = solve_lower(candidate_lam, x)
        candidate_value = problem.validation_loss(candidate_x)
        raw_threshold = value - armijo * candidate_step * float(
            np.dot(gradient, gradient)
        )
        threshold = value if raw_threshold < 0.0 else raw_threshold
        while candidate_value > threshold and candidate_step > step_size_min:
            candidate_step *= shrink_factor
            candidate_lam = np.maximum(lam - candidate_step * gradient, floor)
            candidate_x, candidate_lower_iterations = solve_lower(
                candidate_lam, candidate_x
            )
            candidate_value = problem.validation_loss(candidate_x)
            raw_threshold = value - armijo * candidate_step * float(
                np.dot(gradient, gradient)
            )
            threshold = value if raw_threshold < 0.0 else raw_threshold

        if value < candidate_value:
            status = "no_descent"
            break
        old_x = x.copy()
        old_lam = lam.copy()
        old_value = value
        x, lower_iterations = solve_lower(candidate_lam, candidate_x)
        lam = candidate_lam
        value = problem.validation_loss(x)
        stop = x_lambda_stop(x, lam, old_x, old_lam)
        records.append(
            problem.record(
                iteration,
                x,
                lam,
                stop,
                feasible_lower_iterations=lower_iterations,
                igjo_validation_value=value,
                igjo_step_size=candidate_step,
            )
        )
        step_size = candidate_step
        if old_value - value < decrease_threshold:
            status = "converged"
            break
        if step_size < step_size_min:
            status = "step_too_small"
            break
    return _finish(records, x, lam, status, "IGJO")



METHOD_RUNNERS = {
    "grid": run_grid_search,
    "random": run_random_search,
    "tpe": run_tpe_search,
    "igjo": run_igjo,
    "vf-idca": run_vfidca,
    "ldmma": run_ldmma,
    "meha": run_meha,
    "agils": run_agils,
}


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

