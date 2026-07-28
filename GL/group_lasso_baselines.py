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
            "SGL_HYPEROPT_PATH", "/private/tmp/sgl-table4-deps"
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

    lam = np.full(problem.group_count, float(setting.get("initial_lambda", 1.0)))
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


METHOD_RUNNERS = {
    "grid": run_grid_search,
    "random": run_random_search,
    "tpe": run_tpe_search,
    "vf-idca": run_vfidca,
    "ldmma": run_ldmma,
    "meha": run_meha,
    "agils": run_agils,
}
