import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover - exercised only when cvxpy is absent.
    cp = None


@dataclass
class MatrixSetting:
    num_rows: int = 40
    num_cols: int = 40
    rank: int = 3
    sparsity: float = 0.2
    snr: float = 5.0
    train_fraction: float = 0.25
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    num_train: int = 0
    num_val: int = 0
    num_test: int = 0
    print_flag: bool = False

    @property
    def ambient_dim(self):
        return self.num_rows * self.num_cols


@dataclass
class MatrixDataInfo:
    data: object
    settings: MatrixSetting
    data_index: int = 1


class MatrixData:
    def __init__(self):
        self.A_train = None
        self.b_train = None
        self.A_val = None
        self.b_val = None
        self.A_test = None
        self.b_test = None
        self.S_true = None
        self.S_low_rank = None
        self.S_sparse = None
        self.noise_std = None


def _apply_operator(A, S):
    return A.reshape(A.shape[0], -1) @ np.asarray(S, dtype=float).reshape(-1)


def _adjoint_operator(A, y, shape):
    return (A.reshape(A.shape[0], -1).T @ np.asarray(y, dtype=float).reshape(-1)).reshape(shape)


def _make_measurements(rng, S_true, m, snr):
    rows, cols = S_true.shape
    ambient_dim = rows * cols
    A = rng.normal(size=(m, rows, cols)) / np.sqrt(ambient_dim)
    clean = _apply_operator(A, S_true)
    noise = rng.normal(size=m)
    noise_scale = np.linalg.norm(clean) / max(float(snr) * np.linalg.norm(noise), 1e-12)
    b = clean + noise_scale * noise
    return A, b, noise_scale


def generate_matrix_completion_data(setting=None, seed=0):
    """Generate the sparse-plus-low-rank linear observation problem in the md file."""

    setting = setting or MatrixSetting()
    rng = np.random.default_rng(seed)
    rows, cols = setting.num_rows, setting.num_cols
    ambient_dim = rows * cols
    rank = max(1, int(setting.rank))

    U = rng.normal(size=(rows, rank))
    V = rng.normal(size=(cols, rank))
    low_rank = U @ V.T / np.sqrt(rank)

    mask = rng.random(size=(rows, cols)) < float(setting.sparsity)
    sparse = mask * rng.normal(size=(rows, cols))
    if np.linalg.norm(sparse, "fro") > 0:
        sparse *= np.linalg.norm(low_rank, "fro") / max(np.linalg.norm(sparse, "fro"), 1e-12)

    S_true = low_rank + sparse
    S_true /= max(np.linalg.norm(S_true, "fro") / np.sqrt(ambient_dim), 1e-12)

    m_train = setting.num_train or max(1, int(setting.train_fraction * ambient_dim))
    m_val = setting.num_val or max(1, int(setting.val_fraction * ambient_dim))
    m_test = setting.num_test or max(1, int(setting.test_fraction * ambient_dim))

    A_train, b_train, noise_train = _make_measurements(rng, S_true, m_train, setting.snr)
    A_val, b_val, noise_val = _make_measurements(rng, S_true, m_val, setting.snr)
    A_test, b_test, noise_test = _make_measurements(rng, S_true, m_test, setting.snr)

    data = MatrixData()
    data.A_train = A_train
    data.b_train = b_train
    data.A_val = A_val
    data.b_val = b_val
    data.A_test = A_test
    data.b_test = b_test
    data.S_true = S_true
    data.S_low_rank = low_rank
    data.S_sparse = sparse
    data.noise_std = {
        "train": float(noise_train),
        "validation": float(noise_val),
        "test": float(noise_test),
    }

    if setting.print_flag:
        actual_sparsity = float(np.mean(np.abs(sparse) > 0.0))
        print(
            "sparse-low-rank matrix recovery: "
            f"shape={rows}x{cols}, rank={rank}, sparsity={actual_sparsity:.3f}, "
            f"|train|={m_train}, |val|={m_val}, |test|={m_test}"
        )
    return MatrixDataInfo(data=data, settings=setting)


def _require_cvxpy():
    if cp is None:
        raise ImportError("cvxpy is required for VF-iDCA, LDMMA, and reference lower solves.")
    return cp


def _solver_kwargs(solver_name, setting):
    kwargs = {"verbose": bool(setting.get("solver_verbose", setting.get("cvxpy_verbose", False)))}
    tol = setting.get("solver_tol", setting.get("cvxpy_tol", None))
    max_iter = setting.get("solver_max_iter", setting.get("cvxpy_max_iter", None))
    solver_name = str(solver_name).upper()
    if tol is not None:
        tol = float(tol)
        if solver_name == "SCS":
            kwargs["eps"] = tol
        elif solver_name == "CLARABEL":
            kwargs["tol_gap_abs"] = tol
            kwargs["tol_gap_rel"] = tol
            kwargs["tol_feas"] = tol
        elif solver_name == "ECOS":
            kwargs["abstol"] = tol
            kwargs["reltol"] = tol
            kwargs["feastol"] = tol
    if max_iter is not None:
        if solver_name == "SCS":
            kwargs["max_iters"] = int(max_iter)
        else:
            kwargs["max_iter"] = int(max_iter)
    return kwargs


def _solve_cvxpy_problem(problem, setting, context="cvxpy problem"):
    cp_mod = _require_cvxpy()
    requested = setting.get("cvxpy_solver", setting.get("solver", None))
    candidates = [requested] if requested else ["SCS", "CLARABEL"]
    installed = {name.upper(): name for name in cp_mod.installed_solvers()}
    last_error = None
    for candidate in candidates:
        solver_name = str(candidate).upper()
        if solver_name not in installed:
            continue
        try:
            value = problem.solve(solver=installed[solver_name], **_solver_kwargs(solver_name, setting))
            if problem.status in {cp_mod.OPTIMAL, cp_mod.OPTIMAL_INACCURATE}:
                return value
        except Exception as exc:  # Try the next installed solver.
            last_error = exc
    if last_error is not None:
        raise RuntimeError(f"{context} failed in cvxpy: {last_error}") from last_error
    raise RuntimeError(
        f"No suitable cvxpy solver is installed for {context}. "
        f"Requested {candidates}, installed {sorted(installed)}."
    )


def project_l1_epigraph(v, r):
    original_shape = np.shape(v)
    flat = np.asarray(v, dtype=float).reshape(-1)
    r = float(np.asarray(r))
    if np.sum(np.abs(flat)) <= r and r >= 0.0:
        return flat.reshape(original_shape).copy(), r
    abs_v = np.abs(flat)
    high = max(float(np.max(abs_v)) if abs_v.size else 0.0, -r, 1.0)
    while np.sum(np.maximum(abs_v - high, 0.0)) - (r + high) > 0.0:
        high *= 2.0
    low = 0.0
    for _ in range(70):
        mid = 0.5 * (low + high)
        if np.sum(np.maximum(abs_v - mid, 0.0)) - (r + mid) > 0.0:
            low = mid
        else:
            high = mid
    gamma = high
    projected = np.sign(flat) * np.maximum(abs_v - gamma, 0.0)
    return projected.reshape(original_shape), max(r + gamma, 0.0)


def project_linf_epigraph(v, r):
    original_shape = np.shape(v)
    flat = np.asarray(v, dtype=float).reshape(-1)
    r = float(np.asarray(r))
    abs_v = np.abs(flat)
    max_v = float(np.max(abs_v)) if abs_v.size else 0.0
    if max_v <= r and r >= 0.0:
        return flat.reshape(original_shape).copy(), r

    def derivative(tau):
        active = abs_v > tau
        return tau - r + np.sum(tau - abs_v[active])

    if derivative(0.0) >= 0.0:
        tau = 0.0
    else:
        low = 0.0
        high = max(max_v, r, 1.0)
        while derivative(high) < 0.0:
            high *= 2.0
        for _ in range(70):
            mid = 0.5 * (low + high)
            if derivative(mid) < 0.0:
                low = mid
            else:
                high = mid
        tau = high
    return np.clip(flat, -tau, tau).reshape(original_shape), max(tau, 0.0)


def project_nuclear_epigraph(matrix, r):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    s_proj, r_proj = project_l1_epigraph(s, r)
    return (u * s_proj.reshape(1, -1)) @ vt, r_proj


def project_spectral_epigraph(matrix, r):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    s_proj, r_proj = project_linf_epigraph(s, r)
    return (u * s_proj.reshape(1, -1)) @ vt, r_proj


class SparseLowRankMatrixProblem:
    def __init__(self, data_info, setting=None):
        self.data_info = data_info
        self.data = data_info.data
        self.settings = data_info.settings
        self.setting = dict(setting or {})
        self.shape = (self.settings.num_rows, self.settings.num_cols)
        self.size = self.settings.ambient_dim
        self.A_train = self.data.A_train.reshape(len(self.data.b_train), -1)
        self.A_val = self.data.A_val.reshape(len(self.data.b_val), -1)
        self.A_test = self.data.A_test.reshape(len(self.data.b_test), -1)
        self.b_train = np.asarray(self.data.b_train, dtype=float).reshape(-1)
        self.b_val = np.asarray(self.data.b_val, dtype=float).reshape(-1)
        self.b_test = np.asarray(self.data.b_test, dtype=float).reshape(-1)
        self.m_train = len(self.b_train)
        self.m_val = len(self.b_val)
        self.m_test = len(self.b_test)

    def beta(self, k):
        beta0 = float(self.setting.get("beta0", self.setting.get("beta", 1.0)))
        power = float(self.setting.get("beta_power", 0.3))
        beta_max = float(self.setting.get("beta_max", 1e6))
        return min(beta0 * (1.0 + k) ** power, beta_max)

    def _vec(self, S):
        return np.asarray(S, dtype=float).reshape(-1)

    def apply_train(self, S):
        return self.A_train @ self._vec(S)

    def apply_val(self, S):
        return self.A_val @ self._vec(S)

    def apply_test(self, S):
        return self.A_test @ self._vec(S)

    def adjoint_train(self, y):
        return (self.A_train.T @ np.asarray(y, dtype=float).reshape(-1)).reshape(self.shape)

    def train_loss(self, S):
        residual = self.apply_train(S) - self.b_train
        return 0.5 * float(np.dot(residual, residual)) / max(1, self.m_train)

    def validation_error(self, S):
        residual = self.apply_val(S) - self.b_val
        return 0.5 * float(np.dot(residual, residual)) / max(1, self.m_val)

    def test_error(self, S):
        residual = self.apply_test(S) - self.b_test
        return 0.5 * float(np.dot(residual, residual)) / max(1, self.m_test)

    def train_grad(self, S):
        residual = self.apply_train(S) - self.b_train
        return (self.A_train.T @ residual).reshape(self.shape) / max(1, self.m_train)

    def val_grad(self, S):
        residual = self.apply_val(S) - self.b_val
        return (self.A_val.T @ residual).reshape(self.shape) / max(1, self.m_val)

    def apply_train_matrix(self, H):
        return self.A_train @ self._vec(H)

    def regularizer_values(self, S):
        singular = np.linalg.svd(S, compute_uv=False)
        return np.array([np.sum(np.abs(S)), np.sum(singular)], dtype=float)

    def lower_objective(self, S, lam):
        lam = np.maximum(np.asarray(lam, dtype=float).reshape(2), 0.0)
        regs = self.regularizer_values(S)
        return self.train_loss(S) + float(np.dot(lam, regs))

    def _cp_vec(self, S_expr):
        return cp.reshape(S_expr, (self.size,), order="C")

    def _cp_train_loss(self, S_expr):
        pred = self.A_train @ self._cp_vec(S_expr)
        return 0.5 * cp.sum_squares(pred - self.b_train) / max(1, self.m_train)

    def _cp_val_loss(self, S_expr):
        pred = self.A_val @ self._cp_vec(S_expr)
        return 0.5 * cp.sum_squares(pred - self.b_val) / max(1, self.m_val)

    def solve_penalized_lower(self, lam, x0=None, extra_setting=None):
        _require_cvxpy()
        setting = dict(self.setting)
        if extra_setting:
            setting.update(extra_setting)
        lam = np.maximum(np.asarray(lam, dtype=float).reshape(2), 0.0)
        S_var = cp.Variable(self.shape)
        objective = self._cp_train_loss(S_var) + lam[0] * cp.norm1(S_var) + lam[1] * cp.normNuc(S_var)
        problem = cp.Problem(cp.Minimize(objective))
        if x0 is not None:
            S_var.value = np.asarray(x0, dtype=float).reshape(self.shape)
        _solve_cvxpy_problem(problem, setting, "penalized lower-level matrix problem")
        if S_var.value is None:
            raise RuntimeError(f"Penalized lower-level solve failed with status {problem.status}.")
        S = np.asarray(S_var.value, dtype=float).reshape(self.shape)
        return S, float(problem.value), problem.status

    def solve_constrained_lower(self, r, x0=None, extra_setting=None):
        _require_cvxpy()
        setting = dict(self.setting)
        if extra_setting:
            setting.update(extra_setting)
        r = np.maximum(np.asarray(r, dtype=float).reshape(2), 0.0)
        S_var = cp.Variable(self.shape)
        constraints = [cp.norm1(S_var) <= r[0], cp.normNuc(S_var) <= r[1]]
        problem = cp.Problem(cp.Minimize(self._cp_train_loss(S_var)), constraints)
        if x0 is not None:
            S_var.value = np.asarray(x0, dtype=float).reshape(self.shape)
        _solve_cvxpy_problem(problem, setting, "constrained lower-level matrix problem")
        if S_var.value is None:
            raise RuntimeError(f"Constrained lower-level solve failed with status {problem.status}.")
        duals = np.array(
            [0.0 if constraint.dual_value is None else float(constraint.dual_value) for constraint in constraints],
            dtype=float,
        )
        S = np.asarray(S_var.value, dtype=float).reshape(self.shape)
        return S, float(problem.value), np.maximum(duals, 0.0), problem.status

    def zero_fenchel_state(self):
        S = np.zeros(self.shape)
        r = np.zeros(2)
        xi = -self.b_train / max(1, self.m_train)
        rho1 = np.zeros(self.shape)
        rho2 = -self.adjoint_train(xi)
        lam = np.array([0.0, np.linalg.norm(rho2, 2)], dtype=float)
        return S, np.maximum(lam, 0.0), r, rho1, rho2, xi

    def merit_components(self, S, lam, r, rho1, rho2, xi):
        lam = np.asarray(lam, dtype=float).reshape(2)
        r = np.asarray(r, dtype=float).reshape(2)
        xi = np.asarray(xi, dtype=float).reshape(-1)
        h = self.adjoint_train(xi) + rho1 + rho2
        p = (
            self.train_loss(S)
            + float(np.dot(lam, r))
            + 0.5 * self.m_train * float(np.dot(xi, xi))
            + float(np.dot(xi, self.b_train))
        )
        fy_gap = p - float(np.sum(S * h))
        h_norm = float(np.linalg.norm(h, "fro"))
        psi = fy_gap + 0.5 * h_norm**2
        return {
            "p_value": float(p),
            "fenchel_gap": float(fy_gap),
            "h_norm": h_norm,
            "psi": float(psi),
            "h": h,
        }

    def constraint_violations(self, S, lam, r, rho1, rho2):
        lam = np.asarray(lam, dtype=float).reshape(2)
        r = np.asarray(r, dtype=float).reshape(2)
        regs = self.regularizer_values(S)
        return {
            "l1_epigraph_violation": max(0.0, float(regs[0] - r[0])),
            "nuclear_epigraph_violation": max(0.0, float(regs[1] - r[1])),
            "lambda_negative_violation": max(0.0, float(np.max(-lam))),
            "r_negative_violation": max(0.0, float(np.max(-r))),
            "linf_dual_violation": max(0.0, float(np.max(np.abs(rho1)) - lam[0])),
            "spectral_dual_violation": max(0.0, float(np.linalg.norm(rho2, 2) - lam[1])),
        }

    def basic_record(self, elapsed, iteration, S, lam=None, r=None, extra=None):
        lam = np.zeros(2) if lam is None else np.asarray(lam, dtype=float).reshape(2)
        regs = self.regularizer_values(S)
        singular = np.linalg.svd(S, compute_uv=False)
        record = {
            "iteration": int(iteration),
            "time": float(elapsed),
            "train_error": self.train_loss(S),
            "validation_error": self.validation_error(S),
            "test_error": self.test_error(S),
            "lambda_l1": float(lam[0]),
            "lambda_nuclear": float(lam[1]),
            "l1_norm": float(regs[0]),
            "nuclear_norm": float(regs[1]),
            "rank": int(np.sum(singular > float(self.setting.get("svd_rank_tol", 1e-6)))),
            "nnz": int(np.sum(np.abs(S) > float(self.setting.get("sparsity_tol", 1e-8)))),
        }
        if r is not None:
            r = np.asarray(r, dtype=float).reshape(2)
            record["r_l1"] = float(r[0])
            record["r_nuclear"] = float(r[1])
        if extra:
            record.update(extra)
        return record

    def pack_tilde(self, S, lam, rho1, rho2, r):
        return np.concatenate(
            [
                np.asarray(S, dtype=float).reshape(-1),
                np.asarray(lam, dtype=float).reshape(2),
                np.asarray(rho1, dtype=float).reshape(-1),
                np.asarray(rho2, dtype=float).reshape(-1),
                np.asarray(r, dtype=float).reshape(2),
            ]
        )

    def unpack_tilde(self, vec):
        vec = np.asarray(vec, dtype=float).reshape(-1)
        pos = 0
        S = vec[pos : pos + self.size].reshape(self.shape).copy()
        pos += self.size
        lam = vec[pos : pos + 2].copy()
        pos += 2
        rho1 = vec[pos : pos + self.size].reshape(self.shape).copy()
        pos += self.size
        rho2 = vec[pos : pos + self.size].reshape(self.shape).copy()
        pos += self.size
        r = vec[pos : pos + 2].copy()
        return S, lam, rho1, rho2, r

    def project_c1(self, vec):
        S, lam, rho1, rho2, r = self.unpack_tilde(vec)
        S, r[0] = project_l1_epigraph(S, r[0])
        r[1] = max(float(r[1]), 0.0)
        rho1, lam[0] = project_linf_epigraph(rho1, lam[0])
        rho2, lam[1] = project_spectral_epigraph(rho2, lam[1])
        return self.pack_tilde(S, np.maximum(lam, 0.0), rho1, rho2, np.maximum(r, 0.0))

    def project_c2(self, vec):
        S, lam, rho1, rho2, r = self.unpack_tilde(vec)
        S, r[1] = project_nuclear_epigraph(S, r[1])
        r[0] = max(float(r[0]), 0.0)
        rho1, lam[0] = project_linf_epigraph(rho1, lam[0])
        rho2, lam[1] = project_spectral_epigraph(rho2, lam[1])
        return self.pack_tilde(S, np.maximum(lam, 0.0), rho1, rho2, np.maximum(r, 0.0))


def _state(iteration, S, lam, r=None, rho1=None, rho2=None, xi=None):
    item = {
        "iteration": int(iteration),
        "S": np.asarray(S, dtype=float).copy(),
        "lambda": np.asarray(lam, dtype=float).reshape(2).copy(),
    }
    if r is not None:
        item["r"] = np.asarray(r, dtype=float).reshape(2).copy()
    if rho1 is not None:
        item["rho1"] = np.asarray(rho1, dtype=float).copy()
    if rho2 is not None:
        item["rho2"] = np.asarray(rho2, dtype=float).copy()
    if xi is not None:
        item["xi"] = np.asarray(xi, dtype=float).reshape(-1).copy()
    return item


def _finalize_result(df, method_key, setting, states, selection_rule="latest"):
    df = pd.DataFrame(df).copy()
    df.attrs["method_key"] = method_key
    df.attrs["setting"] = dict(setting or {})
    df.attrs["selection_rule"] = selection_rule
    if states:
        best_pos = int(df["validation_error"].to_numpy().argmin()) if not df.empty else len(states) - 1
        df.attrs["solution_states"] = {
            "latest": states[-1],
            "best": states[min(best_pos, len(states) - 1)],
        }
    return df


def VF_iDCA(data_info, setting=None):
    """VF-iDCA in the radius form described in sparse_low_rank_bilevel_methods.md."""

    setting = dict(setting or {})
    problem = SparseLowRankMatrixProblem(data_info, setting)
    max_iter = int(setting.get("MAX_ITERATION", 8))
    min_iter = int(setting.get("MIN_ITERATION", 1))
    tol = float(setting.get("TOL", 5e-3))
    rho = float(setting.get("rho", 1e-2))
    alpha = float(setting.get("alpha0", setting.get("penalty", 10.0)))
    alpha_growth = float(setting.get("alpha_growth", 2.0))
    alpha_max = float(setting.get("alpha_max", 1e6))
    update_c = float(setting.get("alpha_update_c", 0.25))
    initial_lambda = np.asarray(setting.get("initial_lambda", [0.001, 0.001]), dtype=float).reshape(2)

    S, _, _ = problem.solve_penalized_lower(initial_lambda)
    r = np.maximum(problem.regularizer_values(S), 1e-10)
    records = []
    states = []
    start = time.time()
    final_gamma = initial_lambda.copy()

    for k in range(max_iter):
        lower_S, lower_value, gamma, _ = problem.solve_constrained_lower(r, x0=S)
        final_gamma = gamma.copy()

        S_var = cp.Variable(problem.shape)
        r_var = cp.Variable(2, nonneg=True)
        t_var = cp.Variable(nonneg=True)
        train_loss = problem._cp_train_loss(S_var)
        val_loss = problem._cp_val_loss(S_var)
        value_upper = train_loss - lower_value + gamma @ (r_var - r)
        constraints = [
            t_var >= value_upper,
            t_var >= cp.norm1(S_var) - r_var[0],
            t_var >= cp.normNuc(S_var) - r_var[1],
        ]
        prox = cp.sum_squares(S_var - S) + cp.sum_squares(r_var - r)
        subproblem = cp.Problem(cp.Minimize(val_loss + 0.5 * rho * prox + alpha * t_var), constraints)
        S_var.value = S
        r_var.value = r
        _solve_cvxpy_problem(subproblem, setting, "VF-iDCA joint radius subproblem")
        if S_var.value is None or r_var.value is None:
            raise RuntimeError(f"VF-iDCA subproblem failed with status {subproblem.status}.")

        S_next = np.asarray(S_var.value, dtype=float).reshape(problem.shape)
        r_next = np.maximum(np.asarray(r_var.value, dtype=float).reshape(2), 0.0)
        regs = problem.regularizer_values(S_next)
        value_violation = problem.train_loss(S_next) - lower_value + float(np.dot(gamma, r_next - r))
        t_value = max(0.0, value_violation, float(regs[0] - r_next[0]), float(regs[1] - r_next[1]))
        step = float(
            np.sqrt(np.linalg.norm(S_next - S, "fro") ** 2 + np.linalg.norm(r_next - r) ** 2)
            / max(1.0, np.sqrt(np.linalg.norm(S_next, "fro") ** 2 + np.linalg.norm(r_next) ** 2))
        )

        elapsed = time.time() - start
        extra = {
            "step_err": step,
            "vfidca_t": float(t_value),
            "vfidca_value_upper": float(value_violation),
            "alpha": float(alpha),
            "lower_value_at_r": float(lower_value),
            "lower_solution_validation_error": problem.validation_error(lower_S),
            "lower_solution_test_error": problem.test_error(lower_S),
        }
        records.append(problem.basic_record(elapsed, k + 1, S_next, gamma, r_next, extra))
        states.append(_state(k + 1, S_next, gamma, r_next))

        if k + 1 >= min_iter and max(step, t_value) <= tol:
            S, r = S_next, r_next
            break
        if step <= update_c * min(1.0, max(t_value, 1e-12)):
            alpha = min(alpha * alpha_growth, alpha_max)
        S, r = S_next, r_next

    _, _, final_gamma, _ = problem.solve_constrained_lower(r, x0=S)
    if records:
        records[-1]["lambda_l1"] = float(final_gamma[0])
        records[-1]["lambda_nuclear"] = float(final_gamma[1])
        states[-1]["lambda"] = final_gamma.copy()
    return _finalize_result(records, "VFIDCA", setting, states, "latest")


def LDMMA(data_info, setting=None):
    """Fenchel-duality MM method with the bilinear majorizer from the md file."""

    setting = dict(setting or {})
    problem = SparseLowRankMatrixProblem(data_info, setting)
    max_iter = int(setting.get("MAX_ITERATION", 6))
    min_iter = int(setting.get("MIN_ITERATION", 1))
    tol = float(setting.get("TOL", 5e-3))
    epsilon = float(setting.get("epsilon", 1e-2))
    beta = float(setting.get("prox_beta", 1e-3))
    lambda_max = setting.get("lambda_max", None)

    S, lam, r, rho1, rho2, xi = problem.zero_fenchel_state()
    records = []
    states = []
    start = time.time()

    for k in range(max_iter):
        S_var = cp.Variable(problem.shape)
        lam_var = cp.Variable(2, nonneg=True)
        r_var = cp.Variable(2, nonneg=True)
        rho1_var = cp.Variable(problem.shape)
        rho2_var = cp.Variable(problem.shape)
        xi_var = cp.Variable(problem.m_train)

        def majorizer(index):
            lam0 = float(lam[index])
            r0 = float(r[index])
            return (
                0.25 * cp.square(lam_var[index] + r_var[index])
                + 0.25 * (lam0 - r0) ** 2
                - 0.5 * (lam0 - r0) * (lam_var[index] - r_var[index])
            )

        train_loss = problem._cp_train_loss(S_var)
        val_loss = problem._cp_val_loss(S_var)
        phi_star = 0.5 * problem.m_train * cp.sum_squares(xi_var)
        stationarity = problem.A_train.T @ cp.reshape(xi_var, (problem.m_train,), order="C")
        stationarity = cp.reshape(stationarity, problem.shape, order="C") + rho1_var + rho2_var
        fenchel_budget = train_loss + phi_star + xi_var @ problem.b_train + majorizer(0) + majorizer(1)

        constraints = [
            fenchel_budget <= epsilon,
            stationarity == 0,
            cp.norm1(S_var) <= r_var[0],
            cp.normNuc(S_var) <= r_var[1],
            cp.norm_inf(rho1_var) <= lam_var[0],
            cp.norm(rho2_var, 2) <= lam_var[1],
        ]
        if lambda_max is not None and np.isfinite(float(lambda_max)):
            constraints.append(lam_var <= float(lambda_max))

        prox = (
            cp.sum_squares(S_var - S)
            + cp.sum_squares(lam_var - lam)
            + cp.sum_squares(r_var - r)
            + cp.sum_squares(rho1_var - rho1)
            + cp.sum_squares(rho2_var - rho2)
            + cp.sum_squares(xi_var - xi)
        )
        subproblem = cp.Problem(cp.Minimize(val_loss + 0.5 * beta * prox), constraints)
        S_var.value = S
        lam_var.value = np.maximum(lam, 0.0)
        r_var.value = np.maximum(r, 0.0)
        rho1_var.value = rho1
        rho2_var.value = rho2
        xi_var.value = xi
        _solve_cvxpy_problem(subproblem, setting, "LDMMA Fenchel-MM subproblem")
        if any(var.value is None for var in [S_var, lam_var, r_var, rho1_var, rho2_var, xi_var]):
            raise RuntimeError(f"LDMMA subproblem failed with status {subproblem.status}.")

        S_next = np.asarray(S_var.value, dtype=float).reshape(problem.shape)
        lam_next = np.maximum(np.asarray(lam_var.value, dtype=float).reshape(2), 0.0)
        r_next = np.maximum(np.asarray(r_var.value, dtype=float).reshape(2), 0.0)
        rho1_next = np.asarray(rho1_var.value, dtype=float).reshape(problem.shape)
        rho2_next = np.asarray(rho2_var.value, dtype=float).reshape(problem.shape)
        xi_next = np.asarray(xi_var.value, dtype=float).reshape(-1)

        diff_sq = (
            np.linalg.norm(S_next - S, "fro") ** 2
            + np.linalg.norm(lam_next - lam) ** 2
            + np.linalg.norm(r_next - r) ** 2
            + np.linalg.norm(rho1_next - rho1, "fro") ** 2
            + np.linalg.norm(rho2_next - rho2, "fro") ** 2
            + np.linalg.norm(xi_next - xi) ** 2
        )
        base_sq = (
            np.linalg.norm(S_next, "fro") ** 2
            + np.linalg.norm(lam_next) ** 2
            + np.linalg.norm(r_next) ** 2
            + np.linalg.norm(rho1_next, "fro") ** 2
            + np.linalg.norm(rho2_next, "fro") ** 2
            + np.linalg.norm(xi_next) ** 2
        )
        step = float(np.sqrt(diff_sq / max(1.0, base_sq)))
        components = problem.merit_components(S_next, lam_next, r_next, rho1_next, rho2_next, xi_next)
        violations = problem.constraint_violations(S_next, lam_next, r_next, rho1_next, rho2_next)
        c_dual = max(0.0, float(components["p_value"]))
        budget_violation = max(0.0, c_dual - epsilon)
        all_violation = max([budget_violation, components["h_norm"], *violations.values()])

        elapsed = time.time() - start
        extra = {
            "step_err": step,
            "ll_duality_gap": c_dual,
            "ldmma_fenchel_gap": max(0.0, float(components["fenchel_gap"])),
            "ldmma_h_norm": float(components["h_norm"]),
            "budget_violation": float(budget_violation),
            "all_feasibility": float(all_violation),
            "epsilon": epsilon,
            "cvxpy_status": subproblem.status,
        }
        extra.update(violations)
        records.append(problem.basic_record(elapsed, k + 1, S_next, lam_next, r_next, extra))
        states.append(_state(k + 1, S_next, lam_next, r_next, rho1_next, rho2_next, xi_next))

        S, lam, r, rho1, rho2, xi = S_next, lam_next, r_next, rho1_next, rho2_next, xi_next
        if k + 1 >= min_iter and max(step, budget_violation) <= tol:
            break

    return _finalize_result(records, "LDMMA", setting, states, "latest")


def LDPM(data_info, setting=None):
    """LDPM-CS for the shared sparse/nuclear matrix variable."""

    setting = dict(setting or {})
    problem = SparseLowRankMatrixProblem(data_info, setting)
    max_iter = int(setting.get("MAX_ITERATION", 2000))
    min_iter = int(setting.get("MIN_ITERATION", 100))
    tol = float(setting.get("TOL", 1e-5))
    step = float(setting.get("step_size", 2e-2))
    gamma_penalty = float(setting.get("gamma", 10.0))

    S, lam, r, rho1, rho2, xi = problem.zero_fenchel_state()
    z = problem.pack_tilde(S, lam, rho1, rho2, r)
    projectors = [problem.project_c1, problem.project_c2]
    u = [projector(z) for projector in projectors]
    mu = [np.zeros_like(z), np.zeros_like(z)]
    records = []
    states = []
    start = time.time()

    for k in range(max_iter):
        old_z = z.copy()
        old_xi = xi.copy()
        S, lam, rho1, rho2, r = problem.unpack_tilde(z)
        beta_k = problem.beta(k)
        components = problem.merit_components(S, lam, r, rho1, rho2, xi)
        h = components["h"]

        grad_S = problem.val_grad(S) / max(beta_k, 1e-12) + problem.train_grad(S) - h
        grad_lam = r.copy()
        grad_r = lam.copy()
        grad_rho1 = -S + h
        grad_rho2 = -S + h
        grad_xi = problem.b_train - problem.apply_train(S) + problem.apply_train_matrix(h)
        grad_tilde = problem.pack_tilde(grad_S, grad_lam, grad_rho1, grad_rho2, grad_r)

        consensus_grad = np.zeros_like(z)
        for i in range(2):
            consensus_grad += -mu[i] + gamma_penalty * (z - u[i])
        z = z - step * (grad_tilde + consensus_grad)
        S, lam, rho1, rho2, r = problem.unpack_tilde(z)
        lam = np.maximum(lam, 0.0)
        r = np.maximum(r, 0.0)
        z = problem.pack_tilde(S, lam, rho1, rho2, r)
        xi = (xi - step * grad_xi) / (1.0 + step * problem.m_train)

        for i, projector in enumerate(projectors):
            u[i] = projector(z - mu[i] / gamma_penalty)
            mu[i] = mu[i] + gamma_penalty * (u[i] - z)

        S, lam, rho1, rho2, r = problem.unpack_tilde(z)
        components = problem.merit_components(S, lam, r, rho1, rho2, xi)
        violations = problem.constraint_violations(S, lam, r, rho1, rho2)
        consensus = max(float(np.linalg.norm(u_i - z)) for u_i in u)
        step_err = float(
            np.sqrt(np.linalg.norm(z - old_z) ** 2 + np.linalg.norm(xi - old_xi) ** 2)
            / max(1.0, np.sqrt(np.linalg.norm(z) ** 2 + np.linalg.norm(xi) ** 2))
        )

        elapsed = time.time() - start
        extra = {
            "step_err": step_err,
            "consensus_residual": consensus,
            "ll_duality_gap": max(0.0, float(components["psi"])),
            "ldpm_psi": max(0.0, float(components["psi"])),
            "ldpm_psi_raw": float(components["psi"]),
            "ldpm_fenchel_gap": max(0.0, float(components["fenchel_gap"])),
            "ldpm_fenchel_gap_raw": float(components["fenchel_gap"]),
            "ll_feasibility": float(components["h_norm"]),
            "beta": float(beta_k),
        }
        extra.update(violations)
        records.append(problem.basic_record(elapsed, k + 1, S, lam, r, extra))
        states.append(_state(k + 1, S, lam, r, rho1, rho2, xi))

        if k + 1 >= min_iter and max(step_err, consensus) <= tol:
            break

    return _finalize_result(records, "LDPM", setting, states, "latest")


def _evaluate_lambdas(data_info, lambdas, setting=None):
    setting = dict(setting or {})
    problem = SparseLowRankMatrixProblem(data_info, setting)
    records = []
    states = []
    start = time.time()
    for i, lam in enumerate(lambdas, 1):
        S, _, _ = problem.solve_penalized_lower(lam)
        lam = np.asarray(lam, dtype=float).reshape(2)
        r = problem.regularizer_values(S)
        records.append(problem.basic_record(time.time() - start, i, S, lam, r))
        states.append(_state(i, S, lam, r))
    return _finalize_result(records, "SEARCH", setting, states, "best")


def Grid_Search(data_info, setting=None):
    setting = dict(setting or {})
    grid_size = int(setting.get("grid_size", 5))
    lo, hi = setting.get("log_bounds", (-3.0, -0.5))
    lambdas = []
    for log_l1 in np.linspace(lo, hi, grid_size):
        for log_nuc in np.linspace(lo, hi, grid_size):
            lambdas.append(np.power(10.0, [log_l1, log_nuc]))
    return _evaluate_lambdas(data_info, lambdas, setting)


def Random_Search(data_info, setting=None):
    setting = dict(setting or {})
    n_eval = int(setting.get("n_eval", 20))
    lo, hi = setting.get("log_bounds", (-3.0, -0.5))
    rng = np.random.default_rng(setting.get("seed", 0))
    logs = rng.uniform(lo, hi, size=(n_eval, 2))
    return _evaluate_lambdas(data_info, np.power(10.0, logs), setting)


def TPE_Search(data_info, setting=None):
    return Random_Search(data_info, setting)


def IGJO(data_info, setting=None):
    return Random_Search(data_info, setting)


def _selected_row(result, selection_rule):
    if result.empty:
        raise ValueError("Cannot compute lower-level quality for an empty result table.")
    if selection_rule == "best":
        return result.loc[result["validation_error"].idxmin()]
    if selection_rule == "latest":
        return result.iloc[-1]
    raise ValueError("selection_rule must be either 'best' or 'latest'.")


def compute_lower_level_quality(data_info, result, setting=None, selection_rule=None):
    setting = dict(setting or result.attrs.get("setting", {}) or {})
    selection_rule = selection_rule or result.attrs.get("selection_rule", "latest")
    method_key = result.attrs.get("method_key", "")
    row = _selected_row(result, selection_rule)
    state = result.attrs.get("solution_states", {}).get(selection_rule)
    if state is None:
        raise ValueError("The result table does not contain the selected matrix solution state.")

    problem = SparseLowRankMatrixProblem(data_info, setting)
    S = np.asarray(state["S"], dtype=float).reshape(problem.shape)
    lam = np.maximum(np.asarray(state["lambda"], dtype=float).reshape(2), 0.0)
    r = np.asarray(state.get("r", problem.regularizer_values(S)), dtype=float).reshape(2)
    current_q = problem.lower_objective(S, lam)
    ref_setting = {
        "cvxpy_solver": setting.get("quality_cvxpy_solver", setting.get("cvxpy_solver", "SCS")),
        "cvxpy_tol": setting.get("quality_cvxpy_tol", min(float(setting.get("cvxpy_tol", 1e-4)), 1e-4)),
        "cvxpy_max_iter": setting.get("quality_cvxpy_max_iter", max(int(setting.get("cvxpy_max_iter", 2000)), 5000)),
        "solver_verbose": setting.get("quality_solver_verbose", False),
    }
    reference_S, reference_q, _ = problem.solve_penalized_lower(lam, x0=S, extra_setting=ref_setting)
    common_gap = max(0.0, float(current_q - reference_q))

    quality = {
        "lower_quality_common_gap": common_gap,
        "lower_quality_current_objective": float(current_q),
        "lower_quality_reference_objective": float(reference_q),
        "lower_quality_lambda_l1": float(lam[0]),
        "lower_quality_lambda_nuclear": float(lam[1]),
        "lower_quality_selected_iteration": int(state["iteration"]),
        "refit_validation_error": problem.validation_error(reference_S),
        "refit_test_error": problem.test_error(reference_S),
    }

    native_name = "value_gap"
    native_value = common_gap
    if method_key == "VFIDCA":
        constrained_S, constrained_value, _, _ = problem.solve_constrained_lower(
            r, x0=S, extra_setting=ref_setting
        )
        regs = problem.regularizer_values(S)
        vf_value_gap = problem.train_loss(S) - constrained_value
        l1_violation = max(0.0, float(regs[0] - r[0]))
        nuc_violation = max(0.0, float(regs[1] - r[1]))
        native_name = "vfidca_lower_feasibility"
        native_value = max(0.0, vf_value_gap, l1_violation, nuc_violation)
        quality.update(
            {
                "vfidca_value_gap": max(0.0, float(vf_value_gap)),
                "vfidca_l1_violation": l1_violation,
                "vfidca_nuclear_violation": nuc_violation,
                "vfidca_constrained_reference_value": float(constrained_value),
                "vfidca_constrained_validation_error": problem.validation_error(constrained_S),
                "vfidca_constrained_test_error": problem.test_error(constrained_S),
            }
        )
    elif method_key == "LDMMA":
        native_name = "ldmma_fenchel_gap"
        native_value = float(row.get("ll_duality_gap", np.nan))
        epsilon = float(row.get("epsilon", setting.get("epsilon", np.nan)))
        quality.update(
            {
                "ldmma_fenchel_gap": native_value,
                "ldmma_epsilon": epsilon,
                "ldmma_epsilon_violation": max(0.0, native_value - epsilon)
                if np.isfinite(native_value) and np.isfinite(epsilon)
                else np.nan,
                "ldmma_all_feasibility": float(row.get("all_feasibility", np.nan)),
            }
        )
    elif method_key == "LDPM":
        native_name = "ldpm_psi"
        native_value = float(row.get("ldpm_psi", row.get("ll_duality_gap", np.nan)))
        quality.update(
            {
                "ldpm_psi": native_value,
                "ldpm_psi_raw": float(row.get("ldpm_psi_raw", np.nan)),
                "ldpm_fenchel_gap": float(row.get("ldpm_fenchel_gap", np.nan)),
                "ldpm_fenchel_gap_raw": float(row.get("ldpm_fenchel_gap_raw", np.nan)),
                "ldpm_h_norm": float(row.get("ll_feasibility", np.nan)),
                "ldpm_consensus_residual": float(row.get("consensus_residual", np.nan)),
            }
        )

    quality["lower_quality_native_name"] = native_name
    quality["lower_quality_native_value"] = float(native_value)
    return quality


def attach_lower_level_quality(result, data_info, setting=None, selection_rule=None):
    result = result.copy()
    selection_rule = selection_rule or result.attrs.get("selection_rule", "latest")
    row = _selected_row(result, selection_rule)
    quality = compute_lower_level_quality(data_info, result, setting=setting, selection_rule=selection_rule)
    for key, value in quality.items():
        if isinstance(value, str):
            continue
        result.loc[row.name, key] = value
    result.attrs["lower_quality"] = quality
    return result
