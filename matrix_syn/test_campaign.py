"""Metric and orchestration tests for the sparse low-rank campaign."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from methods import (
    LDPM,
    MatrixSetting,
    SparseLowRankMatrixProblem,
    compute_lower_level_quality,
    generate_matrix_sensing_data,
)
from campaign import (
    NUMERIC_SUMMARY_FIELDS,
    _aggregate,
    _build_parser,
    _validate_refit_quality,
)


class SparseLowRankCampaignTests(unittest.TestCase):
    @staticmethod
    def _small_data(seed=17):
        return generate_matrix_sensing_data(
            MatrixSetting(
                num_rows=4,
                num_cols=4,
                num_train=12,
                num_val=4,
                num_test=4,
            ),
            seed=seed,
        )

    def test_loss_contract_is_half_mse(self):
        data_info = generate_matrix_sensing_data(
            MatrixSetting(
                num_rows=6,
                num_cols=6,
                num_train=9,
                num_val=4,
                num_test=4,
            ),
            seed=5,
        )
        problem = SparseLowRankMatrixProblem(data_info)
        matrix = np.zeros((6, 6), dtype=float)
        for apply, target, observed in (
            (problem.apply_train, problem.train_loss, problem.b_train),
            (problem.apply_val, problem.validation_error, problem.b_val),
            (problem.apply_test, problem.test_error, problem.b_test),
        ):
            residual = apply(matrix) - observed
            expected = 0.5 * np.mean(residual**2)
            self.assertAlmostEqual(target(matrix), expected, places=14)

    def test_sensing_counts_and_snr(self):
        data_info = generate_matrix_sensing_data(
            MatrixSetting(num_rows=12, num_cols=12, snr=5.0), seed=8
        )
        data = data_info.data
        self.assertEqual((len(data.b_train), len(data.b_val), len(data.b_test)), (36, 14, 14))
        for operator, observations in (
            (data.A_train, data.b_train),
            (data.A_val, data.b_val),
            (data.A_test, data.b_test),
        ):
            clean = operator.reshape(operator.shape[0], -1) @ data.S_true.reshape(-1)
            noise = observations - clean
            self.assertAlmostEqual(
                np.linalg.norm(clean) / np.linalg.norm(noise), 5.0, places=12
            )

    def test_common_quality_uses_normalized_gap(self):
        data_info = generate_matrix_sensing_data(
            MatrixSetting(
                num_rows=4,
                num_cols=4,
                num_train=4,
                num_val=2,
                num_test=2,
            ),
            seed=9,
        )
        matrix = np.zeros((4, 4), dtype=float)
        result = pd.DataFrame([{"iteration": 1, "validation_error": 0.0}])
        result.attrs["method_key"] = "LDPM"
        result.attrs["selection_rule"] = "latest"
        result.attrs["solution_states"] = {
            "latest": {
                "iteration": 1,
                "S": matrix,
                "lambda": np.asarray([0.01, 0.01]),
            }
        }
        quality = compute_lower_level_quality(
            data_info,
            result,
            setting={
                "quality_cvxpy_solver": "SCS",
                "quality_cvxpy_tol": 1e-6,
                "quality_cvxpy_max_iter": 5000,
            },
            include_native_quality=False,
        )
        expected = max(quality["lower_quality_raw_gap"], 0.0) / (
            1.0 + abs(quality["lower_quality_reference_objective"])
        )
        self.assertAlmostEqual(
            quality["lower_quality_normalized_gap"], expected, places=15
        )

    def test_bad_refit_is_not_reported_as_zero_feasibility(self):
        quality = {
            "refit_solver_status": "optimal",
            "lower_quality_raw_gap": -0.1,
            "lower_quality_reference_objective": 1.0,
        }
        with self.assertRaisesRegex(RuntimeError, "exceeds the raw endpoint"):
            _validate_refit_quality(quality, 1e-5)

    def test_aggregate_uses_sample_standard_deviation(self):
        rows = []
        for seed in range(1, 6):
            row = {
                "size": 60,
                "seed": seed,
                "method": "LDPM-CS",
                "status": "max_iter",
                "refit_solver_status": "optimal",
                "converged": False,
                "cap_reached": False,
            }
            row.update({field: float(seed) for field in NUMERIC_SUMMARY_FIELDS})
            rows.append(row)
        aggregate = _aggregate(rows).iloc[0]
        self.assertEqual(int(aggregate["n_requested"]), 5)
        self.assertEqual(int(aggregate["n_valid"]), 5)
        self.assertAlmostEqual(aggregate["wall_time_mean"], 3.0, places=15)
        self.assertAlmostEqual(
            aggregate["wall_time_std"], np.std(np.arange(1.0, 6.0), ddof=1), places=15
        )

        bad = dict(rows[0])
        bad["seed"] = 6
        bad["status"] = "postprocess_error"
        bad["refit_solver_status"] = None
        aggregate = _aggregate(rows + [bad]).iloc[0]
        self.assertEqual(int(aggregate["n_requested"]), 6)
        self.assertEqual(int(aggregate["n_valid"]), 5)

        native_bad = dict(rows[0])
        native_bad["seed"] = 7
        native_bad["refit_solver_status"] = "native_optimal"
        native_bad["initialization_mode"] = "lower_kkt_native_normal_mu"
        native_bad["initialization_solver_status"] = "native_max_iter"
        aggregate = _aggregate(rows + [native_bad]).iloc[0]
        self.assertEqual(int(aggregate["n_requested"]), 6)
        self.assertEqual(int(aggregate["n_valid"]), 5)

    def test_campaign_defaults_match_the_md_contract(self):
        args = _build_parser().parse_args(["--output-dir", "/tmp/not-created"])
        self.assertEqual(args.seeds, "1,2,3,4,5,6,7,8,9,10")
        self.assertEqual(args.methods, "vf-idca,ldpm-cs,ldpm-cs-c")
        self.assertEqual(args.ldpm_beta0, 10.0)
        self.assertEqual(args.ldpm_beta_power, 0.3)
        self.assertEqual(args.ldpm_gamma, 10.0)
        self.assertEqual(args.capped_beta_max, 35.0)
        self.assertEqual(args.ldpm_initialization_mode, "zero_fenchel")
        self.assertEqual(args.ldpm_warm_initial_lambda_l1, 1e-3)
        self.assertEqual(args.ldpm_warm_initial_lambda_nuclear, 1e-3)
        self.assertEqual(args.ldpm_warm_cvxpy_solver, "SCS")
        self.assertEqual(args.ldpm_warm_cvxpy_tol, 1e-6)
        self.assertEqual(args.ldpm_warm_cvxpy_max_iter, 100000)
        self.assertEqual(args.ldpm_warm_cvxpy_time_limit, 600.0)
        self.assertEqual(args.native_tol, 1e-5)
        self.assertEqual(args.native_max_iter, 30000)
        self.assertEqual(args.native_check_interval, 250)
        self.assertEqual(args.native_lipschitz_safety, 1.03)
        self.assertEqual(args.native_tau_factor, 1.2)
        self.assertEqual(args.native_sigma_factor, 0.10)
        self.assertEqual(args.reported_time, "wall_time")
        self.assertEqual(args.tol, 1e-5)

    def test_ldpm_default_zero_initialization_is_unchanged(self):
        data_info = self._small_data(seed=23)
        common = {
            "MAX_ITERATION": 2,
            "MIN_ITERATION": 1,
            "TOL": 0.0,
            "step_size": 2e-2,
            "gamma": 10.0,
            "beta0": 1.0,
            "beta_power": 0.3,
            "beta_max": 5.0,
        }
        implicit = LDPM(data_info, common)
        explicit = LDPM(
            data_info,
            {
                **common,
                "initialization_mode": "zero_fenchel",
                # Warm-only settings must not be inspected on the default path.
                "warm_initial_lambda": [-1.0],
            },
        )
        columns = [name for name in implicit.columns if name != "time"]
        np.testing.assert_allclose(
            implicit[columns].to_numpy(dtype=float),
            explicit[columns].to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
        )
        for state_name in ("latest", "best"):
            for field in ("S", "lambda", "r", "rho1", "rho2", "xi"):
                np.testing.assert_array_equal(
                    implicit.attrs["solution_states"][state_name][field],
                    explicit.attrs["solution_states"][state_name][field],
                )
        self.assertEqual(
            implicit.attrs["initialization"],
            {"initialization_mode": "zero_fenchel"},
        )

    def test_normal_cone_multiplier_pack_and_sign(self):
        problem = SparseLowRankMatrixProblem(self._small_data(seed=29))
        S = np.arange(16.0).reshape(4, 4) / 10.0
        lam = np.array([0.2, 0.3])
        r = np.array([4.0, 2.0])
        rho1 = np.full((4, 4), 0.05)
        rho2 = np.eye(4) * 0.1
        mu_l1, mu_nuclear = problem.normal_cone_consensus_multipliers(
            S, lam, r, rho1, rho2
        )

        n_l1 = problem.unpack_tilde(-mu_l1)
        np.testing.assert_array_equal(n_l1[0], rho1)
        np.testing.assert_array_equal(n_l1[1], [-r[0], 0.0])
        np.testing.assert_array_equal(n_l1[2], S)
        np.testing.assert_array_equal(n_l1[3], np.zeros_like(S))
        np.testing.assert_array_equal(n_l1[4], [-lam[0], 0.0])

        n_nuclear = problem.unpack_tilde(-mu_nuclear)
        np.testing.assert_array_equal(n_nuclear[0], rho2)
        np.testing.assert_array_equal(n_nuclear[1], [0.0, -r[1]])
        np.testing.assert_array_equal(n_nuclear[2], np.zeros_like(S))
        np.testing.assert_array_equal(n_nuclear[3], S)
        np.testing.assert_array_equal(n_nuclear[4], [0.0, -lam[1]])

    def test_small_lower_kkt_state_has_projection_fixed_point(self):
        setting = {
            "warm_initial_lambda": [1e-3, 1e-3],
            "warm_cvxpy_solver": "SCS",
            "warm_cvxpy_tol": 1e-7,
            "warm_cvxpy_max_iter": 100000,
            "warm_cvxpy_time_limit": 60.0,
        }
        problem = SparseLowRankMatrixProblem(self._small_data(seed=31), setting)
        S, lam, r, rho1, rho2, xi, diagnostics = (
            problem.lower_kkt_normal_state()
        )
        z = problem.pack_tilde(S, lam, rho1, rho2, r)
        mu = problem.normal_cone_consensus_multipliers(
            S, lam, r, rho1, rho2
        )
        residual = max(
            np.linalg.norm(projector(z - mu_i / 10.0) - z)
            for projector, mu_i in zip(
                (problem.project_c1, problem.project_c2), mu
            )
        )
        np.testing.assert_allclose(r, problem.regularizer_values(S), atol=0.0)
        np.testing.assert_allclose(
            xi,
            (problem.apply_train(S) - problem.b_train) / problem.m_train,
            rtol=0.0,
            atol=0.0,
        )
        self.assertLess(diagnostics["warm_kkt_stationarity_residual"], 2e-5)
        self.assertLess(residual, 2e-5)

    def test_lower_kkt_state_rejects_invalid_warm_lambda(self):
        data_info = self._small_data(seed=37)
        for invalid in ([-1.0, 0.1], [0.1], [0.1, np.nan]):
            with self.subTest(invalid=invalid):
                problem = SparseLowRankMatrixProblem(
                    data_info, {"warm_initial_lambda": invalid}
                )
                with self.assertRaisesRegex(ValueError, "warm_initial_lambda"):
                    problem.lower_kkt_normal_state()

    def test_native_lower_matches_reference_and_satisfies_kkt(self):
        problem = SparseLowRankMatrixProblem(self._small_data(seed=41))
        lam = np.array([1e-3, 1e-3])
        S, objective, status, rho1, rho2, diagnostics = (
            problem.solve_penalized_lower_native(
                lam,
                extra_setting={
                    "native_tol": 1e-6,
                    "native_max_iter": 30000,
                    "native_check_interval": 20,
                },
            )
        )
        reference_S, reference_objective, reference_status = (
            problem.solve_penalized_lower(
                lam,
                extra_setting={
                    "cvxpy_solver": "SCS",
                    "cvxpy_tol": 1e-8,
                    "cvxpy_max_iter": 100000,
                },
            )
        )
        self.assertEqual(status, "native_optimal")
        self.assertIn(reference_status, {"optimal", "optimal_inaccurate"})
        self.assertLess(abs(objective - reference_objective), 2e-6)
        self.assertLess(
            np.linalg.norm(S - reference_S, "fro")
            / max(1.0, np.linalg.norm(reference_S, "fro")),
            5e-4,
        )
        self.assertLessEqual(diagnostics["native_kkt_stationarity"], 1e-6)
        self.assertLessEqual(diagnostics["native_complementarity_gap"], 1e-6)
        self.assertLessEqual(
            diagnostics["native_primal_relative_change"], 1e-6
        )
        self.assertEqual(diagnostics["native_dual_linf_violation"], 0.0)
        self.assertLess(
            diagnostics["native_dual_spectral_violation"], 1e-12
        )
        self.assertGreater(
            diagnostics["native_step_condition_lhs"],
            diagnostics["native_step_condition_rhs"],
        )
        np.testing.assert_allclose(
            problem.train_grad(S) + rho1 + rho2,
            np.zeros(problem.shape),
            atol=1e-6,
        )

    def test_native_initialization_mode_records_normal_fixed_point(self):
        with mock.patch(
            "methods._solve_cvxpy_problem",
            side_effect=AssertionError("CVXPY initialization is forbidden"),
        ):
            result = LDPM(
                self._small_data(seed=43),
                {
                    "MAX_ITERATION": 1,
                    "MIN_ITERATION": 1,
                    "TOL": 0.0,
                    "initialization_mode": "lower_kkt_native_normal_mu",
                    "warm_initial_lambda": [1e-3, 1e-3],
                    "native_tol": 1e-5,
                    "native_max_iter": 30000,
                    "native_check_interval": 25,
                },
            )
        initialization = result.attrs["initialization"]
        self.assertEqual(
            initialization["initialization_mode"],
            "lower_kkt_native_normal_mu",
        )
        self.assertEqual(initialization["warm_solver"], "NATIVE_CONDAT")
        self.assertEqual(
            initialization["warm_solver_status"], "native_optimal"
        )
        self.assertLess(
            initialization["normal_projection_residual_max"], 5e-5
        )

    def test_native_quality_refit_preserves_metric_schema(self):
        data_info = self._small_data(seed=47)
        problem = SparseLowRankMatrixProblem(data_info)
        S = np.zeros(problem.shape)
        lam = np.array([1e-3, 1e-3])
        result = pd.DataFrame([{"iteration": 1, "validation_error": 0.0}])
        result.attrs["method_key"] = "LDPM"
        result.attrs["selection_rule"] = "latest"
        result.attrs["solution_states"] = {
            "latest": {
                "iteration": 1,
                "S": S,
                "lambda": lam,
                "r": problem.regularizer_values(S),
                "rho1": np.zeros(problem.shape),
                "rho2": np.zeros(problem.shape),
            }
        }
        common = {
            "quality_cvxpy_tol": 1e-6,
            "quality_cvxpy_max_iter": 30000,
        }
        cvx_quality = compute_lower_level_quality(
            data_info,
            result,
            setting={**common, "quality_cvxpy_solver": "SCS"},
            include_native_quality=False,
        )
        with mock.patch.object(
            SparseLowRankMatrixProblem,
            "solve_penalized_lower",
            side_effect=AssertionError("generic solver fallback is forbidden"),
        ):
            native_quality = compute_lower_level_quality(
                data_info,
                result,
                setting={
                    **common,
                    "quality_cvxpy_solver": "NATIVE_CONDAT",
                    "quality_native_tol": 1e-6,
                    "quality_native_max_iter": 30000,
                    "quality_native_check_interval": 20,
                },
                include_native_quality=False,
            )
        self.assertEqual(set(native_quality), set(cvx_quality))
        self.assertEqual(
            native_quality["refit_solver_status"], "native_optimal"
        )
        self.assertLess(
            abs(
                native_quality["lower_quality_reference_objective"]
                - cvx_quality["lower_quality_reference_objective"]
            ),
            2e-6,
        )
        self.assertLess(
            abs(
                native_quality["refit_validation_error"]
                - cvx_quality["refit_validation_error"]
            ),
            2e-4,
        )

    def test_native_solver_rejects_unsafe_steps(self):
        problem = SparseLowRankMatrixProblem(self._small_data(seed=53))
        with self.assertRaisesRegex(ValueError, "Unsafe Condat--Vu steps"):
            problem.solve_penalized_lower_native(
                [1e-3, 1e-3],
                extra_setting={
                    "native_tau_factor": 1.8,
                    "native_sigma_factor": 0.2,
                },
            )


if __name__ == "__main__":
    unittest.main()
