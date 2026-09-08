import unittest

import numpy as np

from campaign import SEEDS
from experiment import (
    TaskLossOperator,
    _running_minimum,
    parse_args,
    project_coordinate_epigraph,
    project_nuclear_epigraph,
    project_spectral_epigraph,
)


class SchoolExperimentTests(unittest.TestCase):
    def test_manuscript_defaults(self):
        args = parse_args([])
        self.assertEqual(len(SEEDS), 10)
        self.assertEqual(args.tol, 1e-5)
        self.assertEqual(args.beta0, 1.0)
        self.assertEqual(args.beta_power, 0.2)
        self.assertEqual(args.beta_max, 5.0)
        self.assertEqual(args.gamma, 10.0)
        self.assertEqual(args.consensus_tol, 1e-5)

    def test_coordinate_projection_is_feasible_and_idempotent(self):
        values = np.array([-3.0, 0.5, 2.0, 0.0])
        radii = np.array([1.0, 1.0, -3.0, 0.0])
        projected_values, projected_radii = project_coordinate_epigraph(values, radii)
        self.assertTrue(np.all(np.abs(projected_values) <= projected_radii + 1e-14))
        second_values, second_radii = project_coordinate_epigraph(
            projected_values, projected_radii
        )
        np.testing.assert_allclose(second_values, projected_values)
        np.testing.assert_allclose(second_radii, projected_radii)

    def test_matrix_epigraph_projections_are_feasible(self):
        matrix = np.array([[3.0, -1.0, 0.5], [0.2, 2.0, -4.0]])
        nuclear, nuclear_radius = project_nuclear_epigraph(matrix, 0.25)
        spectral, spectral_radius = project_spectral_epigraph(matrix, 0.25)
        self.assertLessEqual(
            np.sum(np.linalg.svd(nuclear, compute_uv=False)), nuclear_radius + 1e-12
        )
        self.assertLessEqual(
            np.linalg.svd(spectral, compute_uv=False)[0], spectral_radius + 1e-12
        )

    def test_task_operator_adjoint(self):
        rng = np.random.default_rng(7)
        features = [rng.normal(size=(5, 3)), rng.normal(size=(4, 3))]
        responses = [rng.normal(size=5), rng.normal(size=4)]
        operator = TaskLossOperator(features, responses)
        matrix = rng.normal(size=(3, 2))
        dual = rng.normal(size=9)
        left = float(np.dot(operator.apply(matrix), dual))
        right = float(np.sum(matrix * operator.adjoint(dual)))
        self.assertAlmostEqual(left, right, places=12)

    def test_running_minimum_is_nonincreasing(self):
        values = np.array([4.0, 3.0, 3.5, 2.5, 2.7, 2.0])
        envelope = _running_minimum(values)
        np.testing.assert_allclose(envelope, [4.0, 3.0, 3.0, 2.5, 2.5, 2.0])
        self.assertTrue(np.all(np.diff(envelope) <= 0.0))


if __name__ == "__main__":
    unittest.main()
