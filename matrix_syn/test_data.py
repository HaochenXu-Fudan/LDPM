"""Structural tests for the synthetic sparse low-rank ground truth."""

from __future__ import annotations

import unittest

import numpy as np

from methods import (
    MatrixSetting,
    generate_matrix_sensing_data,
    generate_sparse_low_rank_matrix,
)


class SparseLowRankGroundTruthTests(unittest.TestCase):
    def _assert_requested_structure(self, n: int) -> None:
        rng = np.random.default_rng(20260817)
        matrix, row_support, col_support = generate_sparse_low_rank_matrix(
            n,
            rng,
            rank=3,
            target_density=0.20,
        )

        self.assertEqual(matrix.shape, (n, n))
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(
            np.isclose(
                np.linalg.norm(matrix, ord="fro") / n,
                1.0,
                rtol=1e-10,
                atol=1e-10,
            )
        )
        self.assertEqual(np.linalg.matrix_rank(matrix), 3)

        mask = np.zeros((n, n), dtype=bool)
        mask[np.ix_(row_support, col_support)] = True
        self.assertTrue(np.all(matrix[~mask] == 0.0))
        self.assertTrue(np.isclose(mask.mean(), 0.2025))

    def test_requested_structure_at_60(self):
        self._assert_requested_structure(60)

    def test_requested_structure_at_100(self):
        self._assert_requested_structure(100)

    def test_same_seed_reproduces_matrix_and_supports(self):
        first = generate_sparse_low_rank_matrix(
            60, np.random.default_rng(73), rank=3, target_density=0.20
        )
        second = generate_sparse_low_rank_matrix(
            60, np.random.default_rng(73), rank=3, target_density=0.20
        )
        for left, right in zip(first, second):
            self.assertTrue(np.array_equal(left, right))

    def test_sensing_data_uses_new_ground_truth_metadata(self):
        setting = MatrixSetting(
            num_rows=12,
            num_cols=12,
            rank=3,
            sparsity=0.20,
            snr=5.0,
            train_fraction=0.25,
            val_fraction=0.10,
            test_fraction=0.10,
        )
        data_info = generate_matrix_sensing_data(setting, seed=11)
        data = data_info.data
        self.assertEqual(data.numerical_rank, 3)
        self.assertEqual(len(data.b_train), int(0.25 * 12 * 12))
        self.assertEqual(len(data.b_val), int(0.10 * 12 * 12))
        self.assertEqual(len(data.b_test), int(0.10 * 12 * 12))
        self.assertAlmostEqual(data.normalized_frobenius, 1.0, places=12)
        self.assertAlmostEqual(
            data.support_density,
            len(data.row_support) * len(data.col_support) / (12 * 12),
            places=15,
        )
        self.assertFalse(hasattr(data, "S_low_rank"))
        self.assertFalse(hasattr(data, "S_sparse"))


if __name__ == "__main__":
    unittest.main()
