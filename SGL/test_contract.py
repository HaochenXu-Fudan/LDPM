"""Contract checks for the manuscript Sparse Group Lasso experiment."""

from __future__ import annotations

import unittest

import numpy as np

import campaign
import experiment


class SparseGroupLassoContractTests(unittest.TestCase):
    def test_experiment_defaults_match_manuscript(self):
        args = experiment.build_parser().parse_args([])
        self.assertEqual(args.p, 600)
        self.assertEqual((args.n_train, args.n_validate, args.n_test), (400, 400, 400))
        self.assertEqual(args.group_count, 10)
        self.assertEqual(args.snr, 2.0)
        self.assertTrue(args.sparse_group)
        self.assertEqual(args.tol, 1e-5)
        self.assertEqual(args.ldpm_stop_metric, "full_z")
        self.assertEqual(args.cs_beta0, 0.03)
        self.assertEqual(args.cs_beta_power, 1.2)
        self.assertEqual(args.cs_gamma, 1.0)
        self.assertEqual(args.beta_max_capped, 775.0)

    def test_campaign_defaults_use_ten_repetitions(self):
        args = campaign.build_parser().parse_args([])
        seeds = [int(value) for value in args.seeds.split(",")]
        self.assertEqual(seeds, list(range(2026, 2036)))
        self.assertEqual(campaign.EXPECTED_LDPM_INITIAL_LAMBDA, [0.01] * 10 + [0.75])
        self.assertIn("ldmma", campaign.METHOD_KEYS)

    def test_regularizer_groups_match_active_group_formula(self):
        data_info, metadata = experiment.prepare_data(
            p=600,
            seed=2026,
            snr=2.0,
            group_count=10,
            sparse_group=True,
            n_train=8,
            n_validate=4,
            n_test=4,
        )
        truth = np.asarray(data_info.data.true_beta)
        active = [
            group + 1
            for group in range(10)
            if np.any(truth[group * 60 : (group + 1) * 60])
        ]
        self.assertEqual(active, [1, 3, 5, 7, 9])
        self.assertEqual(metadata["group_count"], 10)


if __name__ == "__main__":
    unittest.main()
