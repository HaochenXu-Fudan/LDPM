"""Contract checks for the manuscript Group Lasso experiments."""

from __future__ import annotations

import unittest

import numpy as np

import scale
import synthetic


class GroupLassoContractTests(unittest.TestCase):
    def test_synthetic_defaults_match_manuscript(self):
        args = synthetic.build_parser().parse_args([])
        self.assertEqual(args.p, 2400)
        self.assertEqual((args.n_train, args.n_validate, args.n_test), (1600, 1600, 1600))
        self.assertEqual(args.group_count, 40)
        self.assertEqual(args.snr, 2.0)
        self.assertEqual(args.tol, 1e-5)
        self.assertEqual(args.ldpm_stop_metric, "full_z")
        self.assertEqual(
            synthetic.parse_methods(args.methods),
            [
                "grid",
                "random",
                "tpe",
                "igjo",
                "vf-idca",
                "ldmma",
                "meha",
                "agils",
                "ldpm",
                "ldpm-capped",
            ],
        )

    def test_active_groups_follow_manuscript_formula(self):
        data_info, metadata = synthetic.prepare_data(
            p=600,
            seed=2026,
            snr=2.0,
            group_count=10,
            n_train=8,
            n_validate=4,
            n_test=4,
        )
        truth = np.asarray(data_info.data.true_beta)
        group_size = 60
        active = [
            index + 1
            for index in range(10)
            if np.any(truth[index * group_size : (index + 1) * group_size])
        ]
        self.assertEqual(active, [1, 3, 5, 7, 9])
        self.assertEqual(metadata["group_size"], group_size)
        self.assertAlmostEqual(metadata["realized_snr"], 2.0)

    def test_scalability_constants_match_manuscript(self):
        expected_p = (300, 600, 1200, 2400, 3600, 4800, 6000)
        self.assertEqual(scale.FIXED_GROUP_SIZE_30_P_LIST, expected_p)
        self.assertEqual(scale.FIXED_SAMPLE_SIZES, (150, 25, 25))
        old = (
            scale.P_LIST,
            scale.USE_FIXED_M5,
            scale.USE_FIXED_GROUP_SIZE_30,
            scale.USE_FIXED_N,
        )
        try:
            scale.P_LIST = expected_p
            scale.USE_FIXED_M5 = False
            scale.USE_FIXED_GROUP_SIZE_30 = True
            scale.USE_FIXED_N = True
            scale.validate_contract()
            self.assertEqual(scale.sample_sizes(6000), (150, 25, 25))
            self.assertEqual(scale.group_count_for(6000), 200)
            self.assertEqual(scale.group_size_for(6000), 30)
        finally:
            (
                scale.P_LIST,
                scale.USE_FIXED_M5,
                scale.USE_FIXED_GROUP_SIZE_30,
                scale.USE_FIXED_N,
            ) = old


if __name__ == "__main__":
    unittest.main()
