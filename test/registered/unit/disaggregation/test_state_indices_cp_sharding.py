import unittest

import numpy as np

from sglang.srt.disaggregation.utils import filter_indices_by_position_for_cp_rank
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestStateIndicesCPSharding(unittest.TestCase):
    def test_filter_indices_by_position_for_cp_rank_uses_logical_order(self):
        state_indices = np.array([91, 7, 42, 300, 11], dtype=np.int32)

        rank0_indices, rank0_slice = filter_indices_by_position_for_cp_rank(
            state_indices,
            cp_rank=0,
            cp_size=2,
        )
        rank1_indices, rank1_slice = filter_indices_by_position_for_cp_rank(
            state_indices,
            cp_rank=1,
            cp_size=2,
        )

        np.testing.assert_array_equal(rank0_indices, np.array([91, 7, 42]))
        self.assertEqual(rank0_slice, slice(0, 3))
        np.testing.assert_array_equal(rank1_indices, np.array([300, 11]))
        self.assertEqual(rank1_slice, slice(3, 5))

    def test_filter_indices_by_position_for_cp_rank_preserves_parent_offset(self):
        state_indices = np.array([10, 20, 30, 40], dtype=np.int32)

        filtered, filtered_slice = filter_indices_by_position_for_cp_rank(
            state_indices,
            cp_rank=1,
            cp_size=3,
            index_slice=slice(100, 104),
        )

        np.testing.assert_array_equal(filtered, np.array([30]))
        self.assertEqual(filtered_slice, slice(102, 103))


if __name__ == "__main__":
    unittest.main()
