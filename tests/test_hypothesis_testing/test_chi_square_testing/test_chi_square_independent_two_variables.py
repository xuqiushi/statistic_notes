import unittest

import numpy as np

from data_mocker.distribution_sampling import DistributionSampling
from hypothesis_testing.chi_square_test.chi_square_independent_two_variables import (
    ChiSquareIndependentTwoVariables,
)


class TestChiSquareIndependentTwoVariables(unittest.TestCase):
    def test_chi_square(self):
        array_1 = DistributionSampling(seed=1).generate_bernoulli_samplings(0.5, 1000)
        array_2 = DistributionSampling(seed=2).generate_bernoulli_samplings(0.6, 1000)
        array_3 = DistributionSampling(seed=3).generate_bernoulli_samplings(0.5, 1000)
        test_result = ChiSquareIndependentTwoVariables(array_1, array_2).test()
        self.assertTrue(test_result.rejected)
        test_result = ChiSquareIndependentTwoVariables(array_1, array_3).test()
        self.assertTrue(not test_result.rejected)


if __name__ == "__main__":
    unittest.main()
