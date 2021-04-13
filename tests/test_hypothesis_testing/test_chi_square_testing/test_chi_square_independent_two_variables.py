import unittest

from data_mocker.distribution_sampling import DistributionSampling
from hypothesis_testing.chi_square_test.chi_square_independent_two_variables import (
    ChiSquareIndependentTwoVariables,
)


class TestChiSquareIndependentTwoVariables(unittest.TestCase):
    def test_chi_square_condition(self):
        array_1 = DistributionSampling(seed=1).generate_bernoulli_samplings(0.7, 1000)
        array_2 = DistributionSampling(seed=2).generate_bernoulli_samplings(0.6, 1000)
        test_result = ChiSquareIndependentTwoVariables(array_1, array_2).test()


if __name__ == "__main__":
    unittest.main()
