import unittest

from data_mocker.distribution_sampling import DistributionSampling
from hypothesis_testing.analysis_of_variance.one_way_analysis import OneWayAnalysis


class TestOneWayAnalysis(unittest.TestCase):
    def test_one_way(self):
        array_1 = DistributionSampling(seed=1).generate_bernoulli_samplings(0.5, 1000)
        array_2 = DistributionSampling(seed=22).generate_bernoulli_samplings(0.6, 20000)
        array_3 = DistributionSampling(seed=3).generate_normal_samplings(0.6, 0.2, 2000)
        array_4 = DistributionSampling(seed=4).generate_normal_samplings(0.7, 0.1, 1000)
        array_5 = DistributionSampling(seed=22).generate_bernoulli_samplings(0.6, 20000)

        result = OneWayAnalysis(array_1, array_2).test()
        self.assertTrue(result.rejected)
        result = OneWayAnalysis(array_3, array_4).test()
        self.assertTrue(result.rejected)
        result = OneWayAnalysis(array_2, array_5).test()
        self.assertTrue(not result.rejected)


if __name__ == "__main__":
    unittest.main()
