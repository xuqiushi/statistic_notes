import scipy.stats

from hypothesis_testing.entity.compare_variable_result import CompareVariableResult
from hypothesis_testing.utils.arrays_normal_checker import ArraysNormalChecker
from hypothesis_testing.utils.arrays_variances_homogeneity_checker import (
    ArraysVariancesHomogeneityChecker,
)


class OneWayAnalysis:
    def __init__(self, array_1, array_2, p_thread=0.05):
        self.array_1 = array_1
        self.array_2 = array_2
        self.p_thread = p_thread

    def test(self) -> CompareVariableResult:
        normal_check_result = ArraysNormalChecker.check_double_normal(
            self.array_1, self.array_2
        )
        homogeneity_result = ArraysVariancesHomogeneityChecker.check_double_variances_homogeneity(
            self.array_1, self.array_2
        )
        if normal_check_result and homogeneity_result:
            statistic, p = scipy.stats.f_oneway(self.array_1, self.array_2)
        else:
            # Kruskal-Wallis H-test
            statistic, p = scipy.stats.kruskal(self.array_1, self.array_2)
        return CompareVariableResult(True, p, p <= self.p_thread)
