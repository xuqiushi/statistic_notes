import scipy
import scipy.stats
import numpy as np


class NormalDistributionTestCollection:
    SINGLE_LIMIT = 0.05

    @classmethod
    def test_normal_distribution(cls, array: np.ndarray) -> bool:
        if array.ndim > 1:
            raise ValueError(f"array dim must one")
        if array.shape[0] < 3:
            return False
        if 3 <= array.shape[0] <= 50:
            # Shapiro-Wilk 一般3-50，在大于5000时p值可能有问题。
            statistic, p = scipy.stats.shapiro(array)
            return p > cls.SINGLE_LIMIT
        else:
            # Kolmogorov-Smirnov 一般50以上适用，这里联合下一个进行联合判断
            statistic, p = scipy.stats.kstest(array, "norm")
            return p > cls.SINGLE_LIMIT
            # 这里考虑是否假如此判断与ks一起
            # Anderson-Darling 一般50以上时使用，这里联合下一个进行联合判断
            # anderson_result = scipy.stats.anderson(array, dist="norm")
