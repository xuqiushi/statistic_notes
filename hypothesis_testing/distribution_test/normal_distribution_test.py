import scipy
import scipy.stats
import numpy as np


class NormalDistributionTest:
    SINGLE_LIMIT = 0.05

    @classmethod
    def test_normal_distribution(cls, array: np.ndarray) -> bool:
        if array.ndim > 1:
            raise ValueError(f"array dim must one")
        if array.shape[0] < 3:
            return False
        if 3 <= array.shape[0] <= 50:
            # Shapiro-Wilk 一般3-50，在大于5000时p值可能有问题。
            statistic, p = scipy.stats.shapiro(
                (array - array.mean()) / array.std(ddof=1)
            )
            return p > cls.SINGLE_LIMIT
        elif 50 < array.shape[0] <= 300:
            # Kolmogorov-Smirnov 一般50-300以上适用，这里联合下一个进行联合判断
            statistic, p = scipy.stats.kstest(
                (array - array.mean()) / array.std(ddof=1), "norm"
            )
            return p > cls.SINGLE_LIMIT
            # 这里考虑是否假如此判断与ks一起
            # Anderson-Darling 一般50以上时使用，这里联合下一个进行联合判断
            # anderson_result = scipy.stats.anderson(array, dist="norm")
        else:
            # 如果大于300，则直接使用联合判断，这里融合了峰度与偏度，可以作为类似正态分布的判断
            statistic, p = scipy.stats.normaltest(
                (array - array.mean()) / array.std(ddof=1)
            )
            return p > cls.SINGLE_LIMIT
