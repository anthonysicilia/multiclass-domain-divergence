from torch_two_sample.statistics_diff import \
    EnergyStatistic, MMDStatistic
from torch_two_sample.statistics_nondiff import \
    FRStatistic, KNNStatistic

from .base import Estimator as BaseEstimator
from .utils import lazy_kwarg_init, stack, approx_median_distance

class Estimator(BaseEstimator):

    STATS = ['mmd', 'energy', 'frs', 'knn']

    def __init__(self, stat, a, b, device='cpu'):
        super().__init__()
        
        if stat not in self.STATS:
            raise NotImplementedError(f'{stat} not implemented'
                ' in class two_sample.Estimator.')
        
        self.a = a # dataset, not dataloader
        self.b = b # dataset, not dataloader
        self.device = device
        
        if stat == self.STATS[0]: # mmd
            sigma = approx_median_distance(self.a, self.b)
            alpha = 0.5 / (sigma ** 2)
            self.stat = lazy_kwarg_init(
                MMDStatistic(len(self.a), len(self.b)),
                alphas=(alpha, ), ret_matrix=False)
            # stat is already (sample) normalized - we compute mean
            self.normalization = 1.
        elif stat == self.STATS[1]: # energy
            self.stat = lazy_kwarg_init(
                EnergyStatistic(len(self.a), len(self.b)),
                ret_matrix=False)
            # stat is already (sample) normalized - we compute mean
            self.normalization = 1.
        elif stat == self.STATS[2]: # frs
            self.stat = lazy_kwarg_init(
                FRStatistic(len(self.a), len(self.b)),
                norm=2, ret_matrix=False)
            # max is max number of edges between same sets.
            # MST has len(self.a) + len(self.b) - 1 edges
            # at least 1 edge must go between a and b
            # since tree is connected. 
            self.normalization = len(self.a) + len(self.b) - 2
        elif stat == self.STATS[3]: # knn
            self.stat = lazy_kwarg_init(
                KNNStatistic(len(self.a), len(self.b), 1),
                norm=2, ret_matrix=False)
            # max is max number of nodes with neighbor in same set.
            # each node in a sample could be connected to 
            # another node in the same sample.
            self.normalization = len(self.a) + len(self.b)
        else:
            # this should be unreachable
            raise NotImplementedError(f'{stat} not implemented'
                ' in class two_sample.Estimator.')
    
    def _compute(self):
        a = stack(self.a).to(self.device)
        b = stack(self.b).to(self.device)
        stat = self.stat(a, b).item()
        sample_normalized = stat / self.normalization
        return sample_normalized


    