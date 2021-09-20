from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean

class Estimator(BaseEstimator):

    def __init__(self, m, estimator):
        self.m = m
        self.estimator = estimator
    
    def _compute(self):
        mc_estimate = Mean()
        for _ in range(self.m):
            # lazy kwarg init
            est = self.estimator()
            mc_estimate.update(est.compute())
        return mc_estimate.compute()