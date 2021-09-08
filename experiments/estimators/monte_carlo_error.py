from re import L
from .base import Estimator as BaseEstimator
from .error import Estimator as Error
from .expectation import Estimator as Mean

class Estimator(BaseEstimator):

    def __init__(self, m, *args, **kwargs):
        self.args = args
        kwargs['sample'] = True
        self.kwargs = kwargs
        self.m = m
    
    def _compute(self):
        error = Mean()
        for _ in range(self.m):
            est = Error(*self.args, **self.kwargs)
            error.update(est.compute())
        return error.compute()