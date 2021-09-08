from ..bounds.functional import mauer_bound
from ..models.stochastic import compute_kl
from .base import Bound as BaseBound

class Bound(BaseBound):

    def __init__(self, estimator, hypothesis, m, n, delta):
        super().__init__(estimator)
        self.h = hypothesis
        self.m = m
        self.n = n
        self.delta = delta
    
    def _compute(self, estimate):
        kl_div = compute_kl(self.h)
        return mauer_bound(estimate, self.m, self.n, self.delta, 
            kl_div)