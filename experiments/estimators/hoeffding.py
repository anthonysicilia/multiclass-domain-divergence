from math import sqrt, log

from ..bounds.functional import hoeffding_bound
from .base import Bound as BaseBound

class Bound(BaseBound):

    def __init__(self, estimator, m, delta):
        super().__(estimator)
        self.m = m
        self.delta = delta
    
    def _compute(self, estimate):
        return hoeffding_bound(estimate, self.m, self.delta)