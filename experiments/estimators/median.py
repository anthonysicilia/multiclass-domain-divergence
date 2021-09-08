from .base import Estimator as BaseEstimator

from statistics import median

class Estimator(BaseEstimator):

    """
    Estimate the expected value as the empicial mean.
    """

    def __init__(self):
        super().__init__()
        self.values = []
    
    def update(self, value):
        self.values.append(value)
    
    def _compute(self):
        return median(self.values)