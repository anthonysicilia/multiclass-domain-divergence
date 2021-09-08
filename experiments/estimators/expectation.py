from .base import Estimator as BaseEstimator

class Estimator(BaseEstimator):

    """
    Estimate the expected value as the empicial mean.
    """

    def __init__(self):
        super().__init__()
        self.values = []
        self.weights = []
    
    def update(self, value, weight=1.):
        self.values.append(value)
        self.weights.append(weight)
    
    def _compute(self):
        return sum(self.values) / sum(self.weights)