class Estimator:
    """
    Abstract estimator class from which all estimators should
    be derived.
    """

    def __init__(self):
        self.estimate = None
    
    def _compute(self):
        raise NotImplementedError('Estimator is abstract.'
            ' Please derive a subclass and implement private'
            ' method _compute(...).')
    
    def compute(self):
        """
        Compute and save computation for subsequent calls.
        """
        if self.estimate is not None:
            # if self.estimate: hangs up on zero
            return self.estimate
        else:
            self.estimate = self._compute()
            return self.compute()

class Bound:
    """
    Abstract bound estimator class from which all bounds should
    be derived.
    """

    def __init__(self, estimator):
        self.estimator = estimator
        self.bound = None
    
    def _compute(self, estimate):
        raise NotImplementedError('Bound is abstract.'
            ' Please derive a subclass and implement private'
            ' method _compute(...).')
    
    def compute(self):
        """
        Compute and save computation for subsequent calls.
        """
        if self.bound is not None:
            # if self.estimate: hangs up on zero
            return self.bound
        else:
            self.bound = self._compute(self.estimator.compute())
            return self.compute()
    