import numpy as np
import random
import torch

class LazyCallable:

    def __init__(self, init, kwargs):
        self.init = init
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        for k, v in self.kwargs.items():
            # __init__ kwargs take precedent
            kwargs[k] = v
        return self.init(*args, **kwargs)

def lazy_kwarg_init(init, **kwargs):
    return LazyCallable(init, kwargs)

def set_random_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class PlaceHolderEstimator:

    def __init__(self, *args, **kwargs):
        pass

    def compute(self):
        return None