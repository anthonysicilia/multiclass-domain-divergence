import torch

from random import Random, shuffle

from .median import Estimator as Median

def to_device(iterator, device):
    for x in iterator:
        arr = []
        for xi in x:
            if type(xi) == dict:
                xi = {k : v.to(device) for k,v in xi.items()}
                arr.append(xi)
            else:
                arr.append(xi.to(device))
        yield tuple(arr)

def lazy_kwarg_init(init, **kwargs):

    class LazyCallable:

        def __init__(self, init, kwargs):
            self.init = init
            self.kwargs = kwargs
        
        def __call__(self, *args):
            return self.init(*args, **kwargs)
    
    return LazyCallable(init, kwargs)

def stack(dataset, max_samples=None):
    if type(dataset[0]) != tuple:
        raise ValueError('If dataset.__getitem__(...) does not'
            ' return tuples, this function may malfunction.'
            ' Please, ensure dataset.__getitem__(...) returns'
            ' a tuple; e.g., return "(x, )" in place of "x".'
            f' The misused type was {type(dataset[0])}.')
    arr = [x.reshape(1, -1) if torch.is_tensor(x)
        else torch.tensor(x.reshape(1, -1)) 
        for x, *_ in dataset]
    if max_samples is not None and len(arr) > max_samples:
        shuffle(arr)
        return torch.cat(arr)[:max_samples]
    else:
        return torch.cat(arr)

def approx_median_distance(a, b, nsamples=100, seed=None):
    pooled = [(x, _) for x,*_ in a] + [(x, _) for x,*_ in b]
    if seed is not None: # use specified seed, don't modify global
        Random(seed).shuffle(pooled)
    else: # use global seed
        shuffle(pooled)
    pooled = stack(pooled)[:nsamples]
    med = Median()
    for i, xi in enumerate(pooled):
        for j, xj in enumerate(pooled):
            if i != j:
                sum_of_squares = ((xi - xj) ** 2).sum()
                dist = torch.sqrt(sum_of_squares).item()
                med.update(dist)
    return med.compute()
