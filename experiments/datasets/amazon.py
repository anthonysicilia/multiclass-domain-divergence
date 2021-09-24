import pickle
import torch

from .discourse import Dataset
from .utils import make_frozenset

def _reviews(domain, train, seed):
    loc = open('processed_acl/data.pkl', 'rb')
    data = pickle.load(loc)
    data = data[domain]
    vecs = [torch.tensor(x).float() for x,_ in data]
    labels = [y for _,y in data]
    dset = Dataset(vecs, labels)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=2)

def books(train=True, seed=0):
    return _reviews('books', train, seed)

def dvd(train=True, seed=0):
    return _reviews('dvd', train, seed)

def electronics(train=True, seed=0):
    return _reviews('electronics', train, seed)

def kitchen(train=True, seed=0):
    return _reviews('kitchen', train, seed)

DATASETS = [
    ('books', books), 
    ('dvd', dvd),
    ('electronics', electronics), 
    ('kitchen', kitchen)
]
