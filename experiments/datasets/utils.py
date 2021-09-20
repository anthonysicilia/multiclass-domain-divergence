import copy
import operator
import torch

from random import Random

from ..utils import set_random_seed

class Multisource(torch.utils.data.Dataset):

    def __init__(self, dsets):
        self.dsets = dsets
        self.num_classes = max([dset.num_classes
            for dset in self.dsets])
    
    def __len__(self):
        return sum(len(d) for d in self.dsets)
    
    def __getitem__(self, index):
        oidx = index
        for dset in self.dsets:
            if index >= len(dset):
                index -= len(dset)
            else:
                x, y, *_ = dset[index]
                return x, y, oidx
        raise StopIteration()

class Dataset(torch.utils.data.Dataset):

    def __init__(self, dataset, seed, keep=None, num_classes=None):
        set_random_seed(seed)
        # makes changes to global seed !!!
        indices = keep if keep is not None else range(len(dataset))
        self.dataset = []
        for index in indices:
            self.dataset.append(dataset[index])
        if num_classes is None:
            # infer from provided dataset
            self.num_classes = dataset.num_classes
        else:
            self.num_classes = num_classes
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        elem = copy.deepcopy(self.dataset[index])
        return (*elem, index)

def make_frozenset(dset, train=True, seed=0, num_classes=None):
    scoped_random = Random(seed)
    x = [scoped_random.random() for _ in range(len(dset))]
    comp = operator.le if train else operator.gt
    train = lambda xi: comp(xi, 0.5)
    indices = [i for i, xi in enumerate(x) if train(xi)]
    return Dataset(dset, seed, keep=indices, 
        num_classes=num_classes)

class GaussianNoise(object):
    """
    Based on this answer:
    https://discuss.pytorch.org/t/
    how-to-add-noise-to-mnist-dataset
    -when-using-pytorch/59745/2
    """

    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std \
            + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ \
            + '(mean={0}, std={1})'.format(self.mean, self.std)

