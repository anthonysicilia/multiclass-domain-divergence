import torch

from tqdm import tqdm

from ..models.stochastic import sample as sample_model
from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean
from .utils import to_device

class Estimator(BaseEstimator):

    def __init__(self, hypothesis, hypothesis_space, dataset,  
        device='cpu', verbose=False, sample=True):
        if not sample:
            raise ValueError('Only for stochastic models.')
        self.h1 = hypothesis.to(device)
        self.h2 = hypothesis_space().load_state_dict(
            hypothesis.state_dict()).to(device)
        with torch.no_grad():
            if sample: sample_model(self.h1)
        with torch.no_grad():
            if sample: sample_model(self.h2)
        self.dataset = hypothesis_space.test_dataloader(dataset)
        self.verbose = verbose
        self.to_device = lambda x: to_device(x, device)
    
    def _compute(self):
        dis = Mean()
        iterator = self.to_device(self.dataset)
        if self.verbose:
            iterator = tqdm(iterator)
        with torch.no_grad():
            self.h1.eval()
            self.h2.eval()
            for x, *_ in iterator:
                yhat1 = self.h1(x).argmax(dim=1)
                yhat2 = self.h2(x).argmax(dim=1)
                d = (yhat1 != yhat2).sum().item()
                dis.update(d, weight=x.size(0))
        return dis.compute()
        