import torch

from pathlib import Path
from tqdm import tqdm

from ..models.learners import PyTorchHypothesisSpace
from ..models.stochastic import Stochastic
from ..models.stochastic import sample as sample_model
from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean
from .utils import to_device

class PyTorchEstimator(BaseEstimator):

    # f'{5e-4:.3f}' = '0.001'; f'{4.9e-4:.3f}' = '0.000'
    TOLERANCE = 5e-4

    def __init__(self, hypothesis_space, dataset, device='cpu',
        verbose=False, catch_weights=False, sample=False, 
        kl_reg=False, cache=True):
        super().__init__()
        t = type(hypothesis_space)
        correct_space = issubclass(t, PyTorchHypothesisSpace)
        if not correct_space:
            raise ValueError('PyTorchEstimator requires'
                ' hypothesis_space to be derived '
                ' from PyTorchHypothesisSpace.')
        if kl_reg and not issubclass(t, Stochastic):
            raise ValueError('PyTorchEstimator requires'
                ' hypothesis_space to be derived '
                ' from Stochastic if kl_reg is True.')
        if cache and sample:
            raise ValueError('PyTorchEstimator cannot'
                ' cache intermediate feature representations'
                ' for stochastic models. Sampling would change'
                ' the chached values.')
        self.hspace = hypothesis_space
        self.finetune = self.hspace.finetune
        self.dataset = self.hspace.dataloader(dataset)
        self.device = device
        self.verbose = verbose
        self.catch_weights = catch_weights
        self.pass_h_to_loss_fn = kl_reg
        self.sample = sample
        # cache when possible: finetune=True, sample=False
        self.cache = cache and self.finetune
        if self.cache:
            loc = f'__mycache__/erm/{device}'
            Path(loc).mkdir(parents=True, exist_ok=True)
            self.cache_loc = lambda ext: f'{loc}/{ext}.pkl'
            
    def _cache(self, z, i):
        for j, idx in enumerate(i): 
            zj = z[j].detach().clone().unsqueeze(0)
            torch.save(zj, self.cache_loc(idx))
    
    def _load_cache(self, i):
        return torch.cat([torch.load(self.cache_loc(idx))
            for idx in i], dim=0)
    
    def _compute(self):
        hypothesis = self.hspace().to(self.device)
        # print('Device:', self.device)
        # print('On cuda:', next(hypothesis.parameters()).is_cuda)
        if self.finetune:
            foptim = self.hspace.foptimizer(
                hypothesis.f.parameters())
            coptim = self.hspace.optimizer(
                hypothesis.c.parameters())
            optims = [foptim, coptim]
        else:
            optim = self.hspace.optimizer(hypothesis.parameters())
            optims = [optim]
        schedulers = [self.hspace.scheduler(o) for o in optims]
        epoch_iterator = range(self.hspace.epochs)
        if self.verbose:
            epoch_iterator = tqdm(epoch_iterator)
        for e in epoch_iterator:
            error = Mean()
            if self.cache and e == self.hspace.fepochs:
                optims = optims[-1:]
                schedulers = schedulers[-1:]
            for tup in to_device(self.dataset, self.device):
                if self.catch_weights:
                    x, y, i, w, *_ = tup
                else:
                    x, y, i, *_ = tup; w = None
                hypothesis.train()
                if self.sample: sample_model(hypothesis)
                if self.cache and e >= self.hspace.fepochs:
                    if e == self.hspace.fepochs:
                        hypothesis.f.eval()
                        with torch.no_grad():
                            z = hypothesis.f(x)
                            self._cache(z, i)
                    else:
                        z = self._load_cache(i)
                    z = hypothesis.d(z)
                    yhat = hypothesis.c(z)
                else:
                    yhat = hypothesis(x)
                for o in optims: o.zero_grad()
                h = hypothesis if self.pass_h_to_loss_fn else None
                self.hspace.loss_fn(yhat, y, w=w, h=h).backward()
                for o in optims: o.step()
                errors = self.hspace.errors(yhat, y)
                error.update(errors.item(), weight=y.size(0))
            if self.verbose:
                epoch_iterator.set_postfix(
                    {'error' : error.compute()})
            if error.compute() < self.TOLERANCE:
                break
            for s in schedulers: s.step()
        return hypothesis.eval()


