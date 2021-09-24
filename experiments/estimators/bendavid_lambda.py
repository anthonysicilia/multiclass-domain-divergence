import torch

from .base import Estimator as BaseEstimator
from .erm import PyTorchEstimator as HypothesisEstimator
from .expectation import Estimator as Mean
from .utils import to_device

class JointSet(torch.utils.data.Dataset):

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.weight_a = (max(len(a), len(b))) / len(a)
        self.weight_b = (max(len(a), len(b))) / len(b)

    def __len__(self):
        return len(self.a) + len(self.b)
    
    def __getitem__(self, index):
        oidx = index
        if index >= len(self.a):
            index = index - len(self.a)
            x, y, *_ = self.b.__getitem__(index)
            return (x, y, oidx, self.weight_b, 1)
        else:
            x, y, *_ = self.a.__getitem__(index)
            return (x, y, oidx, self.weight_a, 0)

class Estimator(BaseEstimator):

    def __init__(self, hypothesis_space, a, b,
        device='cpu', verbose=False):
        super().__init__()
        self.a = a
        self.b = b
        self.hspace = hypothesis_space
        self.device = device
        self.verbose = verbose
    
    def _compute(self):
        dataset = JointSet(self.a, self.b)
        hypothesis = HypothesisEstimator(self.hspace, 
            dataset, device=self.device, verbose=self.verbose, 
            catch_weights=True).compute()
        iterator = to_device(self.hspace.test_dataloader(dataset),
            self.device)
        error0 = Mean()
        error1 = Mean()
        with torch.no_grad():
            hypothesis.eval()
            for x, y, _, _, z in iterator:
                yhat = hypothesis(x).argmax(dim=1)
                errors = (yhat != y)
                errors0 = errors[z==0].sum().item()
                errors1 = errors[z==1].sum().item()
                error0.update(errors0, weight=(z==0).sum().item())
                error1.update(errors1, weight=(z==1).sum().item())
        return error0.compute() + error1.compute()
