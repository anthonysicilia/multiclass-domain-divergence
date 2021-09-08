import torch

from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean
from .erm import PyTorchEstimator as HypothesisEstimator, \
    PyTorchHypothesisSpace
from .utils import to_device


class DomainLabeledSet(torch.utils.data.Dataset):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b
        self.weight_a = max(len(a), len(b)) / len(a)
        self.weight_b = max(len(a), len(b)) / len(b)
    
    def __len__(self):
        return len(self.a) + len(self.b)
    
    def __getitem__(self, index):
        oidx = index
        if index >= len(self.a):
            index = index - len(self.a)
            x, *_ = self.b.__getitem__(index)
            return (x, 1, oidx, self.weight_b)
        else:
            x, *_ = self.a.__getitem__(index)
            return (x, 0, oidx, self.weight_a)

def _binary_sym_diff_classify(yhat1, yhat2):
    if len(yhat1.size()) == 2:
        # this is techincally incorrect for multiclass
        return -(yhat1[:, 0] * yhat2[:, 0])
    elif len(yhat1.size()) == 1:
        return -(yhat1 * yhat2)
    else:
        raise NotImplementedError(
            f'Size of output {yhat1.size()}'
            ' not implemented in _binary_sym_diff_classify(...)')

def _multiclass_sym_diff_classify(yhat1, yhat2):
    if len(yhat1.size()) <= 1:
        raise NotImplementedError(
            f'Size of output {yhat1.size()}'
            ' not implemented in _multiclass_sym_diff_classify(...)')
    batch_size = yhat1.size(0)
    num_classes = yhat1.size(1)
    make_pos = torch.nn.Softplus()
    A = torch.einsum('bi,bj->bij', make_pos(yhat1), make_pos(yhat2))
    I = torch.eye(num_classes).bool().repeat(batch_size, 1, 1)
    max_off_diag_A = A[~I].reshape(batch_size, -1).max(dim=1).values
    max_diag_A = A[I].reshape(batch_size, -1).max(dim=1).values
    return max_off_diag_A - max_diag_A

def sym_diff_classify(yhat1, yhat2, binary=True):
    if binary:
        return _binary_sym_diff_classify(yhat1, yhat2)
    else:
        return _multiclass_sym_diff_classify(yhat1, yhat2)

class SymmetricDifferenceHypothesis(torch.nn.Module):

    """
    Returns an (untrained) hypothesis within 
    the Symmetric Difference Hypothesis Class
    := {h xor h | h in H} (= {h neq h | h in H})
    """

    class Fnet(torch.nn.Module):

        def __init__(self, h1f, h2f):
            super().__init__()
            self.h1f = h1f
            self.h2f = h2f
        
        def forward(self, x):
            x = (self.h1f(x), self.h2f(x))
            return torch.cat(tuple(xi.unsqueeze(1) 
                for xi in x), dim=1)
    
    class Dnet(torch.nn.Module):

        def __init__(self, h1d, h2d):
            super().__init__()
            self.h1d = h1d
            self.h2d = h2d
        
        def forward(self, x):
            x[:, 0] = self.h1d(x[:, 0])
            x[:, 1] = self.h2d(x[:, 1])
            return x
    
    class Cnet(torch.nn.Module):

        def __init__(self, h1c, h2c, binary):
            super().__init__()
            self.h1c = h1c
            self.h2c = h2c
            self.binary = binary
        
        def forward(self, x):
            yhat1 = self.h1c(x[:, 0])
            yhat2 = self.h2c(x[:, 1])
            return sym_diff_classify(yhat1, yhat2, binary=self.binary)

    def __init__(self, hypothesis_space, finetune, binary):
        super().__init__()
        self.h1 = hypothesis_space()
        self.h2 = hypothesis_space()
        self.binary = binary
        if finetune:
            self.f = self.Fnet(self.h1.f, self.h2.f)
            self.d = self.Dnet(self.h1.d, self.h2.d)
            self.c = self.Cnet(self.h1.c, self.h2.c, binary)

    def forward(self, x):
        yhat1 = self.h1(x)
        yhat2 = self.h2(x)
        return sym_diff_classify(yhat1, yhat2, binary=self.binary)

class SymmetricDifferenceHypothesisSpace(PyTorchHypothesisSpace):

    def __init__(self, hypothesis_space, binary=True):
        super().__init__(
            1,
            hypothesis_space.epochs, 
            hypothesis_space.optimizer,
            hypothesis_space.scheduler, 
            hypothesis_space.batch_size // 4, 
            hypothesis_space.finetune)
        self.underlying = hypothesis_space
        self.binary = binary
        if self.finetune:
            self.foptimizer = hypothesis_space.foptimizer
            self.fepochs = hypothesis_space.fepochs
    
    def __call__(self):
        return SymmetricDifferenceHypothesis(self.underlying, 
            self.finetune, self.binary)

class Estimator(BaseEstimator):

    def __init__(self, hypothesis_space, a, b, device='cpu', 
        verbose=False, binary=True):
        super().__init__()
        self.hspace = SymmetricDifferenceHypothesisSpace( 
            hypothesis_space, binary=binary)
        self.a = a
        self.b = b
        self.device = device
        self.verbose = verbose
    
    def _compute(self):
        x = self._asymmetric_compute(self.a, self.b)
        y = self._asymmetric_compute(self.b, self.a)
        return max(x, y)
    
    def _asymmetric_compute(self, a, b):
        dataset = DomainLabeledSet(a, b)
        hypothesis = HypothesisEstimator(self.hspace, dataset,
            device=self.device, verbose=self.verbose,
            catch_weights=True).compute()
        iterator = to_device(self.hspace.test_dataloader(dataset),
            self.device)
        prob_ind_a = Mean()
        prob_ind_b = Mean()
        with torch.no_grad():
            hypothesis.eval()
            for x, y, *_ in iterator:
                yhat = (hypothesis(x) >= 0).long()
                a_ind = yhat[y == 0] == 1
                b_ind = yhat[y == 1] == 1
                prob_ind_a.update(a_ind.sum().item(), 
                    weight=len(a_ind))
                prob_ind_b.update(b_ind.sum().item(), 
                    weight=len(b_ind))
        return abs(prob_ind_a.compute() - prob_ind_b.compute())