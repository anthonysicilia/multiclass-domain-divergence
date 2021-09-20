import torch
import torch.nn.functional as F

from torch_two_sample.statistics_diff import MMDStatistic
from tqdm import tqdm

from .base import Estimator as BaseEstimator
from .utils import lazy_kwarg_init, stack, to_device, \
    approx_median_distance

def _total_variation(a, b):
    k = a.size(1)
    a = F.one_hot(a.argmax(dim=1), num_classes=k) \
        .float().mean(dim=0)
    b = F.one_hot(b.argmax(dim=1), num_classes=k) \
        .float().mean(dim=0)
    return (a - b).abs().sum()

class Estimator(BaseEstimator):

    STATS = ['mmd', 'tv']
    MAX_SAMPLES = 1000

    def __init__(self, hypothesis, hypothesis_space, a, b, stat,
        device='cpu', verbose=True):
        super().__init__()
        
        self.to_device = lambda x: to_device(x, device)
        self.verbose = verbose

        if stat not in self.STATS:
            raise NotImplementedError(f'{stat} not implemented'
                ' in class bbsd.Estimator.')

        if hypothesis is None:
            hypothesis = hypothesis_space()
        self.hypothesis = hypothesis.to(device).eval()
        self.dataloader = hypothesis_space.test_dataloader
        self.a = a # dataset, not dataloader
        self.b = b # dataset, not dataloader

        n1 = min(len(self.a), self.MAX_SAMPLES)
        n2 = min(len(self.b), self.MAX_SAMPLES)

        if stat == self.STATS[0]:
            sigma = approx_median_distance(
                self._get_softmax_output(self.a),
                self._get_softmax_output(self.b))
            sigma = max(sigma, 1e-4)
            alpha = 0.5 / (sigma ** 2)
            self.stat = lazy_kwarg_init(
                MMDStatistic(n1, n2),
                alphas=(alpha, ), ret_matrix=False)
        elif stat == self.STATS[1]:
            self.stat = _total_variation

    def _get_softmax_output(self, dset):
        outputs = []
        iterator = self.to_device(self.dataloader(dset))
        if self.verbose:
            iterator = tqdm(iterator)
        with torch.no_grad():
            for x, *_ in iterator:
                yhat = self.hypothesis(x).softmax(dim=1)
                for i in range(yhat.size(0)):
                    outputs.append((yhat[i].cpu().numpy(), ))
        return stack(outputs, self.MAX_SAMPLES)
    
    def _compute(self):
        return self.stat(self._get_softmax_output(self.a),
            self._get_softmax_output(self.b)).item()