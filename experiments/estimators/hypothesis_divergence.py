import torch

from tqdm import tqdm

from .base import Estimator as BaseEstimator
from .expectation import Estimator as Mean
from .erm import PyTorchEstimator as HypothesisEstimator
from .utils import to_device

class DisagreementSet(torch.utils.data.Dataset):

    def __init__(self, a, b, hypothesis, dataloader, device='cpu'):
        super().__init__()
        self.a = a
        self.b = b
        self.weight_a = (max(len(a), len(b))) / len(a)
        self.weight_b = (max(len(a), len(b))) / len(b)
        self.predictions = []
        self.labels = []
        hypothesis.eval()
        with torch.no_grad():
            # predictions for a
            dataset = to_device(dataloader(self.a), device)
            for x, *_ in dataset:
                yhat = hypothesis(x)
                yhat = yhat.argmax(dim=1)
                for i in range(yhat.size(0)):
                    self.labels.append(yhat[i].cpu().item())
                    self.predictions.append(yhat[i].cpu().item())
            # predictions for b
            dataset = to_device(dataloader(self.b), device)
            for x, *_ in dataset:
                yhat = hypothesis(x)
                # use the second prediction, could also randomly
                # sample, or pick lowest. Reasoning is, it should 
                # be easiest to confuse the first and second 
                # most confident prediction
                faux_yhat = yhat.topk(k=2, dim=1).indices[:, 1]
                yhat = yhat.argmax(dim=1)
                for i in range(yhat.size(0)):
                    self.labels.append(faux_yhat[i].cpu().item())
                    self.predictions.append(yhat[i].cpu().item())
        
    def __len__(self):
        return len(self.a) + len(self.b)
    
    def __getitem__(self, index):
        pred = self.predictions[index]
        label = self.labels[index]
        oidx = index
        if index >= len(self.a):
            index = index - len(self.a)
            x, *_ = self.b.__getitem__(index)
            return (x, label, oidx, self.weight_b, pred, 1)
        else:
            x, *_ = self.a.__getitem__(index)
            return (x, label, oidx, self.weight_a, pred, 0)

class Estimator(BaseEstimator):

    def __init__(self, hypothesis, hypothesis_space, a, b, 
        device='cpu', verbose=False):
        super().__init__()
        self.hypothesis = hypothesis.to(device)
        self.hspace = hypothesis_space
        self.a = a
        self.b = b
        self.device = device
        self.verbose = verbose
    
    def _compute(self):
        x = self._asymmetric_compute(self.a, self.b)
        y = self._asymmetric_compute(self.b, self.a)
        return max(x, y)
    
    def _asymmetric_compute(self, a, b):
        dataset = DisagreementSet(a, b, self.hypothesis, 
            self.hspace.test_dataloader, device=self.device)
        hypothesis = HypothesisEstimator(self.hspace, dataset,
            device=self.device, verbose=self.verbose, 
            catch_weights=True).compute()
        iterator = to_device(self.hspace.test_dataloader(dataset),
            self.device)
        prob_dis_a = Mean()
        prob_dis_b = Mean()
        with torch.no_grad():
            hypothesis.eval()
            for x, _, _, _, h_pred, z in iterator:
                yhat = hypothesis(x).argmax(dim=1)
                a_ind = (yhat[z == 0] != h_pred[z == 0])
                b_ind = (yhat[z == 1] != h_pred[z == 1])
                prob_dis_a.update(a_ind.sum().item(), 
                    weight=len(a_ind))
                prob_dis_b.update(b_ind.sum().item(), 
                    weight=len(b_ind))
        return abs(prob_dis_a.compute() - prob_dis_b.compute())