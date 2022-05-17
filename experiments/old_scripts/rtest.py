import torch
import matplotlib.pyplot as plt

from tqdm import tqdm

from ..models.digits import DigitsHypothesisSpace
from ..datasets.digits import mnist

DEVICE = 'cuda:0'

if __name__ == '__main__':
    hspace = DigitsHypothesisSpace()
    uni = torch.distributions.uniform.Uniform(0, 1)
    errors = []
    for _ in tqdm(range(1000)):
        model = hspace().to(DEVICE)
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(uni.rsample(param.size()).to(DEVICE))
        with torch.no_grad():
            e = 0
            n = 0
            for x, y, *_ in hspace.dataloader(mnist()):
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                yhat = model(x).argmax(dim=-1)
                e += (yhat != y).sum().item()
                n += y.size(0)
            errors.append(e / n)
    
    plt.hist(errors, bins=20)
    plt.savefig('errors')