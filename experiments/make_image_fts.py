import pickle
import torch

from tqdm import tqdm

from .datasets.images import PACS_DATASETS, OFFICEHOME_DATASETS
from .models.images import ResNet50HypothesisSpace

DEVICE = 'cuda:0'

if __name__ == '__main__':
    datasets = PACS_DATASETS + OFFICEHOME_DATASETS
    hspace = ResNet50HypothesisSpace()
    h = hspace().to(DEVICE)
    h.eval()
    for dname, dset in tqdm(datasets):
        data = []
        for train in [True, False]:
            loader = hspace.test_dataloader(dset(train=train))
            with torch.no_grad():
                for x, y, *_ in loader:
                    z = h.f(x.to(DEVICE)).cpu().numpy()
                    for i in range(y.size(0)):
                        data.append((z[i], y[i].item()))
        with open(f'{dname}_rn50fts.pkl', 'wb') as out:
            pickle.dump(data, out)
