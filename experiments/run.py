import argparse
import numpy as np
import random
import pandas as pd
import torch

from tqdm import tqdm

from .datasets.digits import DATASETS as DIGITS_DATASETS
from .datasets.discourse import PDTB_DATASETS, GUM_DATASETS
from .datasets.images import PACS_DATASETS, OFFICEHOME_DATASETS
from .datasets.utils import Multisource
from .estimators.bbsd import Estimator as BBSDEstimator
from .estimators.error import Estimator as Error
from .estimators.erm import PyTorchEstimator as HypothesisEstimator
from .estimators.hypothesis_divergence import Estimator \
    as HypothesisDivergence
from .estimators.hypothesis_class_divergence import Estimator \
    as HypothesisClassDivergence
from .estimators.two_sample import Estimator as TwoSampleEstimator
from .models.digits import DigitsHypothesisSpace
from .models.discourse import LinearHypothesisSpace, \
    NonLinearHypothesisSpace
from .models.images import ResNet18HypothesisSpace, \
    ResNet50HypothesisSpace
from .utils import lazy_kwarg_init, PlaceHolderEstimator, \
    set_random_seed

MIN_TRAIN_SAMPLES = 1000

GROUPS = [
    {
        'datasets' : DIGITS_DATASETS,
        'multisource' : False,
        'models' : [('cnn4l', DigitsHypothesisSpace)]
    },

    {
        'datasets' : PACS_DATASETS,
        'multisource' : False,
        'models' : [('rn18', ResNet18HypothesisSpace)]
    },

    {
        'datasets' : OFFICEHOME_DATASETS,
        'multisource' : False,
        'models' : [('rn50', ResNet50HypothesisSpace)]
    },

    {
        'datasets' : PDTB_DATASETS('sentence'),
        'multisource' : False, 
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    {
        'datasets' : PDTB_DATASETS('average'),
        'multisource' : False, 
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    {
        'datasets' : PDTB_DATASETS('pooled'),
        'multisource' : False, 
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    {
        'datasets' : GUM_DATASETS('sentence'),
        'multisource' : True, 
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    {
        'datasets' : GUM_DATASETS('average'),
        'multisource' : True, 
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    {
        'datasets' : GUM_DATASETS('pooled'),
        'multisource' : True, 
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    }

]

# most popular seeds in python according to:
# https://blog.semicolonsoftware.de/the-most-popular-random-seeds/
SEEDS = [0, 1, 100, 1234, 12345]

def _make_single_source_exps(group, dataset_seed):
    datasets = [
        (f'{desc}-{int(train)}', lazy_kwarg_init(dset,  
            train=train, seed=dataset_seed))
        for desc, dset in group['datasets'] 
        for train in [True, False]]
    exps = [(s, t, seed, hspace) 
        for i, s in enumerate(datasets)
        for j, t in enumerate(datasets)
        for seed in SEEDS
        for hspace in group['models']
        if i != j and len(s[1]) >= MIN_TRAIN_SAMPLES]
    return exps

def _make_multisource_exps(group, dataset_seed):
    datasets = [(f'{desc}', lazy_kwarg_init(dset, 
        train=True, seed=dataset_seed))
        for desc, dset in group['datasets']]
    exps = []
    for i in range(len(datasets)):
        for seed in SEEDS:
            for hspace in group['models']:
                target = datasets[i]
                source = [t for j,t in enumerate(datasets) 
                    if j != i]
                descs = [t[0] for t in source]
                dsets = [t[1] for t in source]
                source = ('+'.join(descs), Multisource(dsets))
                exps.append((source, target, seed, hspace))
    return exps
    
def make_experiments(group, dataset_seed):
    if group['multisource']:
        return _make_multisource_exps(group, dataset_seed)
    else:
        return _make_single_source_exps(group, dataset_seed)

def run_experiment(source, target, hspace, experiment_seed,
    verbose=False, device='cpu'):

    set_random_seed(experiment_seed)
    hypothesis_space = hspace(
        num_classes=source.num_classes)

    hypothesis = HypothesisEstimator(hypothesis_space, 
        source, verbose=verbose, device=device).compute()

    estimators = {
        'train_error' : Error(hypothesis, hypothesis_space,
            source, verbose=verbose, device=device),

        'transfer_error' : Error(hypothesis, hypothesis_space, 
            target, verbose=verbose, device=device),

        'our_h_divergence' : HypothesisDivergence(hypothesis, 
            hypothesis_space, source, target, verbose=verbose,
            device=device),

        'h_class_divergence' : HypothesisClassDivergence(
            hypothesis_space, source, target, verbose=verbose,
            binary=False, device=device),

        'mmd_bbsd' : BBSDEstimator(hypothesis, hypothesis_space,
            source, target, 'mmd', verbose=verbose, 
            device=device)
    }

    for stat in TwoSampleEstimator.STATS:
        estimators[stat] = TwoSampleEstimator(stat, 
            source, target, device=device)
        # tokens ???
            
    return {k : e.compute() for k, e in estimators.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose',
        action='store_true',
        help='Use verbose option for all estimators.')
    # making this an argument to help make parallel
    parser.add_argument('--dataset_seed',
        default=0,
        help='The seed to use for the dataset.')
    parser.add_argument('--device',
        default='cpu',
        help='The device to use for all experiments.')
    args = parser.parse_args()
    data = []
    for group_num, group in enumerate(GROUPS):
        exps = make_experiments(group, args.dataset_seed)
        for (sname, s), (tname, t), seed, (hname, h) in exps:
            res = run_experiment(s, t, h, seed, 
                verbose=args.verbose, device=args.device)
            res['source'] = sname
            res['target'] = tname
            res['dataset_seed'] = args.dataset_seed
            res['experiment_seed'] = seed
            res['group_num'] = group_num
            res['hspace'] = hname
            data.append(res)
            # incrementally save results
            write_loc = f'results-{args.dataset_seed}.csv'
            pd.DataFrame(data).to_csv(write_loc)