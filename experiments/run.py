import argparse
import numpy as np
import random
import pandas as pd
import torch

from pathlib import Path
# from tqdm import tqdm

from .datasets.digits import DATASETS as DIGITS_DATASETS, \
    ROTATION_PAIRS, NOISY_PAIRS, FAKE_PAIRS
from .datasets.discourse import PDTB_DATASETS, GUM_DATASETS
from .datasets.images import PACS_DATASETS, OFFICEHOME_DATASETS, \
    PACS_FTS_DATASETS, OFFICEHOME_FTS_DATASETS
from .datasets.utils import Multisource
from .estimators.bbsd import Estimator as BBSDEstimator
from .estimators.bendavid_lambda import Estimator \
    as BenDavidLambdaEstimator
from .estimators.error import Estimator as Error
from .estimators.erm import PyTorchEstimator as HypothesisEstimator
from .estimators.hypothesis_divergence import Estimator \
    as HypothesisDivergence
from .estimators.hypothesis_class_divergence import Estimator \
    as HypothesisClassDivergence
from .estimators.monte_carlo import Estimator as MonteCarloEstimator
from .estimators.stoch_disagreement import Estimator \
    as StochDisagreementEstimator
from .estimators.stoch_joint_error import Estimator \
    as StochJointErrorEstimator
from .estimators.two_sample import Estimator as TwoSampleEstimator
from .models.digits import DigitsHypothesisSpace
from .models.discourse import LinearHypothesisSpace, \
    NonLinearHypothesisSpace
from .models.images import ResNet18HypothesisSpace, \
    ResNet50HypothesisSpace
from .models.stochastic import Stochastic
from .utils import PlaceHolderEstimator, lazy_kwarg_init, set_random_seed

MIN_TRAIN_SAMPLES = 1000
MC_SAMPLES = 10_000
SIGMA_PRIOR = 0.01

GROUPS = {
    'digits' : {
        'name' : 'digits',
        'datasets' : DIGITS_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('cnn4l', DigitsHypothesisSpace)]
    },

    'r_digits' : {
        'name' : 'r_digits',
        'datasets' : ROTATION_PAIRS,
        'multisource' : False,
        'make_pairs' : False,
        'two_sample' : True,
        'models' : [('cnn4l', DigitsHypothesisSpace)]
    },

    'n_digits' : {
        'name' : 'n_digits',
        'datasets' : NOISY_PAIRS,
        'multisource' : False,
        'make_pairs' : False,
        'two_sample' : True,
        'models' : [('cnn4l', DigitsHypothesisSpace)]
    },

    'f_digits' : {
        'name' : 'f_digits',
        'datasets' : FAKE_PAIRS,
        'multisource' : False,
        'make_pairs' : False,
        'two_sample' : True,
        'models' : [('cnn4l', DigitsHypothesisSpace)]
    },

    # Q: runs 10x longer after exp 42 ?? 
    # A: photo too small to train
    'pacs' : {
        'name' : 'pacs',
        'datasets' : PACS_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : False,
        'models' : [('rn18', ResNet18HypothesisSpace)]
    },

    'officehome' : {
        'name' : 'officehome',
        'datasets' : OFFICEHOME_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : False,
        'models' : [('rn50', ResNet50HypothesisSpace)]
    },

    'officehome_fts' : {
        'name' : 'officehome_fts',
        'datasets' : OFFICEHOME_FTS_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [
            ('lin', lazy_kwarg_init(LinearHypothesisSpace, 
                num_inputs=2048)),
            ('fc4l', lazy_kwarg_init(NonLinearHypothesisSpace, 
                num_inputs=2048))]
    },

    'pacs_fts' : {
        'name' : 'pacs_fts',
        'datasets' : PACS_FTS_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [
            ('lin', lazy_kwarg_init(LinearHypothesisSpace, 
                num_inputs=2048)),
            ('fc4l', lazy_kwarg_init(NonLinearHypothesisSpace, 
                num_inputs=2048))]
    },

    'pdtb_sentence' : {
        'name' : 'pdtb_sentence',
        'datasets' : PDTB_DATASETS('sentence'),
        'multisource' : False, 
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'pdtb_average' : {
        'name' : 'pdtb_average',
        'datasets' : PDTB_DATASETS('average'),
        'multisource' : False, 
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'pdtb_pooled' : {
        'name' : 'pdtb_pooled',
        'datasets' : PDTB_DATASETS('pooled'),
        'multisource' : False, 
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'gum_sentence' : {
        'name' : 'gum_sentence',
        'datasets' : GUM_DATASETS('sentence'),
        'multisource' : True, 
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'gum_average' : {
        'name' : 'gum_average',
        'datasets' : GUM_DATASETS('average'),
        'multisource' : True, 
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'gum_pooled' : {
        'name' : 'gum_pooled',
        'datasets' : GUM_DATASETS('pooled'),
        'multisource' : True, 
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    }

}

# most popular seeds in python according to:
# https://blog.semicolonsoftware.de/the-most-popular-random-seeds/
# SEEDS = [0, 1, 100, 1234, 12345]
# use 0, 1, 100 for dataset seeds and 100, ... for exp seeds
# SEEDS = [100 , 1234, 12345]

class SeededEstimator:

    def __init__(self, estimator, seed):
        self.seed = seed
        self.estimator = estimator
    
    def compute(self):
        set_random_seed(self.seed)
        return self.estimator.compute()

def _make_single_source_exps(group, dataset_seed):
    datasets = [
        (f'{desc}-{int(train)}', lazy_kwarg_init(dset,  
            train=train, seed=dataset_seed))
        for desc, dset in group['datasets'] 
        for train in [True, False]]
    exps = [(s, t, hspace) 
        for i, s in enumerate(datasets)
        for j, t in enumerate(datasets)
        for hspace in group['models']
        if i != j]
    return exps

def _make_multisource_exps(group, dataset_seed):
    datasets = [(f'{desc}', lazy_kwarg_init(dset, 
        train=True, seed=dataset_seed))
        for desc, dset in group['datasets']]
    exps = []
    for i in range(len(datasets)):
        for hspace in group['models']:
            target = datasets[i]
            source = [t for j,t in enumerate(datasets) 
                if j != i]
            descs = [t[0] for t in source]
            dsets = [t[1] for t in source]
            source = ('+'.join(descs), Multisource(dsets))
            exps.append((source, target, hspace))
    return exps

def _make_prepackaged_exps(group, dataset_seed):
    exps = []
    for (sname, s, tname, t) in group['datasets']:
        for hspace in group['models']:
            s = lazy_kwarg_init(s, train=True, 
                seed=dataset_seed)
            t = lazy_kwarg_init(t, train=True, 
                seed=dataset_seed)
            source = (sname, s)
            target = (tname, t)
            exps.append((source, target, hspace))
    return exps
    
def make_experiments(group, dataset_seed):
    if group['make_pairs']:
        if group['multisource']:
            return _make_multisource_exps(group, dataset_seed)
        else:
            return _make_single_source_exps(group, dataset_seed)
    else:
        return _make_prepackaged_exps(group, dataset_seed)

def disjoint_split(dataset, seed=0):

    random.seed(seed)
    indices = [(i, random.random() <= 0.5) 
        for i in range(len(dataset))]
    prefix = [i for i, b in indices if b]
    bound = [i for i, b in indices if not b]

    class Dataset:

        def __init__(self, a, index_list):
            self.index_list = index_list
            self.a = a
        
        def __len__(self):
            return len(self.index_list)
        
        def __getitem__(self, index):
            data = self.a.__getitem__(self.index_list[index])
            data[2] = index
            return data
    
    return Dataset(dataset, prefix), Dataset(dataset, bound)

def run_stoch_experiment(source, target, hspace, seed,
    verbose=False, device='cpu'):

    raise NotImplementedError('Not debugged yet...')

    prefix_source, bound_source = disjoint_split(source, seed)
    prefix_source_samples = len(prefix_source)
    bound_source_samples = len(bound_source)
    target_samples = len(target)
    hypothesis_space = hspace(num_classes=source.num_classes)
    prior_estimator = HypothesisEstimator(hypothesis_space, 
        prefix_source, verbose=verbose, device=device)
    prior = SeededEstimator(prior_estimator, seed).compute()
    stoch_hypothesis_space = Stochastic(hypothesis_space, 
        prior=prior, m=bound_source_samples, delta=0.05,
        sigma_prior=SIGMA_PRIOR)
    posterior_estimator = HypothesisEstimator(
        stoch_hypothesis_space, source, kl_reg=True, 
        sample=True, cache=False, verbose=verbose, 
        device=device)
    posterior = SeededEstimator(posterior_estimator, 
        seed).compute()

    estimators = {
        'train_error' : lazy_kwarg_init(Error,
            hypothesis=posterior, 
            hypothesis_space=hypothesis_space, 
            dataset=bound_source, verbose=verbose, device=device),

        'transfer_error' : lazy_kwarg_init(Error, 
            hypothesis=posterior, 
            hypothesis_space=hypothesis_space, 
            dataset=target, verbose=verbose, device=device),
        
        'source_dis' : lazy_kwarg_init(StochDisagreementEstimator,
            hypothesis=posterior, 
            hypothesis_space=stoch_hypothesis_space, 
            dataset=bound_source, 
            device=device, verbose=verbose, sample=True) ,
        
        'source_jerror' : lazy_kwarg_init(StochJointErrorEstimator,
            hypothesis=posterior, 
            hypothesis_space=stoch_hypothesis_space, 
            dataset=bound_source, 
            device=device, verbose=verbose, sample=True),
        
        'target_dis' : lazy_kwarg_init(StochDisagreementEstimator,
            hypothesis=posterior, 
            hypothesis_space=stoch_hypothesis_space, 
            dataset=target, 
            device=device, verbose=verbose, sample=True),
        
        'target_jerror' : lazy_kwarg_init(StochJointErrorEstimator,
            hypothesis=posterior, 
            hypothesis_space=stoch_hypothesis_space, 
            dataset=target, 
            device=device, verbose=verbose, sample=True),
        
        'our_h_divergence_stoch' : lazy_kwarg_init(
            HypothesisDivergence,
            hypothesis=posterior, 
            hypothesis_space=stoch_hypothesis_space, 
            a=bound_source, b=target, 
            verbose=verbose, device=device),
    }

    estimators = {MonteCarloEstimator(MC_SAMPLES, e) 
        for e in estimators}
    
    estimators['h_class_divergence_stoch'] = \
        HypothesisClassDivergence(hypothesis_space, 
        bound_source, target, verbose=verbose,
        binary=False, device=device)
        
    estimators['ben_david_lambda_stoch'] = \
        BenDavidLambdaEstimator(hypothesis_space, 
        bound_source, target, 
        device=device, verbose=verbose),

    return {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

def run_experiment(source, target, hspace, seed,
    two_sample, verbose=False, device='cpu'):

    hypothesis_space = hspace(
        num_classes=source.num_classes)
    hestimator = HypothesisEstimator(hypothesis_space, 
        source, verbose=verbose, device=device)
    hypothesis = SeededEstimator(hestimator, seed).compute()

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
        
        'ben_david_lambda' : BenDavidLambdaEstimator(
            hypothesis_space, source, target, 
            device=device, verbose=verbose),

        'mmd_bbsd' : BBSDEstimator(hypothesis, hypothesis_space,
            source, target, 'mmd', verbose=verbose, 
            device=device)
    }

    for stat in TwoSampleEstimator.STATS:
        # to big for gpu
        estimators[stat] = TwoSampleEstimator(stat, 
            source, target, device=device) \
                if two_sample else PlaceHolderEstimator()
            
    return {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--group',
        type=str,
        default='digits',
        help='The group of experiments to run.')
    parser.add_argument('--verbose',
        action='store_true',
        help='Use verbose option for all estimators.')
    # making this an argument to help make parallel
    parser.add_argument('--dataset_seed',
        default=0,
        type=int,
        help='The seed to use for the dataset.')
    parser.add_argument('--experiment_seed',
        default=100,
        type=int,
        help='The seed to use for the experiment.')
    parser.add_argument('--device',
        type=str,
        default='cpu',
        help='The device to use for all experiments.')
    parser.add_argument('--test',
        action='store_true',
        help='Take steps for a shorter run'
        ' (results will be invalid).')
    args = parser.parse_args()
    data = []
    group = GROUPS[args.group]
    print('group:', group['name'])
    exps = make_experiments(group, args.dataset_seed)
    enumber = 1
    print('Num exps:', len(exps))
    # exps = tqdm(make_experiments(group, args.dataset_seed))
    two_sample = group['two_sample']
    for (sname, s), (tname, t), (hname, h) in exps:
        if args.test: # don't train to long
            h = lazy_kwarg_init(h, epochs=1, 
                features_epochs=1)
        s = s(); t = t()
        if len(s) < MIN_TRAIN_SAMPLES:
            continue
        res = run_experiment(s, t, h, args.experiment_seed, 
            two_sample, verbose=args.verbose, 
            device=args.device)
        res['source'] = sname
        res['target'] = tname
        res['dataset_seed'] = args.dataset_seed
        res['experiment_seed'] = args.experiment_seed
        res['group_num'] = group['name']
        res['hspace'] = hname
        data.append(res)
        # incrementally save results
        g = group['name']
        ds = args.dataset_seed
        es = args.experiment_seed
        Path('out/results').mkdir(parents=True, 
            exist_ok=True)
        write_loc = f'out/results/{g}-{ds}-{es}.csv'
        pd.DataFrame(data).to_csv(write_loc)
        print('Done', enumber); enumber += 1