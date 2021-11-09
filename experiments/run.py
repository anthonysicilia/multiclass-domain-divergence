import argparse
import numpy as np
import random
import pandas as pd
import torch

from pathlib import Path
# from tqdm import tqdm

from .datasets.amazon import DATASETS as AMAZON_DATASETS
from .datasets.digits import DATASETS as DIGITS_DATASETS, \
    ROTATION_PAIRS, NOISY_PAIRS, FAKE_PAIRS
from .datasets.discourse import PDTB_DATASETS, GUM_DATASETS, RST_DATASETS, RST_GUM_PDTB_LABELS_DATASETS
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
from .models.stochastic import Stochastic, mean
from .utils import PlaceHolderEstimator, lazy_kwarg_init, set_random_seed

MIN_TRAIN_SAMPLES = 1000
MC_SAMPLES = 100
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

    'digits_m' : {
        'name' : 'digits_m',
        'datasets' : DIGITS_DATASETS,
        'multisource' : True,
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

    'officehome_fts_m' : {
        'name' : 'officehome_fts_m',
        'datasets' : OFFICEHOME_FTS_DATASETS,
        'multisource' : True,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [
            ('lin', lazy_kwarg_init(LinearHypothesisSpace, 
                num_inputs=2048)),
            ('fc4l', lazy_kwarg_init(NonLinearHypothesisSpace, 
                num_inputs=2048))]
    },

    'pacs_fts_m' : {
        'name' : 'pacs_fts_m',
        'datasets' : PACS_FTS_DATASETS,
        'multisource' : True,
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
    },

    'rst_sentence' : {
        'name' : 'rst_sentence',
        'datasets' : RST_DATASETS('sentence'),
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'rst_average' : {
        'name' : 'rst_average',
        'datasets' : RST_DATASETS('average'),
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'rst_pooled' : {
        'name' : 'rst_pooled',
        'datasets' : RST_DATASETS('pooled'),
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'rst_gum_pdtb_sentence' : {
        'datasets_a' : RST_GUM_PDTB_LABELS_DATASETS('sentence'),
        'datasets_b' : PDTB_DATASETS('sentence'),
        'multisource' : True,
        'two_groups' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'rst_gum_pdtb_average' : {
        'datasets_a' : RST_GUM_PDTB_LABELS_DATASETS('average'),
        'datasets_b' : PDTB_DATASETS('average'),
        'multisource' : True,
        'two_groups' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'rst_gum_pdtb_pooled' : {
        'datasets_a' : RST_GUM_PDTB_LABELS_DATASETS('pooled'),
        'datasets_b' : PDTB_DATASETS('pooled'),
        'multisource' : True,
        'two_groups' : True,
        'two_sample' : True,
        'models' : [('lin', LinearHypothesisSpace),
            ('fc4l', NonLinearHypothesisSpace)],
    },

    'amazon' : {
        'name' : 'amazon',
        'datasets' : AMAZON_DATASETS,
        'multisource' : False,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [
            ('lin', lazy_kwarg_init(LinearHypothesisSpace, 
                num_inputs=4096)),
            ('fc4l', lazy_kwarg_init(NonLinearHypothesisSpace, 
                num_inputs=4096))]
    },

    'amazon_m' : {
        'name' : 'amazon_m',
        'datasets' : AMAZON_DATASETS,
        'multisource' : True,
        'make_pairs' : True,
        'two_sample' : True,
        'models' : [
            ('lin', lazy_kwarg_init(LinearHypothesisSpace, 
                num_inputs=4096)),
            ('fc4l', lazy_kwarg_init(NonLinearHypothesisSpace, 
                num_inputs=4096))]
    },

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
            mdset = lazy_kwarg_init(Multisource, dsets=dsets)
            source = ('+'.join(descs), mdset)
            exps.append((source, target, hspace))
    return exps

def _make_multi_group_exps(group_a, group_b, dataset_seed):
    datasets_a = [(f'{desc}', lazy_kwarg_init(dset,
                                              train=True, seed=dataset_seed))
        for desc, dset in group_a]
    datasets_b = [(f'{desc}',
        dset(train=True, seed=dataset_seed))
        for desc, dset in group_b]
    exps = []
    for i in range(len(datasets_a)):
        for hspace in group['models']:
            target = datasets_a[i]
            descs = [s[0] for s in datasets_b]
            dsets = [s[1] for s in datasets_b]
            mdset = lazy_kwarg_init(Multisource, dsets=dsets)
            source = ('+'.join(descs), mdset)
            exps.append((source, target, hspace))
    for i in range(len(datasets_b)):
        for hspace in group['models']:
            target = datasets_b[i]
            descs = [s[0] for s in datasets_a]
            dsets = [s[1] for s in datasets_a]
            mdset = lazy_kwarg_init(Multisource, dsets=dsets)
            source = ('+'.join(descs), mdset)
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
    if group['two_groups']:
        return _make_multi_group_exps(group['datasets_a'],
                                      group['datasets_b'], dataset_seed)
    else:
        if group['multisource']:
            return _make_multisource_exps(group['datasets'],
                                          dataset_seed)
        else:
            return _make_single_source_exps(group['datasets'],
                                            dataset_seed)
def disjoint_split(dataset, seed=0, prefix_ratio=0.5):

    random.seed(seed)
    indices = [(i, random.random() <= prefix_ratio) 
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
            # can't assing to tuple... 
            # NOTE: this might break cacheing
            # data[2] = index
            return data
    
    return Dataset(dataset, prefix), Dataset(dataset, bound)

def run_lambda_baseline_experiment(source, target, hspace, seed,
    verbose=False, device='cpu'):

    target1, target2 = disjoint_split(target, seed=seed,
        prefix_ratio=0.8)
    
    source1, source2 = disjoint_split(source, seed=seed,
        prefix_ratio=0.8)

    hypothesis_space = hspace(num_classes=source.num_classes)

    hypothesis = BenDavidLambdaEstimator(
        hypothesis_space, source1, target1, 
        return_h=True,
        device=device, verbose=verbose).compute()[1]

    estimators = {
        'holdout_source_error' : Error(hypothesis, hypothesis_space,
            source2, verbose=verbose, device=device),
        'holdout_target_error' : Error(hypothesis, hypothesis_space,
            target2, verbose=verbose, device=device)
    }

    estimates = {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

    estimates['source_samples'] = len(source2)
    estimates['target_samples'] = len(target2)

    return estimates

def run_baseline_experiment(source, target, hspace, seed,
    verbose=False, device='cpu'):

    hypothesis_space = hspace(num_classes=source.num_classes)
    random_h = hypothesis_space()

    estimators = {

        # 'baseline_h_divergence' : HypothesisClassDivergence(
        #     hypothesis_space, source, target, verbose=verbose,
        #     binary=False, device=device, baseline=True),

        # 'random_h_divergence' : HypothesisDivergence(random_h, 
        #     hypothesis_space, source, target, verbose=verbose,
        #     device=device),
        
        'random_train_error' : Error(random_h, hypothesis_space,
            source, verbose=verbose, device=device),

        'random_transfer_error' : Error(random_h, hypothesis_space, 
            target, verbose=verbose, device=device),
        
    }

    return {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}

def run_stoch_experiment(source, target, hspace, seed,
    verbose=False, device='cpu'):

    hypothesis_space = hspace(num_classes=source.num_classes)
    prior_estimator = HypothesisEstimator(hypothesis_space, 
        source, verbose=verbose, device=device)
    prior = SeededEstimator(prior_estimator, seed).compute()
    stoch_hypothesis_space = Stochastic(hypothesis_space, 
        prior=prior, m=len(source), delta=0.05,
        sigma_prior=SIGMA_PRIOR, device=device)
    posterior_estimator = HypothesisEstimator(
        stoch_hypothesis_space, source, kl_reg=True, 
        sample=True, cache=False, verbose=verbose, 
        device=device)
    posterior = SeededEstimator(posterior_estimator, 
        seed).compute()

    estimators = {
        'train_error' : lazy_kwarg_init(Error,
            hypothesis=posterior, 
            hypothesis_space=hypothesis_space, sample=True,
            dataset=source, verbose=verbose, device=device),

        'transfer_error' : lazy_kwarg_init(Error, 
            hypothesis=posterior, 
            hypothesis_space=hypothesis_space, sample=True,
            dataset=target, verbose=verbose, device=device),
        
        'source_dis' : lazy_kwarg_init(StochDisagreementEstimator,
            hypothesis=posterior, 
            hypothesis_space=stoch_hypothesis_space, 
            dataset=source, 
            device=device, verbose=verbose, sample=True),
        
        'source_jerror' : lazy_kwarg_init(StochJointErrorEstimator,
            hypothesis=posterior, 
            hypothesis_space=stoch_hypothesis_space, 
            dataset=source, 
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
        
    }

    estimators = {k : MonteCarloEstimator(MC_SAMPLES, e) 
        for k, e in estimators.items()}

    computed = {k : SeededEstimator(e, seed).compute() 
        for k, e in estimators.items()}
    
    with torch.no_grad():
        posterior = mean(posterior)
    
    estimators = {
        'train_error_ref' : Error(posterior,
            hypothesis_space, source, sample=False,
            verbose=verbose, device=device),

        'transfer_error_ref' : Error(posterior,
            hypothesis_space, target, sample=False,
            verbose=verbose, device=device),

        'our_h_divergence_ref' : HypothesisDivergence(
            posterior, hypothesis_space, 
            source, target, verbose=verbose, 
            device=device)
    }
    
    for k, est in estimators.items():
        computed[k] = SeededEstimator(est, seed).compute()
    
    return computed

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
    parser.add_argument('--stochastic',
        action='store_true',
        help='Run stochastic experiments.')
    parser.add_argument('--baseline',
        action='store_true',
        help='Run baseline experiments.')
    parser.add_argument('--lambda_baseline',
        action='store_true',
        help='Run lambda baseline experiments.')
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
            MC_SAMPLES = 1
        s = s(); t = t()
        if len(s) < MIN_TRAIN_SAMPLES:
            continue
        if args.stochastic:
            res = run_stoch_experiment(s, t, h, 
                args.experiment_seed, verbose=args.verbose, 
                device=args.device)
        elif args.baseline:
            res = run_baseline_experiment(s, t, h, 
                args.experiment_seed, verbose=args.verbose, 
                device=args.device)
        elif args.lambda_baseline:
            res = run_lambda_baseline_experiment(s, t, h, 
                args.experiment_seed, verbose=args.verbose, 
                device=args.device)
        else:
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
        if args.stochastic:
            dir_loc = 'out/stoch_results'
        elif args.baseline:
            dir_loc = 'out/baseline_results'
        elif args.lambda_baseline:
            dir_loc = 'out/lambda_baseline_results'
        else:
            dir_loc = 'out/results'
        Path(dir_loc).mkdir(parents=True, exist_ok=True)
        write_loc = f'{dir_loc}/{g}-{ds}-{es}.csv'
        pd.DataFrame(data).to_csv(write_loc)
        print('Done', enumber); enumber += 1
