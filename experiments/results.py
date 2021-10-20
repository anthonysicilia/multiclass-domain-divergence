import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
import numpy as np
import os
import pandas as pd

from pathlib import Path
from scipy.stats import spearmanr, pearsonr, linregress

CV = 'out/results'
NLP = 'out/results-nlp-10-5-2021'
PATHS = [f'{CV}/{x}' for x in os.listdir(CV)]
PATHS += [f'{NLP}/{x}' for x in os.listdir(NLP)]
CV_LB = 'out/lambda_baseline_results'
NLP_LB = 'out/lambda_baseline_results-nlp-10-11-2021'
PATHS_LB = [f'{CV_LB}/{x}' for x in os.listdir(CV_LB)]
PATHS_LB += [f'{NLP_LB}/{x}' for x in os.listdir(NLP_LB)]
CV_S = 'out/stoch_results'
NLP_S = 'out/stoch_results-nlp-10-8-2021'
PATHS_S = [f'{CV_S}/{x}' for x in os.listdir(CV_S)]
PATHS_S += [f'{NLP_S}/{x}' for x in os.listdir(NLP_S)]
CV_R = 'out/baseline_results'
PATHS_R = [f'{CV_R}/{x}' for x in os.listdir(CV_R)]

SEP = '$$$$'
MCS = 100

def get_holdout_lambda(df):
    x = df['holdout_source_error']
    x = x + df['holdout_target_error']
    x = x + np.sqrt(np.log(4 / 0.05) / (2 * df['source_samples']))
    x = x + np.sqrt(np.log(4 / 0.05) / (2 * df['target_samples']))
    return x

def get_germain_lambda(df):
    x = df['source_jerror']
    x = (x - df['target_jerror']).abs()
    # because we use the same H for both, and diff is bounded in range 1, - 1
    return x + np.sqrt(2 * np.log(2 / 0.05) / MCS)

def get_dis_div(df):
    return (df['source_dis'] - df['target_dis']).abs()

def get_rho(df):
    a = df['train_error_ref']
    a = (a - df['train_error']).abs()
    b = df['transfer_error_ref']
    b = (b - df['transfer_error']).abs()
    return a + b

def get_neg_h_class_div(df):
    return -df['h_class_divergence']

def get_neg_h_div(df):
    return -df['our_h_divergence']

def make_df(paths, mcol):
    # concats paths and grabs min of experiment seeds
    # according to some (function of) column(s)
    df = pd.concat([pd.read_csv(p) for p in paths])
    if type(mcol) == type(lambda x: x):
        _mcol = SEP + '__x__' + SEP
        df[_mcol] = mcol(df)
        mcol = _mcol
    elif mcol is None:
        return df
    df['uid'] = df['source'] + SEP \
        + df['target'] + SEP \
        + df['dataset_seed'].astype(str) + SEP \
        + df['group_num'].astype(str) + SEP \
        + df['hspace']
    smallest = dict()
    exp = dict()
    for u, x, e in zip(df['uid'], df[mcol], df['experiment_seed']):
        if u not in smallest:
            smallest[u] = x
            exp[u] = str(e)
        elif x < smallest[u]:
            smallest[u] = x
            exp[u] = str(e)
    df['uid+e1'] = df['uid'].map(lambda u: u + SEP + exp[u])
    df['uid+e2'] = df['uid'] + SEP \
        + df['experiment_seed'].astype(str)
    idx = (df['uid+e1'] == df['uid+e2'])
    return df[idx].copy()

def adaptability(rf, bf, sf):
    mpl.rcParams['font.size'] = 18
    rf['sample-dependent'] = rf['ben_david_lambda']
    bf['sample-independent'] = get_holdout_lambda(bf)
    sf['Germain et al.'] = get_germain_lambda(sf)
    m = rf['sample-dependent'].min()
    M = rf['sample-dependent'].max()
    m = min(bf['sample-independent'].min(), m)
    m = min(sf['Germain et al.'].min(), m)
    M = max(bf['sample-independent'].max(), M)
    M = max(sf['Germain et al.'].max(), M)
    fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True,
        sharey=True)
    ax.flat[0].hist(rf['sample-dependent'], 
        label='Ours', range=(m,M),
        bins=10, alpha=1, color='b')
    ax.flat[0].legend()
    ax.flat[0].set_ylabel('Count')
    ax.flat[0].set_xlabel('Upperbound')
    ax.flat[1].hist(bf['sample-independent'], 
        label='Ben-David et al.', range=(m,M),
        bins=10, alpha=1, color='r')
    ax.flat[1].set_xlabel('Upperbound')
    ax.flat[1].legend()
    ax.flat[2].hist(sf['Germain et al.'], 
        label='Germain et al.', range=(m,M),
        bins=10, alpha=1, color='g')
    ax.flat[2].set_xlabel('Upperbound')
    ax.flat[2].legend()
    plt.tight_layout()
    plt.savefig('results/adaptability')

def rho(sf):
    mpl.rcParams['font.size'] = 15
    sf['rho'] = get_rho(sf)
    print('Mean rho:', sf['rho'].mean())
    print('Std rho:', sf['rho'].std())
    fig,ax = plt.subplots(figsize=(6,2))
    plt.hist(sf['rho'], label='rho',
        bins=30, alpha=1, color='b')
    x = sf[sf['rho'] > 0.04]['rho'].values
    plt.plot(x, np.zeros(x.shape), 'b+', ms=20)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.legend()
    plt.savefig('results/rho')

def spearman_rank_correlation(df, col):
    df['error_gap'] = df['transfer_error'] - df['train_error']
    x = df['error_gap'].abs()
    return spearmanr(x, df[col]).correlation

def error(rf):
    mpl.rcParams['font.size'] = 15
    x = rf['transfer_error'] - rf['train_error']
    rf['Delta'] = x.abs()
    e1 = rf['Delta'] - rf['ben_david_lambda'] \
        - rf['our_h_divergence']
    e2 = rf['Delta'] - rf['ben_david_lambda'] \
        - rf['h_class_divergence']
    m, M = min(e1.min(), e2.min()), max(e1.max(), e2.max())
    fig, ax = plt.subplots(1, 2, figsize=(8,2.75), sharex=True,sharey=True)
    ax.flat[0].hist(e1, label='model-dependent', range=(m,M),
        bins=10, alpha=1, color='b')
    ax.flat[0].axvline(x=0, color='k', linestyle='--', lw=2)
    ax.flat[0].set_ylim((0, 2500))
    ax.flat[0].legend()
    ax.flat[1].hist(e2, label='model-independent', range=(m,M),
        bins=10, alpha=1, color='r')
    ax.flat[1].axvline(x=0, color='k', linestyle='--', lw=2)
    ax.flat[1].legend()
    ax.flat[0].set_ylabel('Count')
    ax.flat[0].set_xlabel('Error Lowerbound')
    ax.flat[1].set_xlabel('Error Lowerbound')
    plt.tight_layout()
    plt.savefig('results/error')

def hue_lambda(rf):
    import seaborn as sns; sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,6))
    rf['upperbound'] = rf['ben_david_lambda']
    rf['adaptation scenario'] = rf['group_num']
    sns.boxplot(x='upperbound', y='adaptation scenario', 
        data=rf, ax=ax)
    plt.tight_layout()
    plt.savefig('results/hued_groups_lambda')

def hue_error_gap(df, rf):
    import seaborn as sns; sns.set(style='whitegrid')
    fig, ax = plt.subplots(figsize=(10,12))
    fn = lambda s: s.split('-')[0]
    within = (df['source'].map(fn) == df['target'].map(fn))
    df['hue'] = within.map(lambda t: 'OOD' if not t else 'WD')
    within = (rf['source'].map(fn) == rf['target'].map(fn))
    rf['hue'] = within.map(lambda t: 'OOD-r' if not t else 'WD-r')
    df['adaptation scenario'] = df['group_num']
    rf['adaptation scenario'] = rf['group_num']
    df['transfer error'] = df['transfer_error']
    rf['transfer error'] = rf['random_transfer_error']
    cols = ['adaptation scenario', 'transfer error', 'hue']
    rf = pd.concat([df[cols], rf[cols]])
    sns.boxplot(x='transfer error', y='adaptation scenario', 
        hue='hue', data=rf, ax=ax)
    ax.set_xlim((0,1))
    plt.tight_layout()
    plt.savefig('results/hued_groups_transfer_error')


if __name__ == '__main__':
    rf = make_df(PATHS, 'ben_david_lambda')
    bf = make_df(PATHS_LB, get_holdout_lambda)
    sf = make_df(PATHS_S, get_germain_lambda)
    adaptability(rf, bf, sf)
    sf = make_df(PATHS_S, None)
    rho(sf)
    rf = make_df(PATHS, None)
    error(rf)
    c = spearman_rank_correlation(rf, 'h_class_divergence')
    print('H Div Corr: ', c)
    c = spearman_rank_correlation(rf, 'our_h_divergence')
    print('h Div Corr: ', c)
    sf = make_df(PATHS_S, None)
    sf['germain_dis_div'] = get_dis_div(sf)
    c = spearman_rank_correlation(sf, 'germain_dis_div')
    print('Germain Corr: ', c)
    # do last to avoid style conflicts
    rf = make_df(PATHS, 'ben_david_lambda')
    hue_lambda(rf)
    df = make_df(PATHS, None)
    rf = make_df(PATHS_R, None)
    hue_error_gap(df, rf)

