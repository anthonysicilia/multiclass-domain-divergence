import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns; sns.set(style='whitegrid')

from pathlib import Path
from scipy.stats import spearmanr, pearsonr, linregress

PATH1 = 'out/results'
PATH2 = 'out/stoch_results'
PATH3 = 'out/baseline_results'

PATHS = [f'{PATH1}/{x}' for x in os.listdir(PATH1)]
STOCH_PATHS = [f'{PATH2}/{x}' for x in os.listdir(PATH2)]
BAS_PATHS = [f'{PATH3}/{x}' for x in os.listdir(PATH3)]

METRICS = {
    'pearsonr' : lambda x, y: pearsonr(x, y)[0],
    'spearmanr' : lambda x, y: spearmanr(x, y).correlation
}

def zscore(x, col=None):
    return (x - x.mean()) / x.std()

def zscore_by_col(df, stat, col):
    mu = df.groupby(col).mean().to_dict()[stat]
    std = df.groupby(col).std().to_dict()[stat]
    return (df[stat] - df[col].map(mu)) / df[col].map(std)

def summarize(df, write_loc, norm=False, norm_col=None, app='',
    stat_columns=('our_h_divergence', 'h_class_divergence')):
    Path(write_loc).mkdir(parents=True, exist_ok=True)
    data = []
    df['error_gap'] = df['transfer_error'] - df['train_error']
    df['abs_error_gap'] = df['error_gap'].abs()
    if norm:
        for stat in stat_columns:
            df[stat] = zscore_by_col(df, stat, norm_col) \
                if norm_col is not None else zscore(df[stat])
    for stat in stat_columns:
        agg_results = {'stat' : stat}
        for y in ['abs_error_gap']:
            # assumes seaborn uses linregress to make plot
            # this is true as of version ??
            slope, intercept, r_val, p_val, *_ = \
                linregress(df[stat], df[y])
            lb = f'y = {slope:.2f} x + {intercept:.2f}'
            lb += f' | p={p_val:.4f} | r2={r_val ** 2:.4f}'
            sns.regplot(x=stat, y=y, data=df, 
                line_kws={'label' : lb})
            plt.legend()
            plt.savefig(f'{write_loc}/{stat}_{y}')
            plt.clf()
            for k, m in METRICS.items():
                agg_results[f'{y}_{k}'] = m(df[y], df[stat])
        data.append(agg_results)
    pd.DataFrame(data).to_csv(f'{write_loc}/results{app}.csv')

if __name__ == '__main__':
    results = pd.concat([pd.read_csv(p) for p in PATHS])
    results.to_csv('results-all.csv')
    summarize(results.copy(), 'results/agg')
    s_results = pd.concat([pd.read_csv(p) for p in STOCH_PATHS])
    s_results.to_csv('results-stoch.csv')
    s_results['Germain et al.'] = s_results['source_jerror'] \
        - s_results['target_jerror']
    s_results['Germain et al.'] = s_results['Germain et al.'].abs()
    a = (s_results['train_error_ref'] - s_results['train_error']).abs()
    b = (s_results['transfer_error_ref'] - s_results['transfer_error']).abs()
    s_results['rho'] = a + b
    x = [results['ben_david_lambda'], s_results['Germain et al.'], 
        s_results['rho']]
    plt.hist(x, label=['lambda (Ben-David et al.)', 
        'lambda (Germain et al.)', 'rho'],
        bins=25, alpha=0.8)
    plt.legend()
    plt.savefig('results/stoch_assumptions-1')
    plt.clf()
    s_results['germain_dis_div'] = (s_results['source_dis'] 
        - s_results['target_dis']).abs()
    summarize(s_results.copy(), 'results/agg',
        stat_columns=('our_h_divergence_ref', 'germain_dis_div'),
        app='-s')
    b_results = pd.concat([pd.read_csv(p) for p in BAS_PATHS])
    b_results.to_csv('results-basl.csv')
    b_results['train_error'] = b_results['random_train_error']
    b_results['transfer_error'] = b_results['random_transfer_error']
    summarize(b_results.copy(), 'results/agg',
        stat_columns=('random_h_divergence', ), 
        app='-b')
    b_results['uid'] = b_results['source'] + b_results['target'] \
        + b_results['dataset_seed'].astype(str) \
        + b_results['experiment_seed'].astype(str) \
        + b_results['hspace'].astype(str)
    results['uid'] = results['source'] + results['target'] \
        + results['dataset_seed'].astype(str) \
        + results['experiment_seed'].astype(str) \
        + results['hspace'].astype(str)
    merged = {k : {
        'baseline_h_divergence' : a,
        'random_h_divergence' : b } 
        for k, a, b in zip(b_results['uid'], 
        b_results['baseline_h_divergence'],
        b_results['random_h_divergence'])
    }
    tups = zip(results['uid'], results['transfer_error'], 
        results['train_error'], results['our_h_divergence'])
    for k, a, b, c in tups:
        if k in merged:
            merged[k]['transfer_error'] = a
            merged[k]['train_error'] = b
            merged[k]['our_h_divergence'] = c
    data = {'baseline_h_divergence' : [], 
        'random_h_divergence' : [],
        'train_error' : [],
        'transfer_error' : [],
        'our_h_divergence' : []}
    for v in merged.values():
        for k,vi in v.items():
            data[k].append(vi)
    merged = pd.DataFrame(data)
    summarize(merged.copy(), 'results/agg',
        stat_columns=('baseline_h_divergence', ), 
        app='-m')
    x = (merged['our_h_divergence'] 
        - merged['random_h_divergence']).abs()
    plt.hist(x, label='abs diff', bins=25, alpha=0.8)
    plt.legend()
    plt.savefig('results/random_h_div_comp')
    plt.clf()
    results['error_gap'] = results['transfer_error'] \
        - results['train_error']
    results['abs_error_gap'] = results['error_gap'].abs()
    a = results['abs_error_gap'] - results['ben_david_lambda'] \
        - results['our_h_divergence']
    a = [ai if ai > 0 else 0 for ai in a]
    b = results['abs_error_gap'] - results['ben_david_lambda'] \
        - results['h_class_divergence']
    a = [ai for ai in a if ai > 0]
    b = [bi for bi in b if bi > 0]
    plt.hist([a,b], 
        label=['our_h_divergence', 'h_class_divergence'], 
        bins=25, alpha=0.8)
    plt.legend()
    plt.savefig('results/approx_error_comp')
    plt.clf()
    



            



