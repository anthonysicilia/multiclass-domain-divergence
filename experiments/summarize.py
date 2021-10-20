import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns; sns.set(style='whitegrid')

from pathlib import Path
from scipy.stats import spearmanr, pearsonr, linregress

PATH1 = 'out/results'
PATH2 = 'out/stoch_results'
PATH4 = 'out/results-nlp-10-5-2021'
PATH5 = 'out/stoch_results-nlp-10-8-2021'

PATHS = [f'{PATH1}/{x}' for x in os.listdir(PATH1)]
PATHS += [f'{PATH4}/{x}' for x in os.listdir(PATH4)]
STOCH_PATHS = [f'{PATH2}/{x}' for x in os.listdir(PATH2)]
STOCH_PATHS += [f'{PATH5}/{x}' for x in os.listdir(PATH5)]

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
    s_results['germain_dis_div'] = (s_results['source_dis'] 
        - s_results['target_dis']).abs()
    summarize(s_results.copy(), 'results/agg',
        stat_columns=('our_h_divergence_ref', 'germain_dis_div'),
        app='-s')
    



            



