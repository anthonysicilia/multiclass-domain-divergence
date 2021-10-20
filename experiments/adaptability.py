import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as mpl
mpl.rcParams['font.size'] = 16

# sns.set(style='white', font_scale=1.5)

PATH1 = 'out/results'
PATH2 = 'out/results-nlp-10-5-2021'
PATH3 = 'out/lambda_baseline_results'
PATH4 = 'out/stoch_results'
PATH5 = 'out/stoch_results-nlp-10-8-2021'

PATHS = [f'{PATH1}/{x}' for x in os.listdir(PATH1)]
PATHS += [f'{PATH2}/{x}' for x in os.listdir(PATH2)]
B_PATHS = [f'{PATH3}/{x}' for x in os.listdir(PATH3)]
S_PATHS = [f'{PATH4}/{x}' for x in os.listdir(PATH4)]
S_PATHS +=[f'{PATH5}/{x}' for x in os.listdir(PATH5)]

if __name__ == '__main__':
    results = pd.concat([pd.read_csv(p) for p in PATHS])
    baseline = pd.concat([pd.read_csv(p) for p in B_PATHS])
    stochastic = pd.concat([pd.read_csv(p) for p in S_PATHS])

    results['sample-dependent'] = results['ben_david_lambda']
    x = baseline['holdout_source_error']
    x = x + baseline['holdout_target_error']
    x = x + np.sqrt(np.log(2 / 0.05) / (2 * baseline['source_samples']))
    x = x + np.sqrt(np.log(2 / 0.05) / (2 * baseline['target_samples']))
    baseline['sample-independent'] = x
    x = stochastic['source_jerror']
    x = (x - stochastic['target_jerror']).abs()
    stochastic['Germain et al.'] = x
    m, M = results['sample-dependent'].min(), results['sample-dependent'].max()
    m = min(baseline['sample-independent'].min(), m)
    m = min(stochastic['Germain et al.'].min(), m)
    M = max(baseline['sample-independent'].max(), M)
    M = max(stochastic['Germain et al.'].max(), M)
    fig, ax = plt.subplots(1, 3, figsize=(12,4), sharex=True,
        sharey=True)
    ax.flat[0].hist(results['sample-dependent'], 
        label='Ours', range=(m,M),
        bins=10, alpha=1, color='b')
    ax.flat[0].legend()
    ax.flat[1].hist(baseline['sample-independent'], 
        label='Ben-David et al.', range=(m,M),
        bins=10, alpha=1, color='r')
    ax.flat[1].legend()
    ax.flat[2].hist(stochastic['Germain et al.'], 
        label='Germain et al.', range=(m,M),
        bins=10, alpha=1, color='g')
    ax.flat[2].legend()
    plt.tight_layout()
    plt.savefig('results/adaptability')


