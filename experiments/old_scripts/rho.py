import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

import matplotlib.pyplot as mpl
mpl.rcParams['font.size'] = 14

# sns.set(style='white', font_scale=1.5)

PATH1 = 'out/stoch_results'
PATH2 = 'out/stoch_results-nlp-10-8-2021'

PATHS = [f'{PATH1}/{x}' for x in os.listdir(PATH1)]
PATHS += [f'{PATH2}/{x}' for x in os.listdir(PATH2)]

if __name__ == '__main__':
    results = pd.concat([pd.read_csv(p) for p in PATHS])
    a = results['train_error_ref']
    a = (a - results['train_error']).abs()
    b = results['transfer_error_ref']
    b = (b - results['transfer_error']).abs()
    results['rho'] = a + b
    print('Mean:', results['rho'].mean())
    print('Std:', results['rho'].std())
    fig,ax = plt.subplots(figsize=(6,2))
    plt.hist(results['rho'], 
        label='rho',
        bins=30, alpha=1, color='b')
    # sns.boxplot(x='rho', data=results)
    x = results[results['rho'] > 0.04]['rho'].values

    plt.plot(x, np.zeros(x.shape), 'b+', ms=20)
    plt.tight_layout()
    plt.legend()
    plt.savefig('results/rho')
