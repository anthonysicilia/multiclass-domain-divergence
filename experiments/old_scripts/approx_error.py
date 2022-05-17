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

PATHS = [f'{PATH1}/{x}' for x in os.listdir(PATH1)]
PATHS += [f'{PATH2}/{x}' for x in os.listdir(PATH2)]

if __name__ == '__main__':
    results = pd.concat([pd.read_csv(p) for p in PATHS])
    x = results['transfer_error'] - results['train_error']
    results['Delta'] = x.abs()
    e1 = results['Delta'] - results['ben_david_lambda'] \
        - results['our_h_divergence']
    e2 = results['Delta'] - results['ben_david_lambda'] \
        - results['h_class_divergence']
    m, M = min(e1.min(), e2.min()), max(e1.max(), e2.max())
    fig, ax = plt.subplots(1, 2, figsize=(8,3), sharex=True,sharey=True)
    ax.flat[0].hist(e1, label='model-dependent', range=(m,M),
        bins=10, alpha=1, color='b')
    ax.flat[0].legend()
    ax.flat[1].hist(e2, label='model-independent', range=(m,M),
        bins=10, alpha=1, color='r')
    ax.flat[1].legend()
    plt.tight_layout()
    plt.savefig('results/approx_error')