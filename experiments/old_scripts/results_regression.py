import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns; sns.set(style='whitegrid')

from scipy.stats import boxcox

if __name__ == '__main__':

    df = pd.read_csv('results-all-0.csv')
    df['error_gap'] = df['transfer_error'] - df['train_error']
    df['Delta'] = df['error_gap'].abs()
    df['Delta_tilde'] = boxcox(df['Delta'] + 1e-5)[0]
    model = smf.wls('Delta_tilde ~ our_h_divergence', data=df).fit()
    df['residuals'] = model.resid
    sns.scatterplot(x='our_h_divergence', y='residuals', data=df)
    plt.savefig('results-0/residuals')
    print(model.summary())