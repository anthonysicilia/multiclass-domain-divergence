import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns; sns.set(style='whitegrid')

if __name__ == '__main__':
    df = pd.read_csv('results-all-0.csv')
    df['error_gap'] = df['transfer_error'] - df['train_error']
    df['Delta'] = df['error_gap'].abs()
    sns.scatterplot(x='our_h_divergence', y='Delta', data=df, hue='group_num')
    plt.savefig('results-0/hued_groups')