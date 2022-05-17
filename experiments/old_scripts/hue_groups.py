import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns; sns.set(style='whitegrid')

if __name__ == '__main__':
    df = pd.read_csv('results-all.csv')
    df['error_gap'] = df['transfer_error'] - df['train_error']
    df['Delta'] = df['error_gap'].abs()
    # drop amazon casue we couldn't figure out how to train
    # x = (df['group_num'] != 'amazon') & (df['group_num'] != 'amazon_m')
    # df = df[x].copy()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x='ben_david_lambda', y='group_num', 
        data=df, ax=ax)
    plt.tight_layout()
    plt.savefig('results/hued_groups_lambda')