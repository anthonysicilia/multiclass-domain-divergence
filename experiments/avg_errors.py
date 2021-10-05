import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('results-all-0.csv')
    for g in df['group_num'].unique():
        for h in df['hspace'].unique():
            idx = df['group_num'] == g
            idx = idx & (df['hspace'] == h)
            if idx.sum() == 0:
                continue
            x = df[idx].copy()
            train_error = x['train_error'].mean()
            test_error = x['transfer_error'].mean()
            print(f'{g} {h}: train error: {train_error:.4f} transfer error: {test_error:.4f}')