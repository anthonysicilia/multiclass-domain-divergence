from .run import GROUPS
import matplotlib.pyplot as plt

if __name__ == '__main__':

    df = pd.read_csv('results-all.csv')
    same = df['source'].apply(lambda s: s.split('-')[0]) \
        == df['target'].apply(lambda s: s.split('-')[0])
    df[same]['transfer_error']