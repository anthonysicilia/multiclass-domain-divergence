import numpy as np
import pickle

def process_line(line):
    line = line.split(' ')[:-1]
    features = []
    for x in line:
        x = x.split(':')
        if len(x) > 2:
            print('Unabel to handle:', x)
        x = (x[0], int(x[1]))
        features.append(x)
    return features

if __name__ == '__main__':

    # See https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

    domains = ['books', 'dvd', 'electronics', 'kitchen']
    domains = [f'processed_acl/{d}' for d in domains]
    data = {d : [] for d in domains}

    for d in domains:
        for label, name in [(0, 'negative'), (1, 'positive')]:
            lines = open(f'{d}/{name}.review', 'r').readlines()
            for line in lines:
                features = process_line(line)
                data[d].append((features, label))
    
    counts = dict()
    for d in domains:
        for review in data[d]:
            for word, count in review[0]:
                if word not in counts:
                    counts[word] = 0
                counts[word] += count
    n = 4095
    vocab = [word for word, _ in sorted(counts.items(),
        key=lambda x: -x[1])][:4095]
    with open('processed_acl/vocab.txt', 'w') as out:
        for word in vocab:
            out.write(word + '\n')
    vocab = {word : i for i,word in enumerate(vocab)}

    for d in data:
        for i, (features, label) in enumerate(data[d]):
            x = np.zeros(n + 1)
            for word, count in features:
                idx = vocab[word] if word in vocab else n
                x[idx] += count
            data[d][i] = (x, label)
    
    data = {d.split('/')[-1] : arr for d,arr in data.items()}
    pickle.dump(data, open('processed_acl/data.pkl', 'wb'))
