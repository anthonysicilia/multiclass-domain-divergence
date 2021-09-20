import pickle
from random import Random
import torch

from ..utils import lazy_kwarg_init

# TOKEN quantiles for pdtb, biordb, ted
# 99.9% | 221.9719999999943
# 99%   | 133.0
# 98%   | 115.0
# 98.7% | 126.0
# 98.8% | 129.0

BERT_VECTOR_TYPES = [
    'sentence',
    'average',
    'pooled',
    'tokens'
]

NORMED_DISCOURSE_SENSES = {
    'Comparison' : 'Comparison',
    'Contingency' : 'Contingency',
    'Temporal' : 'Temporal',
    'Expansion' : 'Expansion',
    'Concession' : 'Comparison',
    'Contrast' : 'Comparison',
    'Cause' : 'Contingency',
    'Condition' : 'Contingency',
    'Purpose' : 'Contingency',
    'Temporal' : 'Temporal',
    'Alternative' : 'Expansion',
    'Background' : 'Expansion',
    'Circumstance' : 'Expansion',
    'Conjunction' : 'Expansion',
    'Continuation' : 'Expansion',
    'Exception' : 'Expansion',
    'Instantiation' : 'Expansion',
    'Reinforcement' : 'Expansion',
    'Restatement' : 'Expansion',
    'Similarity' : 'Expansion'
}

PDTB_DISCOURSE_SENSE = {
    'Comparison' : 0,
    'Contingency' : 1,
    'Temporal' : 2,
    'Expansion' : 3,
}

GUM_DISCOURSE_SENSE = {
    'Topic-Comment' : 0, 
    'Joint' : 1, 
    'Contrast' : 2, 
    'Background' : 3, 
    'Attribution' : 4, 
    'Temporal' : 5, 
    'Explanation' : 6, 
    'Summary' : 7, 
    'Cause' : 8, 
    'Condition' : 9, 
    'Elaboration' : 10, 
    'Manner-Means' : 11, 
    'Enablement' : 12, 
    'Evaluation' : 13
}

DISCOURSE_SENSE = {
    'pdtb' : PDTB_DISCOURSE_SENSE,
    'gum' : GUM_DISCOURSE_SENSE
}

class Dataset(torch.utils.data.Dataset):

        def __init__(self, vecs, labels):
            self.vecs = vecs
            self.labels = labels
            self.num_classes = max(set(labels)) + 1
            # not really needed anymore, always 768 or tokens
            # try:
            #     self.input_sz = vecs[0].shape[0]
            # except AttributeError:
            #     self.input_sz = None
        
        def __len__(self):
            return len(self.vecs)
        
        def __getitem__(self, index):
            return self.vecs[index], self.labels[index], index

def _bert_vectors(parent, domain, train, seed, bert):
    loc = open(f'{parent}_relations_{bert}.pkl', 'rb')
    data = pickle.load(loc)
    data = [x for x in data if x['domain'] == domain]
    scoped_random = Random(seed)
    scoped_random.shuffle(data)
    n = len(data) // 2

    if train is not None:
        data = data[:n] if train else data[n:]
    
    if bert == 'tokens':
        embeddings = [{
            'input_ids' : x['tokens'][0].numpy().flatten(),
            'attention_mask' : x['tokens'][1].numpy().flatten() 
            } for x in data]
    else:
        embeddings = [x['embeddings'].flatten() for x in data]

    if parent == 'pdtb':
        classes = [x['discourse_sense'].split('.')[0] 
            for x in data]
        try:
            classes = [NORMED_DISCOURSE_SENSES[x] 
                for x in classes]
        except:
            print(classes)
    elif parent == 'gum':
        classes = [x['relation'] for x in data]

    classes = [DISCOURSE_SENSE[parent][x] for x in classes]
    return Dataset(embeddings, classes)

def biodrb(train=True, seed=0, bert='sentence'):
    return _bert_vectors('pdtb', 'BioDRB', train, seed, bert)

def pdtb(train=True, seed=0, bert='sentence'):
    return _bert_vectors('pdtb', 'PDTB', train, seed, bert)

def ted(train=True, seed=0, bert='sentence'):
    return _bert_vectors('pdtb', 'TED-MDB', train, seed, bert)

PDTB_DATASETS = lambda b: [
    (f'{b[0]}_pdtb', lazy_kwarg_init(pdtb, bert=b)),
    (f'{b[0]}_biodrb', lazy_kwarg_init(biodrb, bert=b)),
    (f'{b[0]}_ted', lazy_kwarg_init(ted, bert=b))
]

def reddit(train=True, seed=0, bert='sentence'):
    return _bert_vectors('gum', 'reddit', train, seed, bert)

def voyage(train=True, seed=0, bert='sentence'):
    return _bert_vectors('gum', 'voyage', train, seed, bert)

def interview(train=True, seed=0, bert='sentence'):
    return _bert_vectors('gum', 'interview', train, seed, bert)

def whow(train=True, seed=0, bert='sentence'):
    return _bert_vectors('gum', 'whow', train, seed, bert)

def fiction(train=True, seed=0, bert='sentence'):
    return _bert_vectors('gum', 'fiction', train, seed, bert)

def academic(train=True, seed=0, bert='sentence'):
    return _bert_vectors('gum', 'academic', train, seed, bert)

def bio(train=True, seed=0, bert='sentence'):
    return _bert_vectors('gum', 'bio', train, seed, bert)

GUM_DATASETS = lambda b: [
    (f'{b[0]}_reddit', lazy_kwarg_init(reddit, bert=b)),
    (f'{b[0]}_voyage', lazy_kwarg_init(voyage, bert=b)),
    (f'{b[0]}_interview', lazy_kwarg_init(interview, bert=b)),
    (f'{b[0]}_whow', lazy_kwarg_init(whow, bert=b)),
    (f'{b[0]}_fiction', lazy_kwarg_init(fiction, bert=b)),
    (f'{b[0]}_academic', lazy_kwarg_init(academic, bert=b)),
    (f'{b[0]}_bio', lazy_kwarg_init(bio, bert=b))
]