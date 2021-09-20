import pickle
import torch
import torchvision as tv

from PIL import Image
from random import Random

from .discourse import Dataset as FeatureSet
from .utils import make_frozenset

IMAGE_MEANS = [0.485, 0.456, 0.406]
IMAGE_STDS = [0.229, 0.224, 0.225]

class ImageSet(torch.utils.data.Dataset):
    
    def __init__(self, root, domain, transforms):
        path_file = f'{root}/{domain}/paths.txt'
        lines = [p.strip().split() for p in open(path_file, 'r')]
        self.paths = [f'{root}/{path}' for path, _ in lines]
        self.labels = [int(label) - 1 for _, label in lines]
        self.num_classes = max(set(self.labels)) + 1
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        y = self.labels[index]
        x = Image.open(path)
        return self.transforms(x), y

def pacs_photo(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('PACS', 'photo', transforms)
    return make_frozenset(dset, train=train, seed=seed)

def pacs_art(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('PACS', 'art_painting', transforms)
    return make_frozenset(dset, train=train, seed=seed)

def pacs_cartoon(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('PACS', 'cartoon', transforms)
    return make_frozenset(dset, train=train, seed=seed)

def pacs_sketch(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('PACS', 'sketch', transforms)
    return make_frozenset(dset, train=train, seed=seed)

def officehome_art(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('OfficeHome', 'art', transforms)
    return make_frozenset(dset, train=train, seed=seed)

def officehome_clipart(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('OfficeHome', 'clipart', transforms)
    return make_frozenset(dset, train=train, seed=seed)

def officehome_product(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('OfficeHome', 'product', transforms)
    return make_frozenset(dset, train=train, seed=seed)

def officehome_world(train=True, seed=0):
    transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(IMAGE_MEANS, IMAGE_STDS)
    ])
    dset = ImageSet('OfficeHome', 'real_world', transforms)
    return make_frozenset(dset, train=train, seed=seed)

PACS_DATASETS = [
    ('pacs_photo', pacs_photo), 
    ('pacs_art', pacs_art), 
    ('pacs_cartoon', pacs_cartoon), 
    ('pacs_sketch', pacs_sketch)]

OFFICEHOME_DATASETS = [
    ('oh_world', officehome_world), 
    ('oh_art', officehome_art),
    ('oh_product', officehome_product), 
    ('oh_clipart', officehome_clipart)]

def _ft_vectors(parent, domain, train, seed):
    loc = open(f'{parent}_{domain}_rn50fts.pkl', 'rb')
    data = pickle.load(loc)
    scoped_random = Random(seed)
    scoped_random.shuffle(data)
    n = len(data) // 2
    if train is not None:
        data = data[:n] if train else data[n:]
    return FeatureSet([x for x,_ in data], [y for _,y in data])

def pacs_photo_fts(train=True, seed=0):
    return _ft_vectors('pacs', 'photo', train, seed)

def pacs_art_fts(train=True, seed=0):
    return _ft_vectors('pacs', 'art', train, seed)

def pacs_cartoon_fts(train=True, seed=0):
    return _ft_vectors('pacs', 'cartoon', train, seed)

def pacs_sketch_fts(train=True, seed=0):
    return _ft_vectors('pacs', 'sketch', train, seed)

def officehome_world_fts(train=True, seed=0):
    return _ft_vectors('oh', 'world', train, seed)

def officehome_art_fts(train=True, seed=0):
    return _ft_vectors('oh', 'art', train, seed)

def officehome_product_fts(train=True, seed=0):
    return _ft_vectors('oh', 'product', train, seed)

def officehome_clipart_fts(train=True, seed=0):
    return _ft_vectors('oh', 'clipart', train, seed)

PACS_FTS_DATASETS = [
    ('pacs_photo_fts', pacs_photo_fts), 
    ('pacs_art_fts', pacs_art_fts), 
    ('pacs_cartoon_fts', pacs_cartoon_fts), 
    ('pacs_sketch_fts', pacs_sketch_fts)]

OFFICEHOME_FTS_DATASETS = [
    ('oh_world_fts', officehome_world_fts), 
    ('oh_art_fts', officehome_art_fts),
    ('oh_product_fts', officehome_product_fts), 
    ('oh_clipart_fts', officehome_clipart_fts)]

