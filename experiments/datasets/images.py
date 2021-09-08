import torch
import torchvision as tv

from PIL import Image

from .utils import make_frozenset

IMAGE_MEANS = [0.485, 0.456, 0.406]
IMAGE_STDS = [0.229, 0.224, 0.225]

class ImageSet(torch.utils.data.Dataset):
    
    def __init__(self, root, domain, transforms):
        path_file = f'{root}/{domain}/paths.txt'
        lines = [p.strip().split() for p in open(path_file, 'r')]
        self.paths = [f'{root}/{path}' for path, _ in lines]
        self.labels = [int(label) - 1 for _, label in lines]
        self.num_classes = max(set(self.labels))
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

PACS_FT_DATASETS = []
OFFICEHOME_FT_DATASETS = []

