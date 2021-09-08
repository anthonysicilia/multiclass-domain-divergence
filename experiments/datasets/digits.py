import torchvision as tv

from .utils import make_frozenset, GaussianNoise

IMAGE_MEAN = 0.1307
IMAGE_STD = 0.3081

def mnist(train=True, seed=0):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    dset = tv.datasets.MNIST('./', train=True, download=True, 
        transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def rotated_mnist(train=True, seed=0, degrees=360):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28,28)), 
        tv.transforms.RandomRotation(degrees=degrees),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    dset = tv.datasets.MNIST('./', train=True, download=True, 
        transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def noisy_mnist(train=True, seed=0, scale=1.0):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,)),
        GaussianNoise(std=scale * IMAGE_STD)])
    dset = tv.datasets.MNIST('./', train=True, download=True, 
        transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def svhn(train=True, seed=0):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    dset = tv.datasets.SVHN('SVHN/',
        split='train', download=True, transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def rotated_svhn(train=True, seed=0, degrees=360):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.RandomRotation(degrees=degrees),
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    dset = tv.datasets.SVHN('SVHN/',
        split='train', download=True, transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def noisy_svhn(train=True, seed=0, scale=1.0):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,)),
        GaussianNoise(std=scale * IMAGE_STD)])
    dset = tv.datasets.SVHN('SVHN/',
        split='train', download=True, transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def usps(train=True, seed=0):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    dset = tv.datasets.USPS('USPS/',
        train=True, download=True, transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def rotated_usps(train=True, seed=0, degrees=360):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.RandomRotation(degrees=degrees),
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    dset = tv.datasets.USPS('USPS/',
        train=True, download=True, transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def noisy_usps(train=True, seed=0, scale=1.0):
    transform = tv.transforms.Compose([
        tv.transforms.Grayscale(num_output_channels=1), 
        tv.transforms.Resize((28, 28)), 
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,)),
        GaussianNoise(std=scale * IMAGE_STD)])
    dset = tv.datasets.USPS('USPS/',
        train=True, download=True, transform=transform)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

def fake(train=True, seed=0):
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(), 
        tv.transforms.Normalize((IMAGE_MEAN,), (IMAGE_STD,))])
    dset = tv.datasets.FakeData(size=60_000, image_size=(28, 28), 
        num_classes=10, transform=transform, random_offset=seed)
    return make_frozenset(dset, train=train, seed=seed, 
        num_classes=10)

DATASETS = [
    ('mnist', mnist), 
    ('svhn', svhn), 
    ('usps', usps), 
    ('fake', fake),
    ('r_mnist', rotated_mnist),
    ('r_svhn', rotated_svhn),
    ('r_usps', rotated_usps),
    ('n_mnist', noisy_mnist),
    ('n_svhn', noisy_svhn),
    ('n_usps', noisy_usps)]
