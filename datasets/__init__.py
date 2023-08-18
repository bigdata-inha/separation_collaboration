import torch
import torchvision


def get_dataset(dataset, data_dir, transform, train=True, download=True, debug_subset_size=None):
    if dataset == 'mnist':
        dataset = torchvision.datasets.MNIST(data_dir, train=train, transform=transform, download=True)
    elif dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(data_dir, train=train, transform=transform, download=True)
    elif dataset == 'svhn':
        if train == True:
            dataset = torchvision.datasets.SVHN(data_dir, split = 'train', transform=transform, download=True)
        else:
            dataset = torchvision.datasets.SVHN(data_dir, split = 'test', transform=transform, download=True)
    else:
        raise NotImplementedError
    if debug_subset_size is not None:
        dataset = torch.utils.data.Subset(dataset, range(0, debug_subset_size)) # take only one batch

    return dataset