from torchvision import datasets
from torchvision import transforms

import copy
import numpy as np
import random

import torchvision
import torch
from .utils.noisify import noisify_label


from .utils.dataset import classify_label, show_clients_data_distribution
from .utils.sampling import client_iid_indices, clients_non_iid_indices


def get_cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_train_100 = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    data_local_training = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
    data_global_test = datasets.CIFAR10(args.data_path, train=False, transform=transform_test)
    data_global_distill = datasets.CIFAR100(args.data_path_cifar100, train=True, download=True,
                                            transform=transform_train_100)

    if args.iid:
        list_client2indices = client_iid_indices(data_local_training, args.num_clients)
    else:
        list_label2indices = classify_label(data_local_training, args.num_classes)
        list_client2indices = clients_non_iid_indices(list_label2indices, args.num_classes, args.num_clients, args.non_iid_alpha, args.seed)

    # show_clients_data_distribution(data_local_training, list_client2indices, args.num_classes)


    return data_local_training, data_global_test, list_client2indices,data_global_distill
