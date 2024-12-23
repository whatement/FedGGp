import numpy as np
import os
import torch
import random
from torchvision import datasets, transforms

from alg.fedavg_lower import FedAvg_Lower
from alg.fedggp import FedGGp
from alg.twin_sight import Twin_sight
from model.resnet9 import ResNet9
from src.augmentations import RandAugment
from options import args_parser
from src.split import split_dataset
from src.utils import GaussianBlur, TransformDatasets

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)


def seed_setup(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

def init_set(args):

    if args.dataset == 'cifar10':
        args.num_classes = 10
        weak_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        strong_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10, dataset=args.dataset),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        test_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        
        train = datasets.CIFAR10('~/dataset/cifar10', train=True, download=True, transform=TransformDatasets(weak_trans, strong_trans))
        test = datasets.CIFAR10('~/dataset/cifar10', train=False, download=True, transform=test_trans)
        backbone = ResNet9(in_channels=3, num_classes=args.num_classes) 
        
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        weak_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        strong_trans = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10, dataset=args.dataset),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        test_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
            ])
        
        train = datasets.CIFAR100('~/dataset/cifar100', train=True, download=True, transform=TransformDatasets(weak_trans, strong_trans))
        test = datasets.CIFAR100('~/dataset/cifar100', train=False, download=True, transform=test_trans)
        backbone = ResNet9(in_channels=3, num_classes=args.num_classes)  
        
    elif args.dataset == 'svhn':
        args.num_classes = 10
        weak_trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        strong_trans = transforms.Compose([
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10, dataset=args.dataset),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])
        test_trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
            ])

        train = datasets.SVHN('~/dataset/svhn', split='train', download=True, transform=TransformDatasets(weak_trans, strong_trans))
        test = datasets.SVHN('~/dataset/svhn', split='test', download=True, transform=test_trans)
        backbone = ResNet9(in_channels=3, num_classes=args.num_classes) 
        # backbone = ResNet18(args.num_classes)
        
    elif args.dataset == 'fmnist':
        args.num_classes = 10
        weak_trans = transforms.Compose([
                transforms.Resize(32),
                # transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        strong_trans = transforms.Compose([
                transforms.Resize(32),
                # transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                RandAugment(n=2, m=10,  dataset=args.dataset),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])
        test_trans = transforms.Compose([
                transforms.Resize(32),
                # transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,))
            ])

        train = datasets.FashionMNIST('~/dataset/fmnist', train=True, download=True, transform=TransformDatasets(weak_trans, strong_trans))
        test = datasets.FashionMNIST('~/dataset/fmnist', train=False, download=True, transform=test_trans)
        backbone = ResNet9(in_channels=1, num_classes=args.num_classes) 

    else:
        raise NotImplementedError

    return train, test, backbone


if __name__ == '__main__':

    args = args_parser()
    seed_setup(args.seed)

    train_dataset, test_dataset, model = init_set(args)

    train_labeled_groups, train_unlabeled_groups, test_user_groups, data_distribution_lists = split_dataset(args, train_dataset, test_dataset)
    
    if args.algorithm == "fedAvg_lower":
        alg = FedAvg_Lower(train_dataset, test_dataset, train_labeled_groups, train_unlabeled_groups, test_user_groups, data_distribution_lists, model, args)
        
    elif args.algorithm == "twin_sight":
        alg = Twin_sight(train_dataset, test_dataset, train_labeled_groups, train_unlabeled_groups, test_user_groups, data_distribution_lists, model, args)
    
    elif args.algorithm == "fedggp":
        alg = FedGGp(train_dataset, test_dataset, train_labeled_groups, train_unlabeled_groups, test_user_groups, data_distribution_lists, model, args)
        
    else:
        raise NotImplementedError
    
    alg.trainer()
    

