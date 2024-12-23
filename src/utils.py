import os
import json

from sklearn import metrics
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from src.augmentations import RandAugment


class Averager:
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def sum_precision_score(predict_y, img_y):
    confusion_matrix = metrics.confusion_matrix(img_y, predict_y)
    precision = metrics.precision_score(img_y, predict_y)
    return confusion_matrix, precision

def sum_accuracy_score(predict_y, img_y):
    confusion_matrix = metrics.confusion_matrix(img_y, predict_y)
    accuracy = metrics.accuracy_score(img_y, predict_y)
    return confusion_matrix, accuracy


class DatasetSplit(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        (images1, images2), label = self.dataset[self.idxs[item]]
        return (images1, images2), label
    

class TransformDatasets:
    def __init__(self, weak_trans, strong_trans):
        self.trans1 = weak_trans
        self.trans2 = strong_trans
        
    def __call__(self, x):
        x1 = self.trans1(x) 
        x2 = self.trans2(x) 
        return x1, x2


def client_dataloader(dataset, user_groups, num_clients, batch_size):
    client_loaders_dict = {}
    for i in user_groups.keys():
        split = DatasetSplit(dataset, user_groups[i])
        client_loaders_dict[i] = DataLoader(split, batch_size=batch_size, shuffle=True, drop_last=True)

    return client_loaders_dict


def server_dataloader(dataset, batch_size):
    server_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return server_loader


def save_to_json(data, directory, filename):
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # 将numpy数组转换为Python列表
        return json.JSONEncoder.default(self, obj)
    

def convert_keys_to_int(d):
    return {int(k): v for k, v in d.items()}



class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img