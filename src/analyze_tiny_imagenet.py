import os
from json import load, dump
import pandas as pd
from glob import glob
import os

import torch
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10
from PIL import Image

class CIFAR10(CIFAR10):
    def __init__(self, root,train,download,transform):
        super(CIFAR10, self).__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, torch.Tensor):
            img = Image.fromarray(img.numpy())
        else:
            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        paths = f'{index}.JPEG'
        return img, target, paths

def create_cifar_dict():
    dataset = CIFAR10(root='./cifar10', train=False,
                                                download=True, transform=None)

    cifar_dict = {}

    for item in dataset:
        cifar_dict[item[2]] = {'index': item[1]}
    
    dump(cifar_dict, open('./cifar10/val_class_dict.json', 'w'),
         indent=2)

def create_class_dict():
    # Create a new version only including tiny 200 classes
    df = pd.read_csv('./tiny-imagenet-200/words.txt', sep='\t', header=None)
    keys, classes = df[0], df[1]
    class_dict = dict(zip(keys, classes))

    tiny_class_dict = {}
    cur_index = 0

    for directory in glob('./tiny-imagenet-200/train/*'):
        cur_key = os.path.basename(directory)
        tiny_class_dict[cur_key] = {'class': class_dict[cur_key],
                                    'index': cur_index}
        cur_index += 1

    dump(tiny_class_dict, open('./tiny-imagenet-200/class_dict.json', 'w'),
         indent=2)


def create_val_class_dict():
    tiny_class_dict = load(open('./tiny-imagenet-200/class_dict.json', 'r'))
    tiny_val_class_dict = {}

    # Create a dictionary for validation images
    df = pd.read_csv('./tiny-imagenet-200/val/val_annotations.txt', sep='\t',
                     header=None)
    image_names = df[0]
    image_classes = df[1]

    for i in range(len(image_names)):
        tiny_val_class_dict[image_names[i]] = {
            'class': tiny_class_dict[image_classes[i]]['class'],
            'index': tiny_class_dict[image_classes[i]]['index'],
        }

    dump(tiny_val_class_dict, open('./tiny-imagenet-200/val_class_dict.json',
                                   'w'),
         indent=2)

def create_train_class_dict():
    tiny_class_dict = load(open('./tiny-imagenet-200/class_dict.json', 'r'))
    tiny_val_class_dict = {}

    for root, dir, files in os.walk('./tiny-imagenet-200/train'):
        for file in files:
            if '.JPEG' in file:
                image_class = file.split('_')[0]
                tiny_val_class_dict[file] = {
                    'class': tiny_class_dict[image_class]['class'],
                    'index': tiny_class_dict[image_class]['index'],
                }
        
    dump(tiny_val_class_dict, open('./tiny-imagenet-200/train_class_dict.json',
                                   'w'),
         indent=2)


if __name__ == '__main__':
    # create_class_dict()
    # create_val_class_dict()
    # create_train_class_dict()
    create_cifar_dict()