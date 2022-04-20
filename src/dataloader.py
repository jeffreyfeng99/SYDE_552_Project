import os
import cv2
from PIL import Image
from json import load
import numpy as np

import torch
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data
from torch.utils.data import Subset

class CustomDataset(data.Dataset):
    def __init__(self, data_root, transform=None, train=False):
        self.root = data_root
        self.transform = transform

        self.img_paths = [f for root, dirs, files in os.walk(data_root) for f in files if f.endswith('.JPEG')]
        
        if train:
            self.img_labels = load(open('./tiny-imagenet-200/train_class_dict.json', 'r'))
        else:
            self.img_labels = load(open('./tiny-imagenet-200/val_class_dict.json', 'r'))

        self.n_data = len(self.img_paths)

    def __getitem__(self, item):
        img_paths = self.img_paths[item]
        labels = self.img_labels[os.path.basename(img_paths)]['index']

        imgs = Image.open(os.path.join(self.root, img_paths).replace('\\', '/')).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels, img_paths

    def __len__(self):
        return self.n_data


def load_target_image(path, normalize=True):

    imgs = np.asarray(Image.open(path).convert('L'))

    if normalize:
        imgs = (imgs - np.min(imgs))/(np.max(imgs)-np.min(imgs))

    return imgs


def create_dataloader(dataset_path, dataset_size, batch_size, num_workers, no_norm=False):

    if no_norm:
        transform = transforms.Compose([
                            transforms.ToTensor()
                        ])
    else:
        if 'cifar' not in dataset_path:
            transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.243, 0.261))
            ])

    if 'cifar' not in dataset_path:
        dataset = CustomDataset(dataset_path, transform=transform)
    else:
        dataset = datasets.CIFAR10(root='./cifar10', train=False,
                                                download=True, transform=transform)

    dataset = Subset(dataset, torch.arange(dataset_size))
    data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)

    return data_loader