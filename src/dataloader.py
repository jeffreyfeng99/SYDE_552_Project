import os
import cv2
from PIL import Image
from json import load

import torch
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as data
from torch.utils.data import Subset

class CustomDataset(data.Dataset):
    def __init__(self, data_root, transform=None):
        self.root = data_root
        self.transform = transform

        self.img_paths = [f for root, dirs, files in os.walk(data_root) for f in files if f.endswith('.JPEG')]
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


def create_dataloader(dataset_path, dataset_size, batch_size, num_workers):
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    dataset = CustomDataset(dataset_path, transform=transform)
    dataset = Subset(dataset, torch.arange(dataset_size))
    data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers)

    return data_loader