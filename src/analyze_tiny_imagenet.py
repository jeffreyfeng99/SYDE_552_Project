import os
from json import load, dump
import pandas as pd
from glob import glob
import os

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
    create_train_class_dict()