import argparse
import numpy as np
import os
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.utils.data as data
from torchvision import transforms, datasets
from datetime import datetime 
import cv2
from PIL import Image
from tqdm import tqdm
from json import load

from model import *
from utils import *

class CustomDataset(data.Dataset):
    def __init__(self, data_root, transform=None):
        self.root = data_root
        self.transform = transform

        self.img_paths = [f for root,dirs,files in os.walk(data_root) for f in files if f.endswith('.JPEG')]
        self.img_labels = load(open('./tiny-imagenet-200/val_class_dict.json', 'r'))
    
        self.n_data = len(self.img_paths)

    def __getitem__(self, item):
        img_paths = self.img_paths[item]
        labels = self.img_labels[os.path.basename(img_paths)]['index']
        
        imgs = Image.open(os.path.join(self.root, img_paths).replace('\\','/')).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels, img_paths

    def __len__(self):
        return self.n_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual explanations from spiking neural networks using inter‑spike intervals', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pretrainedmodel_pth', default='./pretrained/pretrained_tiny_t30.pth.tar', type=str, help='path for pretrained model')
    parser.add_argument('--dataset_pth', default='./tiny-imagenet-200/val/images', type=str, help='path for validation dataset')
    parser.add_argument('--timesteps', default=30, type=float, help='timesteps')
    parser.add_argument('--batch_size', default=1, type=int,   help='batch size should be 1')
    parser.add_argument('--leak_mem', default=0.99, type=float, help='leak_mem')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--gamma', default=0.5, type=float, help='float')
    parser.add_argument('--target_layer', default=8, type=int, help='target_layer [4, 6, 8] is available')
    parser.add_argument('--output_pth',  default='./output', type=str, help='path for output directory')
    parser.add_argument('--limit_output',  action='store_true')
    parser.add_argument('--visual_imagesize', default=128, type=int)
    parser.add_argument('--dset_size', default=100, type=int)

    args = parser.parse_args()
    
    output_path = os.path.join(args.output_pth, datetime.now().strftime("%m%d%Y")).replace('\\','/')
    os.makedirs(output_path, exist_ok=True)

    # select number of samples for visualization
    img_nums = [10, 52]

    gamma = args.gamma
    num_timestep = args.timesteps
    visual_imagesize = args.visual_imagesize
    target_layer = args.target_layer

    # Mean and SD are calculuated
    val_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
    val_dataset = CustomDataset(args.dataset_pth, transform=val_transform)
    val_dataset = Subset(val_dataset, torch.arange(args.dset_size))
    val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=args.batch_size, 
                    shuffle=False,
                    num_workers=args.num_workers)

    #--------------------------------------------------
    # Instantiate the SNN model
    #--------------------------------------------------
    model = SNN_VGG11()

    print('********** Loading Model **********')

    model = torch.nn.DataParallel(model).cuda()
    save_model_statedict = torch.load(args.pretrainedmodel_pth)['state_dict']
    cur_dict = model.state_dict()

    for key in save_model_statedict.keys():
        if key in cur_dict:
            if (save_model_statedict[key].shape == cur_dict[key].shape):
                cur_dict[key] = save_model_statedict[key]
            else:
                print("Error mismatch")

    model.load_state_dict(cur_dict)
    model.eval()

    #--------------------------------------------------
    # Extracting heatmap
    #--------------------------------------------------

    def getCAM(feature_conv, weight):
        _, nc, h, w = feature_conv.shape
        cam = weight.dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / (np.max(cam) +1e-3)
        return [cam_img]

    def getForwardCAM(feature_conv):
        cam = feature_conv.sum(axis =0).sum(axis =0)
        cam = cam - np.min(cam)
        cam_img = cam / (np.max(cam) +1e-3)
        return [cam_img]

    cam_dict = {}

    img_idx = 0
    
    classification_accuracy_tracker = AverageMeter()

    for j, data in enumerate(tqdm(val_loader)):
        if args.limit_output is True:
            if j > np.max(img_nums):
                continue
            if j not in img_nums:
                continue
            
        model.zero_grad()
        model.module.saved_grad = 0
        model.module.saved_forward = []

        images, labels, paths = data
        images = images.cuda()
        labels = labels.cuda()

        output  = model(images, target_layer=target_layer)
        classification_accuracy_tracker.update(accuracy(output,labels)[0])

        if j in img_nums:
            original = images[0, ...].cpu().numpy().transpose(1,2,0)
            original = cv2.resize(original, dsize=(visual_imagesize, visual_imagesize))
            original = (original - np.min(original))/(np.max(original)-np.min(original))
            original = (np.array(original)*255).astype('uint8')
            original_save = Image.fromarray(original)
            original_save.save(os.path.join(output_path, 'original_%s.jpg' % j).replace('\\','/'))

        process = 0
        time = 0
        overlay_list = []
        previous_spike_time_list = []
        activation_list_value = (model.module.saved_forward)

        for l, activation in enumerate(activation_list_value):
            activation = activation
            previous_spike_time_list.append(activation)
            weight = 0

            for prev_t in range(len(previous_spike_time_list)):
                delta_t = time - previous_spike_time_list[prev_t]* prev_t
                weight +=  torch.exp(gamma * (-1) * delta_t)

            weighted_activation = weight.cuda() * activation
            weighted_activation = weighted_activation.data.cpu().numpy()
            overlay = getForwardCAM(weighted_activation)
            overlay_list.append(overlay[0])

            if j in img_nums:
                if process%3 == 0:
                    sam = (np.array(1.-overlay[0])*255).astype('uint8')
                    sam = cv2.resize(sam, dsize=(visual_imagesize, visual_imagesize))
                    sam = cv2.applyColorMap(sam, cv2.COLORMAP_JET)
                    blended = cv2.addWeighted(original, 0.5, sam, 0.5, 0.0)
                    sam = Image.fromarray(sam)
                    sam.save(os.path.join(output_path, 'sam_%s_%s.jpg' % (j,process//3+1)).replace('\\','/'))
                    blended = Image.fromarray(blended)
                    blended.save(os.path.join(output_path, 'sam_overlay_%s_%s.jpg' % (j,process//3+1)).replace('\\','/'))

            process += 1
            time += 1

        cam_dict[j] = overlay_list
        img_idx +=1

    print(classification_accuracy_tracker)