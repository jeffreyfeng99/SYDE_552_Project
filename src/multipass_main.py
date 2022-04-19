import argparse
from matplotlib import image
import numpy as np
import random
import os
import cv2
from PIL import Image
from tqdm import tqdm
from json import load

from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import torch.utils.data as data
from torchvision import transforms, datasets, models
from datetime import datetime

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

from model import *
from utils import *
from dataloader import *


def setup_gradcam():
    reference_model = models.vgg11_bn(pretrained=True)
    reference_model.eval()

    reference_target_layer = reference_model.features[-1]
    cam = GradCAM(model=reference_model,
                  target_layers=[reference_target_layer],
                  use_cuda=True)
    return cam


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    dataset_path = args.dataset_pth

    output_dirs = {}
    output_basedirs = ['original', 'gradcam','grayscale_gradcam',
                       'noisy', 'noisy_SAM_A', 'noisy_SAM_B',
                       'clean_SAM_A', 'clean_SAM_B',
                       'clean_minerror_SAM_A', 'clean_minerror_SAM_B',
                       'attacked_SAM_1A', 'attacked_SAM_1B',
                       'attacked_minerror_SAM_1A', 'attacked_minerror_SAM_1B',
                       'attacked_SAM_2A', 'attacked_SAM_2B',
                       'attacked_minerror_SAM_2A', 'attacked_minerror_SAM_2B']
    for output_base in output_basedirs:
        output_dir = os.path.join(args.output_root, datetime.now().strftime("%m%d%Y"), output_base).replace('\\', '/')
        os.makedirs(output_dir, exist_ok=True)
        output_dirs[output_base] = output_dir

    # select number of samples for visualization
    img_nums = list(range(20)) #[10, 52] 

    gamma = args.gamma
    num_timestep = args.timesteps
    leak_mem = args.leak_mem
    visual_imagesize = args.visual_imagesize
    target_layer = args.target_layer

    T_min = args.T_min
    T_max = num_timestep * T_min / 2.
    N_max = args.N_max

    # Set up dataloader
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataset_size = args.dataset_size

    val_loader = create_dataloader(dataset_path, dataset_size, batch_size, num_workers)

    # Set up adversarial noise
    flow = args.flow
    if flow == "attack-img" or flow == "attack-sam" or flow == "all":
        gaussian_noise = AddGaussianNoise(mean=args.g_mu, std=args.g_sigma)

    #--------------------------------------------------
    # Obtain GradCAMs from validation dataset
    #--------------------------------------------------
    aug_smooth = args.aug_smooth
    eigen_smooth = args.eigen_smooth
    with setup_gradcam() as cam:
        cam.batch_size = batch_size

        for j, data in enumerate(tqdm(val_loader)):
            if args.limit_output is True:
                if j > np.max(img_nums):
                    continue
                if j not in img_nums:
                    continue

            images, _, paths = data
            images = images.cuda()

            grayscale_cam = cam(input_tensor=transforms.functional.resize(images, 224),
                                targets=None,
                                aug_smooth=aug_smooth,
                                eigen_smooth=eigen_smooth)

            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, dsize=(64, 64))
            np_img = images.cpu().numpy()[0]
            np_img = np_img.transpose(1,2,0)
            np_img = (np_img-np.min(np_img))/(np.max(np_img) - np.min(np_img))

            gscale_img = (grayscale_cam*255).astype('uint8')
            grayscale_cam_image_path = os.path.join(output_dirs['grayscale_gradcam'], paths[0]).replace('\\', '/')
            cv2.imwrite(grayscale_cam_image_path, gscale_img)
            
            gradcam_image = show_cam_on_image(np_img, grayscale_cam, use_rgb=True)
            
            # gradcam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            gradcam_image = cv2.cvtColor(gradcam_image, cv2.COLOR_RGB2BGR)
            cam_image_path = os.path.join(output_dirs['gradcam'], paths[0]).replace('\\', '/')
            cv2.imwrite(cam_image_path, gradcam_image)
    del cam
    #--------------------------------------------------
    # Instantiate both SNN models
    #--------------------------------------------------
    save_model_statedict = torch.load(args.pretrainedmodel_pth)['state_dict']

    model_A = SNN_VGG11(spike_code='poisson') # Kim & Panda model

    print('********** Loading Model A **********')

    model_A = torch.nn.DataParallel(model_A).cuda()
    cur_dict = model_A.state_dict()

    for key in save_model_statedict.keys():
        if key in cur_dict:
            if (save_model_statedict[key].shape == cur_dict[key].shape):
                cur_dict[key] = save_model_statedict[key]
            else:
                print("Error mismatch")

    model_A.load_state_dict(cur_dict)
    model_A.eval()

    model_B = SNN_VGG11(spike_code='poisson', alif=True)
    # model_B = SNN_VGG11(T_max=T_max, T_min=T_min, spike_code='burst') # Our model

    print('********** Loading Model B **********')

    model_B = torch.nn.DataParallel(model_B).cuda()
    cur_dict = model_B.state_dict()

    for key in save_model_statedict.keys():
        if key in cur_dict:
            if (save_model_statedict[key].shape == cur_dict[key].shape):
                cur_dict[key] = save_model_statedict[key]
            else:
                print("Error mismatch")

    model_B.load_state_dict(cur_dict)
    model_B.eval()

    models = [model_A, model_B]

    #--------------------------------------------------
    # Start multipass flow
    #--------------------------------------------------

    cam_dict_A = {}
    cam_dict_B = {}
    img_idx = 0

    classification_accuracy_tracker_A = AverageMeter()
    classification_accuracy_tracker_B = AverageMeter()
    classification_accuracy_trackers = [classification_accuracy_tracker_A, classification_accuracy_tracker_B]
    localization_tracker_A = LocalizationMeter()
    localization_tracker_B = LocalizationMeter()
    localization_trackers = [localization_tracker_A, localization_tracker_B]

    for j, data in enumerate(tqdm(val_loader)):
        if args.limit_output is True:
            if j > np.max(img_nums):
                continue
            if j not in img_nums:
                continue

        for model in models:
            model.zero_grad()
            model.module.saved_grad = 0
            model.module.saved_forward = []

        images, labels, paths = data
        images = images.cuda()
        labels = labels.cuda()
        paths = paths[0]

        # Save original images
        if j in img_nums:
            original = images[0, ...].cpu().numpy().transpose(1, 2, 0)
            original = cv2.resize(original, dsize=(visual_imagesize, visual_imagesize))
            original = (original - np.min(original)) / (np.max(original) - np.min(original))
            original = (np.array(original) * 255).astype('uint8')
            cv2.imwrite(os.path.join(output_dirs['original'], f'original_{j}.jpg').replace('\\', '/'),
                        cv2.cvtColor(original, cv2.COLOR_RGB2BGR))

        # Compute classification accuracies
        for n in range(len(models)):
            output = models[n](images, target_layer=target_layer)
            classification_accuracy_trackers[n].update(accuracy(output, labels)[0])

        process = 0
        time = 0
        overlay_list = {'A': [], 'B': []}
        previous_spike_time_list = {'A': [], 'B': []}

        for l, (activation_A, activation_B) in enumerate(zip(model_A.module.saved_forward, model_B.module.saved_forward)):
            previous_spike_time_list['A'].append(activation_A)
            previous_spike_time_list['B'].append(activation_B)
            weight = 0
            
            for k in previous_spike_time_list.keys():
                for prev_t in range(len(previous_spike_time_list[k])):
                    delta_t = time - previous_spike_time_list[k][prev_t] * prev_t
                    weight += torch.exp(gamma * (-1) * delta_t)

            # Compute SAMs
            weighted_activation_A = weight.cuda() * activation_A
            weighted_activation_A = weighted_activation_A.data.cpu().numpy()
            overlay_A = getForwardCAM(weighted_activation_A)
            overlay_list['A'].append(overlay_A)

            weighted_activation_B = weight.cuda() * activation_B
            weighted_activation_B = weighted_activation_B.data.cpu().numpy()
            overlay_B = getForwardCAM(weighted_activation_B)
            overlay_list['B'].append(overlay_B)

            # Save SAMs
            if j in img_nums:
                if process % 3 == 0:
                    for k in overlay_list.keys():
                        sam = (np.array(1. - overlay_list[k][-1]) * 255).astype('uint8')
                        sam = cv2.resize(sam, dsize=(visual_imagesize, visual_imagesize))
                        sam = cv2.applyColorMap(sam, cv2.COLORMAP_JET)
                        blended = cv2.addWeighted(original, 0.5, sam, 0.5, 0.0)
                        cv2.imwrite(
                            os.path.join(output_dirs[f'clean_SAM_{k}'], f'sam_{j}_{process // 3 + 1}.jpg').replace('\\', '/'),
                            cv2.cvtColor(sam, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(
                            os.path.join(f'clean_SAM_{k}', f'sam_overlay_{j}_{process // 3 + 1}.jpg').replace('\\', '/'),
                            cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

            process += 1
            time += 1

        for n, k in enumerate(overlay_list.keys()):
            output_path = os.path.join(output_dirs[f'clean_minerror_SAM_{k}'], paths).replace('\\','/')
            grayscale_cam = load_target_image(os.path.join(output_dirs['grayscale_gradcam'], paths).replace('\\','/'))
            error, index = localization_error(overlay_list[k], grayscale_cam, save_image=output_path)
            localization_trackers[n].update(error, paths, index)

        img_idx += 1

    for n, k in enumerate(overlay_list.keys()):
        localization_trackers[n].export(os.path.join(output_dirs[f'clean_minerror_SAM_{k}'], 'localization_error.csv').replace('\\','/'))
        localization_trackers[n].print_output()
        classification_accuracy_trackers[n].print_output()

    # --------------------------------------------------
    # PASS #2: Add Gaussian noise to images
    # --------------------------------------------------
    img_idx = 0

    classification_accuracy_tracker_2A = AverageMeter()
    classification_accuracy_tracker_2B = AverageMeter()
    classification_accuracy_trackers_2 = [classification_accuracy_tracker_2A, classification_accuracy_tracker_2B]
    localization_tracker_2A = LocalizationMeter()
    localization_tracker_2B = LocalizationMeter()
    localization_trackers_2 = [localization_tracker_2A, localization_tracker_2B]

    for j, data in enumerate(tqdm(val_loader)):
        if args.limit_output is True:
            if j > np.max(img_nums):
                continue
            if j not in img_nums:
                continue

        for model in models:
            model.zero_grad()
            model.module.saved_grad = 0
            model.module.saved_forward = []

        images, labels, paths = data
        images = images.cuda()
        labels = labels.cuda()
        paths = paths[0]

        # Apply gaussian noise to images
        images_noisy = gaussian_noise(images)

        # Save noisy images
        if j in img_nums:
            noisy = images_noisy[0, ...].cpu().numpy().transpose(1, 2, 0)
            noisy = cv2.resize(noisy, dsize=(visual_imagesize, visual_imagesize))
            noisy = (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy))
            noisy = (np.array(noisy) * 255).astype('uint8')
            cv2.imwrite(os.path.join(output_dirs['noisy'], f'noisy_{j}.jpg').replace('\\', '/'),
                        cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))

        # Compute classification accuracies
        for n in range(len(models)):
            output = models[n](images_noisy, target_layer=target_layer)
            classification_accuracy_trackers_2[n].update(accuracy(output, labels)[0])

        process = 0
        time = 0
        overlay_list_2 = {'A': [], 'B': []}
        previous_spike_time_list_2 = {'A': [], 'B': []}

        for l, (activation_A, activation_B) in enumerate(zip(model_A.module.saved_forward, model_B.module.saved_forward)):
            previous_spike_time_list_2['A'].append(activation_A)
            previous_spike_time_list_2['B'].append(activation_B)
            weights_2 = {'A': 0, 'B': 0}

            for k in previous_spike_time_list_2.keys():
                for prev_t in range(len(previous_spike_time_list_2[k])):
                    delta_t = time - previous_spike_time_list_2[k][prev_t] * prev_t
                    weights_2[k] += torch.exp(gamma * (-1) * delta_t)

            # Compute attacked SAMs
            weighted_activation_A = weights_2['A'].cuda() * activation_A
            weighted_activation_A = weighted_activation_A.data.cpu().numpy()
            overlay_A = getForwardCAM(weighted_activation_A)
            overlay_list_2['A'].append(overlay_A)

            weighted_activation_B = weights_2['B'].cuda() * activation_B
            weighted_activation_B = weighted_activation_B.data.cpu().numpy()
            overlay_B = getForwardCAM(weighted_activation_B)
            overlay_list_2['B'].append(overlay_B)

            # Save SAMs
            if j in img_nums:
                if process % 3 == 0:
                    for k in overlay_list_2.keys():
                        sam = (np.array(1. - overlay_list_2[k][-1]) * 255).astype('uint8')
                        sam = cv2.resize(sam, dsize=(visual_imagesize, visual_imagesize))
                        sam = cv2.applyColorMap(sam, cv2.COLORMAP_JET)
                        blended = cv2.addWeighted(noisy, 0.5, sam, 0.5, 0.0)
                        cv2.imwrite(
                            os.path.join(output_dirs[f'attacked_SAM_1{k}'], f'sam_{j}_{process // 3 + 1}.jpg').replace('\\', '/'),
                            cv2.cvtColor(sam, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(
                            os.path.join(f'attacked_SAM_1{k}', f'sam_overlay_{j}_{process // 3 + 1}.jpg').replace('\\', '/'),
                            cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

            process += 1
            time += 1

        for n, k in enumerate(overlay_list_2.keys()):
            output_path = os.path.join(output_dirs[f'attacked_minerror_SAM_1{k}'], paths).replace('\\','/')
            grayscale_cam = load_target_image(os.path.join(output_dirs['grayscale_gradcam'], paths).replace('\\','/'))
            error, index = localization_error(overlay_list_2[k], grayscale_cam, save_image=output_path)
            localization_trackers_2[n].update(error, paths, index)

        img_idx += 1

    for n, k in enumerate(overlay_list_2.keys()):
        localization_trackers_2[n].export(os.path.join(output_dirs[f'attacked_minerror_SAM_1{k}'], 'localization_error.csv').replace('\\','/'))
        localization_trackers_2[n].print_output()
        classification_accuracy_trackers_2[n].print_output()

    # --------------------------------------------------
    # PASS #3: Add Gaussian noise to SAMs
    # --------------------------------------------------
    del val_loader
    val_loader = create_dataloader(dataset_path, len(img_nums), batch_size, num_workers)

    if flow == 'attack-sam' or flow == 'all':
        # Need to create a new dataloader for each clean SAM
        SAM_1A_dataloader = create_dataloader(output_dirs['clean_minerror_SAM_A'], len(img_nums), batch_size, num_workers, no_norm=True)
        SAM_1B_dataloader = create_dataloader(output_dirs['clean_minerror_SAM_B'], len(img_nums), batch_size, num_workers, no_norm=True)
        img_idx = 0

        classification_accuracy_tracker_3A = AverageMeter()
        classification_accuracy_tracker_3B = AverageMeter()
        classification_accuracy_trackers_3 = [classification_accuracy_tracker_3A, classification_accuracy_tracker_3B]
        localization_tracker_3A = LocalizationMeter()
        localization_tracker_3B = LocalizationMeter()
        localization_trackers_3 = [localization_tracker_3A, localization_tracker_3B]

        for j, (data, data_A, data_B) in enumerate(tqdm(zip(val_loader, SAM_1A_dataloader,SAM_1B_dataloader), total=len(SAM_1A_dataloader))):
            if args.limit_output is True:
                if j > np.max(img_nums):
                    continue
                if j not in img_nums:
                    continue

            for model in models:
                model.zero_grad()
                model.module.saved_grad = 0
                model.module.saved_forward = []

            images, labels, paths = data
            images_A, labels_A, paths_A = data_A
            images_B, labels_B, paths_B = data_B
            images = images.cuda()
            images_A = images_A.cuda()
            labels_A = labels_A.cuda()
            images_B = images_B.cuda()
            labels_B = labels_B.cuda()

            # Apply gaussian noise to images
            # images_noisy = [gaussian_noise(images)*images_A, gaussian_noise(images)*images_B]
            images_noisy = [images*(images_A),images*(images_B)]
            # images_noisy = [gaussian_noise(images,modifier=images_A), gaussian_noise(images,modifier=images_B)]
            labels = [labels_A, labels_B]
            paths = [paths_A[0], paths_B[0]]

            # Save noisy SAMs
            numpy_noisy = []
            if j in img_nums:
                for n in range(len(images_noisy)):
                    key = 'A' if n == 1 else 'B'
                    noisy = images_noisy[n][0, ...].cpu().numpy().transpose(1, 2, 0)
                    noisy = cv2.resize(noisy, dsize=(visual_imagesize, visual_imagesize))
                    noisy = (noisy - np.min(noisy)) / (np.max(noisy) - np.min(noisy))
                    noisy = (np.array(noisy) * 255).astype('uint8')
                    numpy_noisy.append(noisy)
                    cv2.imwrite(os.path.join(output_dirs[f'noisy_SAM_{key}'], f'noisy_{j}.jpg').replace('\\', '/'),
                                cv2.cvtColor(noisy, cv2.COLOR_RGB2BGR))

            # Compute classification accuracies
            for n in range(len(models)):
                output = models[n](images_noisy[n], target_layer=target_layer)
                classification_accuracy_trackers_3[n].update(accuracy(output, labels[n])[0])

            process = 0
            time = 0
            overlay_list_3 = {'A': [], 'B': []}
            previous_spike_time_list_3 = {'A': [], 'B': []}

            for l, (activation_A, activation_B) in enumerate(zip(model_A.module.saved_forward, model_B.module.saved_forward)):
                previous_spike_time_list_3['A'].append(activation_A)
                previous_spike_time_list_3['B'].append(activation_B)
                weights_3 = {'A': 0, 'B': 0}

                for k in previous_spike_time_list_3.keys():
                    for prev_t in range(len(previous_spike_time_list_3[k])):
                        delta_t = time - previous_spike_time_list_3[k][prev_t] * prev_t
                        weights_3[k] += torch.exp(gamma * (-1) * delta_t)

                # Compute attacked SAMs
                weighted_activation_A = weights_3['A'].cuda() * activation_A
                weighted_activation_A = weighted_activation_A.data.cpu().numpy()
                overlay_A = getForwardCAM(weighted_activation_A)
                overlay_list_3['A'].append(overlay_A)

                weighted_activation_B = weights_3['B'].cuda() * activation_B
                weighted_activation_B = weighted_activation_B.data.cpu().numpy()
                overlay_B = getForwardCAM(weighted_activation_B)
                overlay_list_3['B'].append(overlay_B)

                # Save SAMs
                if j in img_nums:
                    if process % 3 == 0:
                        for n, k in enumerate(overlay_list_3.keys()):
                            sam = (np.array(1. - overlay_list_3[k][-1]) * 255).astype('uint8')
                            sam = cv2.resize(sam, dsize=(visual_imagesize, visual_imagesize))
                            sam = cv2.applyColorMap(sam, cv2.COLORMAP_JET)
                            blended = cv2.addWeighted(numpy_noisy[n], 0.5, sam, 0.5, 0.0)
                            cv2.imwrite(
                                os.path.join(output_dirs[f'attacked_SAM_2{k}'], f'sam_{j}_{process // 3 + 1}.jpg').replace('\\', '/'),
                                cv2.cvtColor(sam, cv2.COLOR_RGB2BGR))
                            cv2.imwrite(
                                os.path.join(f'attacked_SAM_2{k}', f'sam_overlay_{j}_{process // 3 + 1}.jpg').replace('\\', '/'),
                                cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

                process += 1
                time += 1

            for n, k in enumerate(overlay_list_3.keys()):
                output_path = os.path.join(output_dirs[f'attacked_minerror_SAM_2{k}'], paths[n]).replace('\\','/')
                grayscale_cam = load_target_image(os.path.join(output_dirs['grayscale_gradcam'], paths[n]).replace('\\','/'))
                error, index = localization_error(overlay_list_3[k], grayscale_cam, save_image=output_path)
                localization_trackers_3[n].update(error, paths[n], index)

            img_idx += 1

        for n, k in enumerate(overlay_list_3.keys()):
            localization_trackers_3[n].export(os.path.join(output_dirs[f'attacked_minerror_SAM_2{k}'], 'localization_error.csv').replace('\\','/'))
            localization_trackers_3[n].print_output()
            classification_accuracy_trackers_3[n].print_output()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-pass flow for comparing spike generations in SNNs',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--flow', default='all', choices=['clean', 'attack-img', 'attack-sam', 'all'],
                        type=str, help='multipass flow option')
    parser.add_argument('--dataset_pth', default='./tiny-imagenet-200/val/images',
                        type=str, help='path for validation dataset')
    parser.add_argument('--output_root', default='./output_debug',
                        type=str, help='root for output SAM directories')
    # Run settings
    parser.add_argument('--batch_size', default=1, type=int, help='batch size should be 1')
    parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
    parser.add_argument('--limit_output', action='store_true',
                        help='limits the number of images to pass')
    parser.add_argument('--visual_imagesize', default=128, type=int)
    parser.add_argument('--dataset_size', default=100, type=int)
    # SNN settings
    parser.add_argument('--pretrainedmodel_pth', default='./pretrained/pretrained_tiny_t30.pth.tar',
                        type=str, help='path for pretrained model')
    parser.add_argument('--timesteps', default=30, type=float, help='timesteps')
    parser.add_argument('--leak_mem', default=0.99, type=float, help='leak_mem')
    parser.add_argument('--gamma', default=0.5, type=float, help='parameter for spike exponential function')
    parser.add_argument('--target_layer', default=6, choices=[4, 6, 8],
                        type=int, help='target_layer [4, 6, 8 (default)] is available')
    parser.add_argument('--T_min', default=2., type=float, help='min time interval (in ms), also the time per timestep')
    parser.add_argument('--N_max', default=5, type=int, help='max number of spikes in a burst')
    parser.add_argument('--seed', default=1, type=int, help='sets seed for Poisson generator')
    # GradCAM settings
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth CAM')
    parser.add_argument('--eigen_smooth', action='store_true',
                        help='Reduce noise by taking first PC of cam_weights*activations')
    # Attack settings
    parser.add_argument('--g_mu', default=0., type=float, help='mean for gaussian noise generation')
    parser.add_argument('--g_sigma', default=1., type=float, help='std for gaussian noise generation')

    args = parser.parse_args()
    main(args)






