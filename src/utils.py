import torch
import numpy as np
import pandas as pd
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.preds = []
        self.labels = []
        self.names = []

    def update(self, pred, label, name, n=1):
        val = accuracy(pred, label)[0]
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.preds.append(np.argmax(pred.cpu().numpy()))
        self.labels.append(label.cpu().numpy()[0])
        self.names.append(name)

    def export(self, output_file):
        df = pd.DataFrame()
        df['files'] = self.names
        df['predictions'] = self.preds
        df['labels'] = self.labels
        df.to_csv(output_file, index=False)
    
    def print_output(self):  
        print("Accuracy: %s" % self.avg.cpu().numpy()[0])

def accuracy(outp, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = outp.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def localization_error(sam_list, reference, save_image=None, eps=1e-2):
    errors = []

    # reference = cv2.resize(reference, dsize=(64, 64))

    for t in range(len(sam_list)):
        
        sam = sam_list[t]
        sam = cv2.resize(sam, dsize=(64, 64))

        sam[sam==1.] = 1.-eps
        sam[sam==0.] = eps

        # error = -np.sum(reference*np.log(sam) + (1.-reference)*np.log(1.-sam))
        error = np.mean((reference-sam)**2)
        # error = ssim(reference, sam)
        # error = mean_squared_error(reference,sam)
        errors.append(error)
    
    if save_image is not None:
        save_sam = sam_list[np.argmin(errors)]
        save_sam = (save_sam*255).astype('uint8')
        save_sam = cv2.resize(save_sam, dsize=(64, 64))
        cv2.imwrite(save_image,  save_sam)

        # TODO: save blended here
        # save_sam = cv2.applyColorMap(save_sam, cv2.COLORMAP_JET)
        #cv2.imwrite(save_image,  save_sam) #cv2.cvtColor(save_sam, cv2.COLOR_RGB2BGR))

    return np.min(errors), np.argmin(errors)

class LocalizationMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.errors = []
        self.names = []
        self.indices = []
        self.sum = 0
        self.count = 0

    def update(self, val, name, index):
        self.errors.append(val)
        self.names.append(name)
        self.indices.append(index)

        self.sum += val
        self.count += 1
    
    def export(self, output_file):
        df = pd.DataFrame()
        df['files'] = self.names
        df['errors'] = self.errors
        df['indices'] = self.indices
        df.to_csv(output_file, index=False)
    
    def print_output(self):
        print(f"Mean localization error {self.sum/self.count}")


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor, modifier=None):
        if modifier is not None:
            return tensor + (torch.randn(tensor.size()).cuda() * self.std + self.mean)*(modifier)
        else:
            return tensor + torch.randn(tensor.size()).cuda() * self.std + self.mean


def reshape_transform(tensor, height=64, width=64):
    # result = tensor[:, 1:, :].reshape(tensor.size(0),
    #                                   height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = tensor.transpose(1, 2).transpose(2, 3)
    return result


def getForwardCAM(feature_conv):
    cam = feature_conv.sum(axis =0).sum(axis =0)
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam) +1e-3)
    return cam_img