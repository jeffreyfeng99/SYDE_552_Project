import torch
import numpy as np
import pandas as pd

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

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):  
        return "Accuracy: %s" % self.avg.cpu().numpy()[0]

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

def localization_error(sam_list, reference, eps=1e-8):
    errors = []
    for t in range(len(sam_list)):
        
        sam = sam_list[t]
        sam[sam==1.] = 1.-eps
        sam[sam==0.] = eps

        error = -np.sum(reference*np.log(sam) + (1.-reference)*np.log(1.-sam))
        errors.append(error)

    return np.min(errors), np.argmin(errors)

class LocalizationMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.errors = []
        self.names = []
        self.indices = []

    def update(self, val, name, index):
        self.errors.append(val)
        self.names.append(name)
        self.indices.append(index)
    
    def export(self, output_file):
        df = pd.DataFrame()
        df['files'] = self.names
        df['errors'] = self.errors
        df['indices'] = self.indices
        df.to_csv(output_file, index=False)
    
    def print_output(self):
        for error, name, index in zip(self.errors, self.names, self.indices):
            print("Localization error for %s: %s at %s" % (name, error, index))


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def getForwardCAM(feature_conv):
    cam = feature_conv.sum(axis =0).sum(axis =0)
    cam = cam - np.min(cam)
    cam_img = cam / (np.max(cam) +1e-3)
    return cam_img