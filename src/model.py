"""
    Overall code flow is from Rathi et al. (2020) [https://github.com/nitin-rathi/hybrid-snn-conversion]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.autograd import Variable



# --------------------------------------------------
# Spiking neuron with piecewise-linear surrogate gradient
# --------------------------------------------------
class LinearSpike(torch.autograd.Function):
    gamma = 0.3  # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * LinearSpike.gamma * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


# --------------------------------------------------
# Spiking neuron with pass-through surrogate gradient
# --------------------------------------------------
class PassThruSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# Overwrite the naive spike function by differentiable spiking nonlinearity which implements a surrogate gradient
def init_spike_fn(grad_type):
    if (grad_type == 'Linear'):
        spike_fn = LinearSpike.apply
    elif (grad_type == 'PassThru'):
        spike_fn = PassThruSpike.apply
    else:
        sys.exit("Unknown gradient type '{}'".format(grad_type))
    return spike_fn


def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()

    # inp = (inp-torch.min(inp))/(torch.max(inp)-torch.min(inp))

    # return torch.le(rand_inp, inp).float()

    return torch.mul(torch.le(rand_inp * rescale_fac, torch.abs(inp)).float(), torch.sign(inp))


class BurstGen():
    def __init__(self, inp, n_timesteps=30, T_max=30.0, T_min=2.0, N_max=5):
        self.inp = inp
        self.norm_inp = self._normalize_inp()
        self.n_timesteps = n_timesteps
        self.T_max = T_max # ms
        self.T_min = T_min # ms, representing time per timestep
        self.N_max = N_max

        self.n_spikes_map = torch.ceil(self.norm_inp*self.N_max).cuda() # bxcxwxh

        self.spike_counts = torch.zeros_like(self.norm_inp).cuda() # bxcxwxh
        self.spike_map = torch.zeros_like(self.norm_inp).cuda() # bxcxwxh
        self.ISI = self._delta_ISI_timestep()
        self.ISI_countdown = self.ISI.clone()

    def _normalize_inp(self):
        return (self.inp - torch.min(self.inp)) / (torch.max(self.inp) - torch.min(self.inp))

    # def _calc_N_spikes(self, p):
    #     if p < 0. or p > 1.:
    #         raise ValueError(f'pixel intensity should be between [0., 1.] but is {p}')
    #     return self.N_max*p
    #
    # def _n_spikes_map(self):
    #     self.n_spikes_map = self.norm_inp*self.N_max

    def _delta_ISI_timestep(self):
        ISI_default = torch.tensor(self.T_max).repeat(list(self.norm_inp.size())).cuda()
        
        ISI = torch.ceil(self.T_max - (self.T_max - self.T_min)*self.norm_inp).cuda()

        ISI = torch.where(self.n_spikes_map > 1, ISI, ISI_default)

        return torch.ceil(ISI / self.T_min).cuda()

    def spike_fn(self):

        self.ISI_countdown = self.ISI_countdown - 1.
        
        spike_train = torch.where(self.ISI_countdown <= 0, 1.0, 0.0)
        
        self.spike_counts = self.spike_counts + spike_train

        viable = torch.where(self.spike_counts < self.n_spikes_map, 1.0, 0.0)

        self.ISI_countdown = torch.where(spike_train == 1.0, self.ISI, self.ISI_countdown)

        return torch.mul(spike_train, viable)


class SNN_VGG11(nn.Module):
    def __init__(self, num_timestep=30, leak_mem=0.99, spike_code='poisson', T_max=30.0, T_min=2.0, N_max=5, alif=False,
                        leak_thresh=0.99, delta_thresh=0.02):
        super(SNN_VGG11, self).__init__()

        self.img_size = 64
        self.num_steps = num_timestep
        self.leak_mem = leak_mem
        self.spike_code = spike_code
        self.T_max = T_max
        self.T_min = T_min
        self.N_max = N_max
        self.batch_num = self.num_steps
        self.alif = alif
        self.leak_thresh = leak_thresh
        self.delta_thresh = delta_thresh

        affine_flag = True
        bias_flag = False

        # Instantiate the ConvSNN layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn1_list = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn2_list = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn3_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn4_list = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)
        
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn5_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn6_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)
        
        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn7_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bn8_list = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AvgPool2d(kernel_size=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(512, 4096, bias=bias_flag)
        self.bnfc_list = nn.ModuleList(
            [nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in
             range(self.batch_num)])
        self.fc2 = nn.Linear(4096, 10, bias=bias_flag)

        batchnormlist = [self.bn1_list, self.bn2_list, self.bn3_list, self.bn4_list, self.bn5_list,
                         self.bn6_list, self.bn7_list, self.bn8_list, self.bnfc_list]

        for bnlist in batchnormlist:
            for bnbn in bnlist:
                bnbn.bias = None

            # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0

        self.saved_forward= []

        # Instantiate differentiable spiking nonlinearity
        self.spike_fn = init_spike_fn('Linear')
        self.spike_pool = init_spike_fn('PassThru')


    def forward(self, inp, target_layer=2):
        outputList = []

        batch_size = inp.size(0)
        h, w = inp.size(2) ,inp.size(3)

        mem_conv1 = Variable(torch.zeros(batch_size, 64, h, w), requires_grad=True).cuda()
        mem_conv2 = Variable(torch.zeros(batch_size, 128, h//2, w//2).cuda(), requires_grad=True)
        mem_conv3 = Variable(torch.zeros(batch_size, 256, h//4, w//4).cuda(), requires_grad=True)
        mem_conv4 = Variable(torch.zeros(batch_size, 256, h//4, w//4).cuda(), requires_grad=True)
        mem_conv5 = Variable(torch.zeros(batch_size, 512, h//8, w//8).cuda(), requires_grad=True)
        mem_conv6 = Variable(torch.zeros(batch_size, 512, h//8, w//8).cuda(), requires_grad=True)
        mem_conv7 = Variable(torch.zeros(batch_size, 512, h // 16, w// 16).cuda(), requires_grad=True)
        mem_conv8 = Variable(torch.zeros(batch_size, 512, h// 16, w// 16).cuda(), requires_grad=True)

        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, 10).cuda()

        thresh_conv1 = torch.ones(batch_size, 64, h, w).cuda()
        thresh_conv2 = torch.ones(batch_size, 128, h//2, w//2).cuda()
        thresh_conv3 = torch.ones(batch_size, 256, h//4, w//4).cuda()
        thresh_conv4 = torch.ones(batch_size, 256, h//4, w//4).cuda()
        thresh_conv5 = torch.ones(batch_size, 512, h//8, w//8).cuda()
        thresh_conv6 = torch.ones(batch_size, 512, h//8, w//8).cuda()
        thresh_conv7 = torch.ones(batch_size, 512, h // 16, w// 16).cuda()
        thresh_conv8 = torch.ones(batch_size, 512, h// 16, w// 16).cuda()
        thresh_fc1 = torch.ones(batch_size, 4096).cuda()
        thresh_fc2 = torch.ones(batch_size, 10).cuda()

        # TODO need args
        spike_gen = None
        if self.spike_code == 'burst':
            spike_gen = BurstGen(inp, self.num_steps, T_max=self.T_max, T_min=self.T_min, N_max=self.N_max)

        for t in range(self.num_steps):
            if self.spike_code == 'poisson':
                spike_inp = PoissonGen(inp)
            elif self.spike_code == 'burst':
                spike_inp = spike_gen.spike_fn()
            else:
                raise ValueError(f'spike code {self.spike_code} unknown')
            out_prev = spike_inp

            # Compute the conv1 outputs
            mem_thr   = (mem_conv1/thresh_conv1) - thresh_conv1
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv1).cuda()
            rst = torch.where(mem_thr>0,thresh_conv1, rst)
            #rst[mem_thr>0] = thresh_conv1

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv1).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv1 = self.leak_thresh * thresh_conv1 + thresh_rst

            mem_conv1 = (self.leak_mem*mem_conv1 + self.bn1_list[int(t)](self.conv1(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the avgpool1 outputs
            out =  self.pool1(out_prev)
            out_prev = out.clone()

            # Compute the conv2 outputs
            mem_thr   = (mem_conv2/thresh_conv2) - thresh_conv2
            out       = self.spike_fn(mem_thr)
            rst       = torch.zeros_like(mem_conv2).cuda()
            rst = torch.where(mem_thr>0,thresh_conv2, rst)
            # rst[mem_thr>0] = thresh_conv2

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv2).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv2 = self.leak_thresh * thresh_conv2 + thresh_rst

            mem_conv2 = (self.leak_mem*mem_conv2 + self.bn2_list[int(t)](self.conv2(out_prev)) -rst)
            out_prev  = out.clone()

            # Compute the avgpool2 outputs
            out = self.pool2(out_prev)
            out_prev = out.clone()

            # Compute the conv3 outputs
            mem_thr = (mem_conv3 / thresh_conv3) - thresh_conv3
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv3).cuda()
            rst = torch.where(mem_thr>0,thresh_conv3, rst)
            # rst[mem_thr>0] = thresh_conv2

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv3).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv3 = self.leak_thresh * thresh_conv3 + thresh_rst

            mem_conv3 = (self.leak_mem * mem_conv3 + self.bn3_list[int(t)](self.conv3(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv4 outputs
            mem_thr = (mem_conv4 / thresh_conv4) - thresh_conv4
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv4).cuda()
            rst = torch.where(mem_thr>0,thresh_conv4, rst)
            # rst[mem_thr>0] = thresh_conv2

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv4).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv4 = self.leak_thresh * thresh_conv4 + thresh_rst

            mem_conv4 = (self.leak_mem * mem_conv4 + self.bn4_list[int(t)](self.conv4(out_prev)) - rst)
            out_prev = out.clone()

            if target_layer == 4:
                self.saved_forward.append(out_prev)

            # Compute the avgpool3 outputs
            out = self.pool3(out_prev)
            out_prev = out.clone()

            # Compute the conv5 outputs
            mem_thr = (mem_conv5 / thresh_conv5) - thresh_conv5
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv5).cuda()
            rst = torch.where(mem_thr>0,thresh_conv5, rst)
            # rst[mem_thr>0] = thresh_conv2

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv5).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv5 = self.leak_thresh * thresh_conv5 + thresh_rst
                
            mem_conv5 = (self.leak_mem * mem_conv5 + self.bn5_list[int(t)](self.conv5(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv6 outputs
            mem_thr = (mem_conv6 / thresh_conv6) - thresh_conv6
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv6).cuda()
            rst = torch.where(mem_thr>0,thresh_conv6, rst)
            # rst[mem_thr>0] = thresh_conv2

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv6).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv6 = self.leak_thresh * thresh_conv6 + thresh_rst
                
            mem_conv6 = (self.leak_mem * mem_conv6 + self.bn6_list[int(t)](self.conv6(out_prev)) - rst)
            out_prev = out.clone()

            if target_layer == 6:
                self.saved_forward.append(out_prev)


            # Compute the avgpool4 outputs
            out = self.pool4(out_prev)
            out_prev = out.clone()

            # Compute the conv7 outputs
            mem_thr = (mem_conv7 / thresh_conv7) - thresh_conv7
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv7).cuda()
            rst = torch.where(mem_thr>0,thresh_conv7, rst)
            # rst[mem_thr>0] = thresh_conv2

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv7).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv7 = self.leak_thresh * thresh_conv7 + thresh_rst

            mem_conv7 = (self.leak_mem * mem_conv7 + self.bn7_list[int(t)](self.conv7(out_prev)) - rst)
            out_prev = out.clone()

            # Compute the conv8 outputs
            mem_thr = (mem_conv8 / thresh_conv8) - thresh_conv8
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_conv8).cuda()
            rst = torch.where(mem_thr>0,thresh_conv8, rst)
            # rst[mem_thr>0] = thresh_conv2

            if self.alif:
                thresh_rst = torch.zeros_like(mem_conv8).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_conv8 = self.leak_thresh * thresh_conv8 + thresh_rst
                
            mem_conv8 = (self.leak_mem * mem_conv8 + self.bn8_list[int(t)](self.conv8(out_prev)) - rst)
            out_prev = out.clone()

            if target_layer == 8:
                self.saved_forward.append(out_prev)

            # Compute the avgpool5 outputs
            out = self.avg_pool(out_prev)
            out_prev = out.clone()
            out_prev = out_prev.reshape(batch_size, -1)

            # compute fc1
            mem_thr = (mem_fc1 / thresh_fc1) - thresh_fc1
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst = torch.where(mem_thr>0,thresh_fc1, rst)
            # rst[mem_thr>0] = thresh_conv2


            if self.alif:
                thresh_rst = torch.zeros_like(mem_fc1).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_fc1 = self.leak_thresh * thresh_fc1 + thresh_rst
                
            mem_fc1 = (self.leak_mem * mem_fc1 + self.bnfc_list[int(t)](self.fc1(out_prev)) - rst)

            out_prev = out.clone()
            mem_fc2 = (1 * mem_fc2 + self.fc2(out_prev))
            out_voltage_tmp = (mem_fc2) / (t+1e-3)
            outputList.append(out_voltage_tmp)

        out_voltage  = mem_fc2
        out_voltage = (out_voltage) / self.num_steps

        return out_voltage


