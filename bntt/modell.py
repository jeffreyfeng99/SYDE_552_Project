import torch
import torch.nn as nn
import torch.nn.functional as F
import sys




class Surrogate_BP_Function(torch.autograd.Function):


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
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)
        return grad


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
        self.ISI_countdown = self.ISI.clone().cuda()

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
        
        spike_train = torch.where(self.ISI_countdown <= 0, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())
        
        self.spike_counts = self.spike_counts + spike_train

        viable = torch.where(self.spike_counts < self.n_spikes_map, torch.tensor(1.0).cuda(), torch.tensor(0.0).cuda())

        self.ISI_countdown = torch.where(spike_train == 1.0, self.ISI, self.ISI_countdown)

        return torch.mul(spike_train, viable)


class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32,  num_cls=10):
        super(SNN_VGG9_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps

        print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)


        self.fc1 = nn.Linear((self.img_size//8)*(self.img_size//8)*256, 1024, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(1024, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt_fc]
        self.pool_list = [False, self.pool1, False, self.pool2, False, False, self.pool3]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size//2, self.img_size//2).cuda()
        mem_conv5 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv6 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size//4, self.img_size//4).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        
        for t in range(self.num_steps):

            spike_inp = PoissonGen(inp)
            out_prev = spike_inp

            for i in range(len(self.conv_list)):
                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_thr = (mem_conv_list[i] / self.conv_list[i].threshold) - 1.0
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst[mem_thr > 0] = self.conv_list[i].threshold
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_thr = (mem_fc1 / self.fc1.threshold) - 1.0
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst[mem_thr > 0] = self.fc1.threshold
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)

        out_voltage = mem_fc2 / self.num_steps


        return out_voltage


class SNN_VGG11_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=0.95, img_size=32,  num_cls=10, alif=False,
                        leak_thresh=0.99, delta_thresh=0.01, spike_code='poisson', T_max=30.0, T_min=2.0, N_max=5,):
        super(SNN_VGG11_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        self.alif = alif
        self.leak_thresh = leak_thresh
        self.delta_thresh = delta_thresh
        self.spike_code = spike_code
        self.T_max = T_max
        self.T_min = T_min
        self.N_max = N_max

        print (">>>>>>>>>>>>>>>>> VGG11 >>>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

        affine_flag = True
        bias_flag = False




        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.ModuleList([nn.BatchNorm2d(64, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.ModuleList([nn.BatchNorm2d(128, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.ModuleList([nn.BatchNorm2d(256, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool3 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool4 = nn.AvgPool2d(kernel_size=2)

        self.conv7 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt8 = nn.ModuleList([nn.BatchNorm2d(512, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.pool5 = nn.AdaptiveAvgPool2d((1,1))


        self.fc1 = nn.Linear(512, 4096, bias=bias_flag)
        self.bntt_fc = nn.ModuleList([nn.BatchNorm1d(4096, eps=1e-4, momentum=0.1, affine=affine_flag) for i in range(self.batch_num)])
        self.fc2 = nn.Linear(4096, self.num_cls, bias=bias_flag)

        self.conv_list = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]
        self.bntt_list = [self.bntt1, self.bntt2, self.bntt3, self.bntt4, self.bntt5, self.bntt6, self.bntt7, self.bntt8, self.bntt_fc]
        self.pool_list = [self.pool1, self.pool2, False, self.pool3, False, self.pool4, False, self.pool5]

        # Turn off bias of BNTT
        for bn_list in self.bntt_list:
            for bn_temp in bn_list:
                bn_temp.bias = None


        # Initialize the firing thresholds of all the layers
        for m in self.modules():
            if (isinstance(m, nn.Conv2d)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)
            elif (isinstance(m, nn.Linear)):
                m.threshold = 1.0
                torch.nn.init.xavier_uniform_(m.weight, gain=2)




    def forward(self, inp):

        batch_size = inp.size(0)
        h, w = inp.size(2) ,inp.size(3)

        mem_conv1 = torch.zeros(batch_size, 64, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size // 2, self.img_size // 2).cuda()
        mem_conv3 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv4 = torch.zeros(batch_size, 256, self.img_size // 4, self.img_size // 4).cuda()
        mem_conv5 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv6 = torch.zeros(batch_size, 512, self.img_size // 8, self.img_size // 8).cuda()
        mem_conv7 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv8 = torch.zeros(batch_size, 512, self.img_size // 16, self.img_size // 16).cuda()
        mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7, mem_conv8]

        mem_fc1 = torch.zeros(batch_size, 4096).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()

        thresh_conv1 = torch.ones(batch_size, 64, h, w).cuda()
        thresh_conv2 = torch.ones(batch_size, 128, h//2, w//2).cuda()
        thresh_conv3 = torch.ones(batch_size, 256, h//4, w//4).cuda()
        thresh_conv4 = torch.ones(batch_size, 256, h//4, w//4).cuda()
        thresh_conv5 = torch.ones(batch_size, 512, h//8, w//8).cuda()
        thresh_conv6 = torch.ones(batch_size, 512, h//8, w//8).cuda()
        thresh_conv7 = torch.ones(batch_size, 512, h // 16, w// 16).cuda()
        thresh_conv8 = torch.ones(batch_size, 512, h// 16, w// 16).cuda()
        thresh_fc1 = torch.ones(batch_size, 4096).cuda()
        thresh_fc2 = torch.ones(batch_size, 200).cuda()
        thresh_conv_list = [thresh_conv1, thresh_conv2, thresh_conv3, thresh_conv4, thresh_conv5, thresh_conv6, thresh_conv7, thresh_conv8]

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

            for i in range(len(self.conv_list)):
                
                mem_thr = (mem_conv_list[i] / thresh_conv_list[i]) - thresh_conv_list[i]
                out = self.spike_fn(mem_thr)
                rst = torch.zeros_like(mem_conv_list[i]).cuda()
                rst = torch.where(mem_thr>0,thresh_conv_list[i], rst)
                # rst[mem_thr > 0] = self.conv_list[i].threshold

                if self.alif:
                    thresh_rst = torch.zeros_like(mem_conv_list[i]).cuda()
                    thresh_rst[mem_thr>0] = self.delta_thresh
                    thresh_conv_list[i] = self.leak_thresh * thresh_conv_list[i] + thresh_rst

                mem_conv_list[i] = self.leak_mem * mem_conv_list[i] + self.bntt_list[i][t](self.conv_list[i](out_prev))
                mem_conv_list[i] = mem_conv_list[i] - rst
                out_prev = out.clone()


                if self.pool_list[i] is not False:
                    out = self.pool_list[i](out_prev)
                    out_prev = out.clone()


            out_prev = out_prev.reshape(batch_size, -1)

            mem_thr = (mem_fc1 / thresh_fc1) - thresh_fc1
            out = self.spike_fn(mem_thr)
            rst = torch.zeros_like(mem_fc1).cuda()
            rst = torch.where(mem_thr>0,thresh_fc1, rst)
            # rst[mem_thr > 0] = self.fc1.threshold

            if self.alif:
                thresh_rst = torch.zeros_like(mem_fc1).cuda()
                thresh_rst[mem_thr>0] = self.delta_thresh
                thresh_fc1 = self.leak_thresh * thresh_fc1 + thresh_rst

            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc[t](self.fc1(out_prev))
            mem_fc1 = mem_fc1 - rst
            out_prev = out.clone()

            # accumulate voltage in the last layer
            mem_fc2 = mem_fc2 + self.fc2(out_prev)


        out_voltage = mem_fc2 / self.num_steps

        return out_voltage



