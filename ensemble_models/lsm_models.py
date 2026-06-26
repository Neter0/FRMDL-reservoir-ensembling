import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

import snntorch as snn

from RF import RFSynaptic

class LSM(nn.Module):
    def __init__(self, N, in_sz, Win, Wlsm, alpha=0.9, beta=0.9, th=20, neuron_dynamics = 'SLIF', 
                 T_min=3, T_max=100, rho=0.95, bias = True):
        super().__init__()
        self.fc1 = nn.Linear(in_sz, N, bias=bias)
        self.fc1.weight = nn.Parameter(torch.from_numpy(Win))
        self.neuron_dynamics = neuron_dynamics
        if self.neuron_dynamics == 'SLIF':
            self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        elif self.neuron_dynamics == 'LIF':
            self.lsm = snn.RLeaky(beta=beta, all_to_all=True, linear_features=N, threshold=th)
        elif self.neuron_dynamics == 'IF':
            self.lsm = snn.RLeaky(beta=1, all_to_all=True, linear_features=N, threshold=th)
        elif self.neuron_dynamics == 'RF':
            self.lsm = RFSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th, 
                                      T_min=T_min, T_max=T_max, rho=rho)
        self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(Wlsm))
        if self.lsm.recurrent.bias is not None and bias == False:
            nn.init.zeros_(self.lsm.recurrent.bias)
    def forward(self, x):
        num_steps = x.size(0)
        spk_rec = []
        if self.neuron_dynamics == 'SLIF':
            spk, syn, mem = self.lsm.init_rsynaptic()
            for step in range(num_steps):
                curr = self.fc1(x[step])
                spk, syn, mem = self.lsm(curr, spk, syn, mem)
                spk_rec.append(spk)
        elif self.neuron_dynamics == 'RF':
            spk, syn, mem, refrac = self.lsm.init_rsynaptic()
            for step in range(num_steps):
                curr = self.fc1(x[step])
                spk, syn, mem, refrac = self.lsm(curr, spk, syn, mem, refrac)
                spk_rec.append(spk)
        elif self.neuron_dynamics == 'IF' or self.neuron_dynamics == 'LIF':
            spk, mem = self.lsm.init_rleaky()
            for step in range(num_steps):
                curr = self.fc1(x[step])
                spk, mem = self.lsm(curr, spk, mem)
                spk_rec.append(spk)
        spk_rec_out = torch.stack(spk_rec)
        return spk_rec_out

class LSM_partition(nn.Module):
    def __init__(self, N, in_sz, Wins, Wlsm, num_partitions, alpha=0.9, beta=0.9, th=20, 
                 neuron_dynamics = 'SLIF', T_min=3, T_max=100, rho=0.95, bias=True):
        super().__init__()
        self.Wins = Wins
        self.num_partitions = num_partitions
        self.fc1 = nn.Linear(in_sz, N, bias=bias)
        self.fc1.weight = nn.Parameter(torch.from_numpy(Wins[0]))
        self.neuron_dynamics = neuron_dynamics
        if self.neuron_dynamics == 'SLIF':
            self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        elif self.neuron_dynamics == 'LIF':
            self.lsm = snn.RLeaky(beta=beta, all_to_all=True, linear_features=N, threshold=th)
        elif self.neuron_dynamics == 'IF':
            self.lsm = snn.RLeaky(beta=1, all_to_all=True, linear_features=N, threshold=th)
        elif self.neuron_dynamics == 'RF':
            self.lsm = RFSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th, 
                                      T_min=T_min, T_max=T_max, rho=rho)
        self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(Wlsm))
        if self.lsm.recurrent.bias is not None and bias == False:
            nn.init.zeros_(self.lsm.recurrent.bias)
    def forward(self, x, device):
        num_steps = x.size(0)
        spk_rec = []
        partition_steps = num_steps//self.num_partitions
        Win_ind = 0
        if self.neuron_dynamics == 'SLIF':
            spk, syn, mem = self.lsm.init_rsynaptic()
            for step in range(num_steps):
                if (step%partition_steps==0):
                    self.fc1.weight = nn.Parameter(torch.from_numpy(self.Wins[Win_ind]).to(device))
                    Win_ind = (Win_ind + 1)%self.num_partitions
                curr = self.fc1(x[step])
                spk, syn, mem = self.lsm(curr, spk, syn, mem)
                spk_rec.append(spk)
        elif self.neuron_dynamics == 'RF':
            spk, syn, mem, refrac = self.lsm.init_rsynaptic()
            for step in range(num_steps):
                if (step%partition_steps==0):
                    self.fc1.weight = nn.Parameter(torch.from_numpy(self.Wins[Win_ind]).to(device))
                    Win_ind = (Win_ind + 1)%self.num_partitions
                curr = self.fc1(x[step])
                spk, syn, mem, refrac = self.lsm(curr, spk, syn, mem, refrac)
                spk_rec.append(spk)
        elif self.neuron_dynamics == 'IF' or self.neuron_dynamics == 'LIF':
            spk, mem = self.lsm.init_rleaky()
            for step in range(num_steps):
                if (step%partition_steps==0):
                    self.fc1.weight = nn.Parameter(torch.from_numpy(self.Wins[Win_ind]).to(device))
                    Win_ind = (Win_ind + 1)%self.num_partitions
                curr = self.fc1(x[step])
                spk, mem = self.lsm(curr, spk, mem)
                spk_rec.append(spk)
        spk_rec_out = torch.stack(spk_rec)
        return spk_rec_out

class LSM_partition_cross_partition_inh(nn.Module):
    def __init__(self, N, in_sz, Wins, Wlsm, Wlsm_long, num_partitions, alpha=0.9, beta=0.9, th=20):
        super().__init__()
        self.Wins = Wins
        self.num_partitions = num_partitions
        self.fc1 = nn.Linear(in_sz, N)
        self.fc1.weight = nn.Parameter(torch.from_numpy(Wins[0]))
        self.long_inh = nn.Linear(N, N)
        self.long_inh.weight = nn.Parameter(torch.from_numpy(Wlsm_long))
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(Wlsm))
    def forward(self, x, device):
        num_steps = x.size(0)
        spk, syn, mem = self.lsm.init_rsynaptic()
        spk_rec = []
        partition_steps = num_steps//self.num_partitions
        Win_ind = 0
        for step in range(num_steps):
            if (step%partition_steps==0):
                self.fc1.weight = nn.Parameter(torch.from_numpy(self.Wins[Win_ind]).to(device))
                Win_ind = (Win_ind + 1)%self.num_partitions
            
            if (step>partition_steps):
                curr = self.fc1(x[step]) + self.long_inh(spk_rec[step-partition_steps])
            else:
                curr = self.fc1(x[step])
            spk, syn, mem = self.lsm(curr, spk, syn, mem)
            spk_rec.append(spk)
        spk_rec_out = torch.stack(spk_rec)
        return spk_rec_out

class Gabor_LSM_partition(nn.Module):
    def __init__(self, N, in_sz, Wins, Wlsm, Gabor_filters, stride, num_partitions, alpha=0.9, beta=0.9, th=20):
        super().__init__()
        in_ch = Gabor_filters.shape[1]
        out_ch = Gabor_filters.shape[0]
        k_sz = (Gabor_filters.shape[2], Gabor_filters.shape[3])
        self.gabor_filter = nn.Conv2d(in_ch, out_ch, k_sz, stride=stride, padding=0, dilation=1, groups=1, bias=False)
        self.gabor_filter.weight = nn.Parameter(Gabor_filters)
        self.Wins = Wins
        self.num_partitions = num_partitions
        self.fc1 = nn.Linear(in_sz, N)
        self.fc1.weight = nn.Parameter(torch.from_numpy(Wins[0]))
        self.lsm = snn.RSynaptic(alpha=alpha, beta=beta, all_to_all=True, linear_features=N, threshold=th)
        self.lsm.recurrent.weight = nn.Parameter(torch.from_numpy(Wlsm))
    def forward(self, x, device):
        num_steps = x.size(0)
        spk, syn, mem = self.lsm.init_rsynaptic()
        spk_rec = []
        partition_steps = num_steps//self.num_partitions
        Win_ind = 0
        for step in range(num_steps):
            if (step%partition_steps==0):
                self.fc1.weight = nn.Parameter(torch.from_numpy(self.Wins[Win_ind]).to(device))
                Win_ind = (Win_ind + 1)%self.num_partitions
            #gabor_out = nn.functional.conv2d(x[step], self.G_filters, stride=self.conv_stride, padding=0)
            gabor_out = self.gabor_filter(x[step])
            gabor_out_flat = torch.reshape(gabor_out, (gabor_out.shape[0], -1))
            curr = self.fc1(gabor_out_flat)
            spk, syn, mem = self.lsm(curr, spk, syn, mem)
            spk_rec.append(spk)
        spk_rec_out = torch.stack(spk_rec)
        return spk_rec_out
