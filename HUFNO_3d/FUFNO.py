import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from einops import rearrange

from feedforward import FeedForward
from linear import WNLinear

import matplotlib.pyplot as plt
from utilities3 import *


import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

torch.manual_seed(123)
np.random.seed(123)

class SpectralConv2d(nn.Module):
    def __init__(self, in_dim, out_dim, modes_x, modes_y, modes_z, forecast_ff, backcast_ff,
                 fourier_weight, factor, ff_weight_norm,
                 n_ff_layers, layer_norm, use_fork, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.use_fork = use_fork

        self.fourier_weight = fourier_weight

        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        if use_fork:
            self.forecast_ff = forecast_ff
            if not self.forecast_ff:
                self.forecast_ff = FeedForward(
                    out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

        self.backcast_ff = backcast_ff
        if not self.backcast_ff:
            self.backcast_ff = FeedForward(
                out_dim, factor, ff_weight_norm, n_ff_layers, layer_norm, dropout)

    def forward(self, x):
        x = self.forward_fourier(x)

        b = self.backcast_ff(x)
        f = self.forecast_ff(x) if self.use_fork else None
        return b, f

    def forward_fourier(self, x):
        x = rearrange(x, 'b s1 s2 s3 i -> b i s1 s2 s3')

        B, I, S1, S2, S3 = x.shape

        x_ftz = torch.fft.rfft(x, dim=-1, norm='ortho')

        out_ft = x_ftz.new_zeros(B, I, S1, S2, S3 // 2 + 1)

        out_ft[:, :, :, :, :self.modes_z] = torch.einsum(
            "bixyz,ioz->boxyz",
            x_ftz[:, :, :, :, :self.modes_z],
            torch.view_as_complex(self.fourier_weight[2]))

        xz = torch.fft.irfft(out_ft, n=S3, dim=-1, norm='ortho')

        xy = 0.0
    
        x_ftx = torch.fft.rfft(x, dim=-3, norm='ortho')

        out_ft = x_ftx.new_zeros(B, I, S1 // 2 + 1, S2, S3)
        
        out_ft[:, :, :self.modes_x, :, :] = torch.einsum(
            "bixyz,iox->boxyz",
            x_ftx[:, :, :self.modes_x, :, :],
            torch.view_as_complex(self.fourier_weight[0]))

        xx = torch.fft.irfft(out_ft, n=S1, dim=-3, norm='ortho')

        x = xx + xy + xz

        x = rearrange(x, 'b i s1 s2 s3 -> b s1 s2 s3 i')

        return x

    
class U_net(nn.Module):  
    def __init__(self, input_channels, output_channels, kernel_size, dropout_rate):
        super(U_net, self).__init__()
        self.input_channels = input_channels
        self.conv1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate) 
        self.conv2 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv2_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        self.conv3 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=2, dropout_rate = dropout_rate)
        self.conv3_1 = self.conv(input_channels, output_channels, kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)
        
        self.deconv2 = self.deconv(input_channels, output_channels)
        self.deconv1 = self.deconv(input_channels*2, output_channels)
        self.deconv0 = self.deconv(input_channels*2, output_channels)
    
        self.output_layer = self.output(input_channels*2, output_channels, 
                                         kernel_size=kernel_size, stride=1, dropout_rate = dropout_rate)


    def forward(self, x): 
        batchsize, width = x.shape[0], x.shape[1]
        out_conv1 = self.conv1(x.reshape(batchsize, width, -1)) 
        out_conv2 = self.conv2_1(self.conv2(out_conv1))  
        out_conv3 = self.conv3_1(self.conv3(out_conv2))  

        out_deconv2 = self.deconv2(out_conv3) 
        concat2 = torch.cat((out_conv2, out_deconv2), 1) 
        out_deconv1 = self.deconv1(concat2) 
        concat1 = torch.cat((out_conv1, out_deconv1), 1) 
        out_deconv0 = self.deconv0(concat1)  
        concat0 = torch.cat((x, out_deconv0.reshape(batchsize, width, 16, 32, 33)), 1)  
        out = self.output_layer(concat0.reshape(concat0.shape[0], concat0.shape[1], -1)).reshape(batchsize, width, 16, 32, 33)  
        
        return out   

    def conv(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size,
                      stride=stride, padding=(kernel_size - 1) // 2, bias = False),
            nn.BatchNorm1d(output_channels),
            nn.LeakyReLU(0.1, inplace=True),  
            nn.Dropout(dropout_rate)
        )

    def deconv(self, input_channels, output_channels):
        return nn.Sequential(
            nn.ConvTranspose1d(input_channels, output_channels, kernel_size=4,
                                stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def output(self, input_channels, output_channels, kernel_size, stride, dropout_rate):
        return nn.Conv1d(input_channels, output_channels, kernel_size=kernel_size,
                          stride=stride, padding=(kernel_size - 1) // 2)

    
class FNOFactorizedMesh3D(nn.Module):
    def __init__(self, modes_x, modes_y, modes_z, width, input_dim, output_dim, times,
                 n_layers, share_weight=True, factor=1, ff_weight_norm=False, n_ff_layers=2,
                 layer_norm=False):
        super().__init__()
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        self.width = width
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.in_proj = WNLinear(self.width // 2 + 3, self.width, wnorm=ff_weight_norm)
        self.n_layers = n_layers

        self.fourier_weight = None
        if share_weight:
            self.fourier_weight = nn.ParameterList([])
            for n_modes in [modes_x, modes_y, modes_z]:
                weight = torch.FloatTensor(width, width, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)

        self.spectral_layers = nn.ModuleList([])
        self.unet_layers =  nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d(in_dim=width,
                                                       out_dim=width,
                                                       modes_x=modes_x,
                                                       modes_y=modes_y,
                                                       modes_z=modes_z,
                                                       forecast_ff=None,
                                                       backcast_ff=None,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       ff_weight_norm=ff_weight_norm,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm,
                                                       use_fork=False,
                                                       dropout=0.0))
            self.unet_layers.append(U_net(self.width, self.width, 3, 0))

        self.out = nn.Sequential(
            WNLinear(self.width, 128, wnorm=ff_weight_norm),
            WNLinear(128, output_dim, wnorm=ff_weight_norm))
        
        self.to_in = nn.Sequential(
            nn.Conv2d(input_dim, input_dim * 5, kernel_size=1, stride=1, padding=0, groups=input_dim),
            nn.GELU(),
            nn.Conv2d(input_dim * 5, self.width // 2, kernel_size=(times, 1), stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        b, nx, ny, nz, c, t = x.shape
        x = rearrange(x, 'b nx ny nz c t -> b c t (nx ny nz)')
        x = self.to_in(x)
        x = rearrange(x, 'b c 1 (nx ny nz) -> b nx ny nz c', nx=nx, ny=ny, nz=nz)
        
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)  
        x = self.in_proj(x) 
        x = x.permute(0, 4, 1, 2, 3)  
        x = x.permute(0, 2, 3, 4, 1)  

        for i in range(self.n_layers):
            layer = self.spectral_layers[i]
            u_net = self.unet_layers[i]
            b, _ = layer(x)
            
            x_u = x - b
            x_u = rearrange(x_u, 'b nx ny nz c -> b c nz nx ny')
            x_u = u_net(x_u)
            x_u = rearrange(x_u, 'b c nz nx ny -> b nx ny nz c')
            b = x_u + b
            
            x = x + b

        output = self.out(b)

        return output[..., None]

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat(
            [batchsize, 1, size_y, size_z, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat(
            [batchsize, size_x, 1, size_z, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat(
            [batchsize, size_x, size_y, 1, 1])
        return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


def get_energy_loss(x, y, norm, tol=1.):
    velocity_old = x[...,-1]
    velocity_new = velocity_old + y
    delta_energy = torch.sum(velocity_new**2, dim=(1,2,3)) - torch.sum(velocity_old**2, dim=(1,2,3))
    
    max_regular = delta_energy > tol*norm['delta_energy_max']
    min_regular = delta_energy < tol*norm['delta_energy_min']
    
    energy_loss = torch.mean((torch.abs(delta_energy-tol*norm['delta_energy_max']) * max_regular)) + \
              torch.mean((torch.abs(delta_energy-tol*norm['delta_energy_min']) * min_regular))
    return energy_loss




device = torch.device("cuda")

modes1 = 5
modes2 = 5
modes3 = 5
width = 80
epochs = 100
learning_rate = 0.001
weight_decay_value = 1e-11
nlayer = 5


batch_size = 4
scheduler_step = 10
scheduler_gamma = 0.5  

print(epochs, learning_rate, scheduler_step, scheduler_gamma)


runtime = np.zeros(2, )
t1 = default_timer()



vor_data = np.load('../../data/2D_hill_re700_400d200_uniform_grid_21x400x32x33x16x3_shape_a1.0.npy') 

vor_data = vor_data[0:20,...]

vor_data = torch.from_numpy(vor_data) 

input_list = []
output_list = []

for j in range(vor_data.shape[0]):
    for i in range(vor_data.shape[1]-5):
        input_list.append(vor_data[j,i:i+5,...])
        output_6m5 = (vor_data[j,i+5,...]-vor_data[j,i+4,...])
        output_list.append(output_6m5) 
     
input_set = torch.stack(input_list) 
output_set = torch.stack(output_list) 
input_set = input_set.permute(0,2,3,4,5,1) 


full_set = torch.utils.data.TensorDataset(input_set, output_set)
train_dataset, test_dataset = torch.utils.data.random_split(full_set, [int(0.8*len(full_set)), 
                                                                       len(full_set)-int(0.8*len(full_set))])

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=True)

model = FNOFactorizedMesh3D(modes1, modes2, modes3, width, 3, 3, 5, nlayer).to(device)



print(count_params(model))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

mse_train = []
mse_test = []
eloss_test = []


myloss = LpLoss()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    for xx, yy in train_loader:
        
        xx = xx.to(device)
        yy = yy.to(device)
        im = model(xx).to(device)
        
        train_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    mse_train.append(train_loss.item())
        

    ii=0
    eloss_val=0.0
    tloss_val=0.0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            im = model(xx).to(device)
            test_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))
            test_eloss = get_energy_loss(xx[...,0:3,:], im[...,0:3,0], norm, 1.)
            ii=ii+1
            eloss_val=eloss_val+test_eloss.item()
            tloss_val=tloss_val+test_loss.item()
        eloss_val=eloss_val/ii
        tloss_val=tloss_val/ii
        mse_test.append(tloss_val)
        eloss_test.append(eloss_val)
      

    t2 = default_timer()
    
    
    print(ep, "%.2f" % (t2 - t1), 'train_loss: {:.4f}'.format(train_loss.item()), 
          'test_loss: {:.4f}'.format(tloss_val))

    torch.save(model.state_dict(), 'NN_parameters_%04d.pth' % ep) 

MSE_save=np.dstack((mse_train,mse_test,eloss_test)).squeeze()
np.savetxt('./losses.dat',MSE_save,fmt="%16.7f")


