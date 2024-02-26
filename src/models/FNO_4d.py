import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from layers.Conv4d import Conv4d
from layers.BatchNorm4d import BatchNorm4d

from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

"""
adapted from FNO code from https://github.com/khassibi/fourier-neural-operator/blob/main/fourier_3d.py 
4D spectral convolution corresponds to (x,y,z,t) dimension input. 
"""


class SpectralConv4D(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4D, self).__init__()
        
        """
        4D Fourier Layer, FFT -> linear transform -> IFFT 
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 
        self.modes2 = modes2 
        self.modes3 = modes3 
        self.modes4 = modes4 
        
        """
        linear transform weights
        """
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(self.scale * torch.rand(self.in_channels, self.out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))


    def complex_mul4d(self, input, weights):
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights) 
    
    
    
    """
    4D has the following combination of indices 
    [:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4]
    [:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4]
    [:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4]
    [:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4]
    [:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4]
    [:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4]
    [:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4]
    [:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4]
    """
    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=[-4,-3,-2,-1])
        
        out_ft = torch.zeros(batchsize, self.out_channels, 
                             x.size(-4), x.size(-3), x.size(-2),x.size(-1)//2+1, dtype=torch.cfloat)
        
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights2)
        
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4], self.weights3)   
        
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4], self.weights4)
        
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights5)
        
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights6)
        
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4], self.weights7)
        
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4] = \
            self.complex_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4], self.weights8)
        
        x = torch.fft.ifftn(out_ft, s = (x.size(-4),x.size(-3),x.size(-2), x.size(-1)))
        return x
    

class FNO4d(nn.Module):
    def __init__(self, in_channels, modes1, modes2, modes3, modes4, width):
        super(FNO4d, self).__init__()
        """
        4 layers of Fourier layer
        1. Lift the input to desired channel dimension by self.fc0 
        2. 5 layers of integral operators 
        3. project from channel space to output space by self.fc1 and self.fc2 
        
        input: 
        input shape: (batchsize, x=10, y=10, z=10, t=, c=)
        output shape: (batchsize, x=10, y=10, z=10, y=, c=)
        """
        
        self.modes1 = modes1 
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.width = width
        self.padding = 6 
        self.in_channels = in_channels
        
        self.fc0 = nn.Linear(self.in_channels, self.width)
        self.conv0 = SpectralConv4D(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv1 = SpectralConv4D(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv2 = SpectralConv4D(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        self.conv3 = SpectralConv4D(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
        
        self.w0 = Conv4d(self.width, self.width, 1)    
        self.w1 = Conv4d(self.width, self.width, 1)
        self.w2 = Conv4d(self.width, self.width, 1)
        self.w3 = Conv4d(self.width, self.width, 1)
        self.bn0 = BatchNorm4d(self.width)        
        self.bn1 = BatchNorm4d(self.width)
        self.bn2 = BatchNorm4d(self.width)
        self.bn3 = BatchNorm4d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    
    """
    get grid helper
    """
    def get_grid(self, shape, device):
        batchsize, size_x, size_y, size_z, size_t = shape[0], shape[1], shape[2], shape[3], shape[4]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, size_t, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1, 1, 1).repeat([batchsize, size_x, 1, size_z, size_t, 1])
        gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, size_z, 1, 1).repeat([batchsize, size_x, size_y, 1, size_t, 1])
        gridt = torch.tensor(np.linspace(0, 1, size_t), dtype=torch.float)
        gridt = gridt.reshape(1, 1, 1, 1, size_t, 1).repeat([batchsize, size_x, size_y, size_z, 1, 1])
        
        return torch.cat((gridx, gridy, gridz, gridt), dim=-1).to(device)



    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x) 
        x = x.permute(0,5,1,2,3,4)
        x = F.pad(x, [0, self.padding])
        
        x1 = self.conv0(x)
        x2 = self.w0(x) 
        x = x1 + x2 
        x = F.gelu(x) 
        
        x1 = self.conv1(x)
        x2 = self.w1(x) 
        x = x1 + x2 
        x = F.gelu(x) 
        
        x1 = self.conv2(x)
        x2 = self.w2(x) 
        x = x1 + x2 
        x = F.gelu(x) 
        
        x1 = self.conv3(x)
        x2 = self.w3(x) 
        x = x1 + x2 
        x = F.gelu(x) 
        
        x = x[..., :-self.padding]
        x = x.permute(0, 2, 3, 4, 5, 1)
        
        x = self.fc1(x) 
        x = F.gelu(x) 
        x = self.fc2(x) 
        return x 
    
    
# wait for the arrival of the dataset 
if __name__ == "__main__":
    pass