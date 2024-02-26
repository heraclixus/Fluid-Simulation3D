import torch 
import torch.nn as nn
import math
import torch.nn.functional as F

"""
applies batchnorm to 6D input (a minibatch of 4D inputs with additional channel dimensino)

Example::
    m = BatchNorm4d(100) # channel size = 100 
    input = torch.randn(batch_size, C_in, X, Y, Z, T) 
    output = m(input)
"""
class BatchNorm4d(nn.Module):    
    def __init__(self, num_features, eps=1e-05, 
                 momentum=0.1, affine=True, track_running_stats=True, 
                 device=None, dtype=None):
        super(BatchNorm4d).__init__() 
        
        shape = (1, num_features, 1, 1, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.ones(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
        
    def batch_norm(self, X, gamma, beta, moving_mean, moving_var, eps=1e-05, momentum=0.1):
        if not torch.is_grad_enabled():
            X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            assert len(X.shape) == 6
            mean = X.mean(dim=(0,2,3,4,5), keepdim=True)
            var = ((X-mean) ** 2).mean(dim=(0,2,3,4,5), keepdim=True)
            X_hat = (X-mean) / torch.sqrt(var + eps)
            moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
            moving_var = (1.0 - momentum) * moving_var + momentum * var
        Y  = gamma * X_hat + beta
        return Y, moving_mean.data, moving_var.data
        
    def forward(self, X):
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var = self.batch_norm(X, self.gamma, self.beta, self.moving_mean,self.moving_var)
        return Y 
        