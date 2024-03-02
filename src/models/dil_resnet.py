import torch.nn as nn

"""
we only have a very coarse-grained grid of 10x10x10, therefore, instead of 7 dilated conv3d, we use [1,2,1] or [1,2,4,2,1] 
kernel size for cnn either 1 or 2 
"""

class BasicBlock(nn.Module):
    expansion = 1 
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None, dilations=[1,2,1], residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[0], dilation=dilations[0])
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[1], dilation=dilations[1])
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[0], dilation=dilations[0])
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.kernel_size = kernel_size
        self.stride = stride
        self.downsample = downsample
        self.residual = residual
    
    """
    input x: (batch_size, C_in, D, H, W)
    """
    def forward(self, x):
        residual = x 
        # print(f"block1 x = {x.shape}")
        out = self.conv1(x)
        # print(f"block1 conv1 = {out.shape}")

        out = self.bn1(x) 
        out = self.relu(x)

        out = self.conv2(x)
        # print(f"block1 conv2 = {out.shape}")

        out = self.bn2(x)
        out = self.relu(x) 

        out = self.conv3(x) 
        # print(f"block1 conv3 = {out.shape}")

        out = self.bn3(x) 
        out = self.relu(x) 

        if self.downsample is not None:
            residual = self.downsample(x) 
        if self.residual:
            out += residual 
        # print(f"block1 output shape = {out.shape}")
        out = self.relu(out)
        return out


"""
Dilated Residual ConvNet 
- key point is that downsample means we are doing dimension reduction when autoencoding 
"""
class DRN(nn.Module):
    def __init__(self, in_channels, out_channels_lst=[16,32], kernel_size=1, down_sample=False):
        super(DRN, self).__init__() 
        self.in_channels = in_channels
        self.out_channels_lst = out_channels_lst
        self.down_sample = down_sample
        self.kernel_size = kernel_size
        
        self.encoder = nn.Conv3d(in_channels=in_channels, out_channels=out_channels_lst[0], kernel_size=kernel_size, stride=1)
        self.bn_enc = nn.BatchNorm3d(out_channels_lst[0])

        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=2)
        
        self.block1 = BasicBlock(in_channels=out_channels_lst[0], out_channels=out_channels_lst[1], kernel_size=kernel_size)
        
        self.decoder = nn.Conv3d(in_channels=out_channels_lst[1], out_channels=3, kernel_size=kernel_size, stride=1)
        self.bn_dec = nn.BatchNorm3d(in_channels)

        self.decoder2 = nn.ConvTranspose3d(in_channels=out_channels_lst[1], out_channels=3, kernel_size=kernel_size, stride=2)

        self.relu = nn.ReLU()

    """
    expect input of shape (batch_size, C_in, D,H,W)
    in our case: (18, 3, 10, 10, 10)
    """
    def forward(self, x):

        # print(f"input shape = {x.shape}")

        x = x.permute(0, 4, 1, 2, 3)

        # print(f"DRN x = {x.shape}")

        # print(self.encoder)

        out = self.encoder(x)
        # print(f"DRN: encoderx = {out.shape}")

        if self.down_sample:
            out = self.avg_pool(out)
            # print(f"DRN downsample = {out.shape}")


        out = self.bn_enc(out)
        # print(f"DRN bn_enc = {out.shape}")
        
        out = self.relu(out) 

        out = self.block1(out)

        # print(f"DRN block1 = {out.shape}")
        
        if self.down_sample: 
            out = self.decoder2(out)
            # print(f"DRN decoder ds = {out.shape}")
        else:
            out = self.decoder(out)
            # print(f"DRN dec = {out.shape}")
        out = self.bn_dec(out)
        # print(f"DRN dec_bn = {out.shape}")
        out = self.relu(out)
        return out 