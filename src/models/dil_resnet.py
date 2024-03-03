import torch.nn as nn

"""
we only have a very coarse-grained grid of 10x10x10, therefore, instead of 7 dilated conv3d, we use [1,2,1] or [1,2,4,2,1] 
kernel size for cnn either 1 or 2 
"""

class BasicBlock(nn.Module):
    expansion = 1 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, downsample=None, dilations=[1,2,4,2,1], residual=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[0], dilation=dilations[0])
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[1], dilation=dilations[1])
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[2], dilation=dilations[2])
        self.bn3 = nn.BatchNorm3d(out_channels)
        
        self.conv4 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[3], dilation=dilations[3])
        self.bn4 = nn.BatchNorm3d(out_channels)
        
        self.conv5 = nn.Conv3d(in_channels=out_channels, out_channels=in_channels, stride=stride, kernel_size=kernel_size,
                               padding=dilations[-1], dilation=dilations[-1])
        self.bn5 = nn.BatchNorm3d(in_channels)
        
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
        out = self.bn1(out) 
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out) 

        out = self.conv3(out) 
        out = self.bn3(out) 
        out = self.relu(out)

        out = self.conv4(out) 
        out = self.bn4(out) 
        out = self.relu(out)

        out = self.conv5(out) 
        out = self.bn5(out) 
        out = self.relu(out)
         
        if self.downsample is not None:
            residual = self.downsample(out) 
        if self.residual:
            out += residual 
        out = self.relu(out)
        return out


"""
Dilated Residual ConvNet 
- key point is that downsample means we are doing dimension reduction when autoencoding 
"""
class DRN(nn.Module):
    def __init__(self, in_channels, out_channels_lst=[16,16], kernel_size=1, down_sample=False, use_bn=False):
        super(DRN, self).__init__() 
        self.in_channels = in_channels
        self.out_channels_lst = out_channels_lst
        self.down_sample = down_sample
        self.kernel_size = kernel_size
        self.use_bn = use_bn 

        self.encoder_channel_dim = out_channels_lst[0]
        self.block1_dim = out_channels_lst[0] if not down_sample else out_channels_lst[0] // 2
        self.decoder_channel_dim = out_channels_lst[0]

        self.encoder = nn.Conv3d(in_channels=in_channels, out_channels=self.encoder_channel_dim, kernel_size=kernel_size, stride=1)
        self.bn_enc = nn.BatchNorm3d(self.encoder_channel_dim)

        self.avg_pool = nn.AvgPool3d(kernel_size=kernel_size, stride=2)
        
        self.block1 = BasicBlock(in_channels=self.block1_dim, out_channels=self.out_channels_lst[1])
        
        self.decoder = nn.Conv3d(in_channels=self.decoder_channel_dim, out_channels=3, kernel_size=(kernel_size, kernel_size+1, kernel_size), stride=1)
        self.bn_dec = nn.BatchNorm3d(in_channels)
        self.decoder2 = nn.ConvTranspose3d(in_channels=self.decoder_channel_dim, out_channels=3, kernel_size=(kernel_size, kernel_size+1, kernel_size), stride=2)

        self.relu = nn.ReLU()

    """
    expect input of shape (batch_size, C_in, D,H,W)
    in our case: (18, 6, 10, 11, 10)
    """
    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        out = self.encoder(x)
        if self.use_bn:
            out = self.bn_enc(out)
        if self.down_sample:
            out = self.avg_pool(out)
            print(f"DRN downsample = {out.shape}")        
        out = self.relu(out) 
        out = self.block1(out)        
        if self.down_sample: 
            out = self.decoder2(out)
            print(f"DRN decoder ds = {out.shape}")
        else:
            out = self.decoder(out)
        if self.use_bn:
            out = self.bn_dec(out)
        out = self.relu(out)
        return out 