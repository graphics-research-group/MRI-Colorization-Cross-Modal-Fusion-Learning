import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torchvision.ops
from torch import nn
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch
import torchvision.ops
from torch import nn
import numpy as np

class SqEx(nn.Module):

    def __init__(self, n_features, reduction=16):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 16)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class DeformableConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,
                                     2 * kernel_size[0] * kernel_size[1],
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     padding=self.padding,
                                     dilation=self.dilation,
                                     bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,
                                        1 * kernel_size[0] * kernel_size[1],
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      dilation=self.dilation,
                                      bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.regular_conv.weight,
                                          bias=self.regular_conv.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation)
        return x
    

class SplitResBlock(nn.Module):
    def __init__(self, in_features, deformable=False):
        super(SplitResBlock, self).__init__()
        conv = nn.Conv2d if deformable==False else DeformableConv2d 
        self.refPad1 = nn.ReflectionPad2d(1)
        self.conv1 = conv(in_features, in_features, 3)
        self.instNorm1= nn.InstanceNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.refPad2 = nn.ReflectionPad2d(1)
        self.conv2 = conv(in_features, in_features, 3)
        self.instNorm2 = nn.InstanceNorm2d(in_features)  
        self.sqex = SqEx(in_features)


    def forward(self, x):
        x_sqex = self.sqex(x)
        # x_out = self.refPad1(x)
        # x_out = self.conv1(x_out)
        # x_out = self.instNorm1(x_out)
        # x_out = self.relu(x_out)
        # x_out = self.refPad2(x_out)
        # x_out = self.conv2(x_out)
        # x_out = self.instNorm2(x_out)
        return x  + x_sqex #+ x_out
    

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        ## Different scales convs
        self.conv128 = nn.Sequential(nn.Conv2d(input_nc, 64, 3, 1, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(64, 128, 3, 1, 1), 
                      nn.ReLU(),
                      nn.Conv2d(128, 256, 3, 2, 1), 
                      nn.ReLU())

        self.conv64 = nn.Sequential(nn.Conv2d(input_nc, 64, 3, 1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 128, 3, 1, 1), 
                            nn.ReLU(),
                            nn.Conv2d(128, 256, 3, 1, 1), 
                            nn.ReLU())
        
        
        # Initial convolution block       
        self.refpad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, 128, 7)
        self.instNorm = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        model =[]

        # Downsampling
        in_features = 128
        out_features = in_features*2
        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.instNorm2 = nn.InstanceNorm2d(out_features)
        self.relu =  nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2

        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.instNorm3 = nn.InstanceNorm2d(out_features)
        self.relu =  nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2


        # Residual blocks
        for _ in range(n_residual_blocks):
            # model += [ResidualBlock(in_features)]
            model += [SplitResBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Sigmoid()] # change to tanh, sigmoid for hsv

        self.model = nn.Sequential(*model)
            
        self.up_64 = torch.nn.Upsample(size=None, scale_factor=0.25, mode='bilinear')
        self.up_128 = torch.nn.Upsample(size=None, scale_factor=0.5, mode='bilinear')


    def forward(self, x):
        x_128 = self.up_128(x)
        x_64 = self.up_64(x)
        x_out = self.refpad1(x)
        x_out = self.conv1(x_out)
        x_out = self.instNorm(x_out)
        x_out = self.relu(x_out)
        
        

        x_out = self.conv2(x_out)
        x_out = self.instNorm2(x_out)
        x_out = self.relu(x_out)
        

        x_out = self.conv3(x_out)
        x_out = self.instNorm3(x_out)
        x_out = self.relu(x_out) 
        
        x_out_128 = self.conv128(x_128)
        x_out_64 = self.conv64(x_64)
        
        x_out_multiscale = torch.cat([x_out_128, x_out_64], dim=1)
        
        x_out = x_out + x_out_multiscale        
        
              
        return self.model(x_out)
    

class GeneratorM2CM_C(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(GeneratorM2CM_C, self).__init__()

        ## Different scales convs
        self.conv128 = nn.Sequential(nn.Conv2d(input_nc, 64, 3, 1, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(64, 128, 3, 1, 1), 
                      nn.ReLU(),
                      nn.Conv2d(128, 256, 3, 2, 1), 
                      nn.ReLU())

        self.conv64 = nn.Sequential(nn.Conv2d(input_nc, 64, 3, 1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 128, 3, 1, 1), 
                            nn.ReLU(),
                            nn.Conv2d(128, 256, 3, 1, 1), 
                            nn.ReLU())
        
        
        # Initial convolution block       
        self.refpad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, 128, 7)
        self.instNorm = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        model =[]

        # Downsampling
        in_features = 128
        out_features = in_features*2
        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.instNorm2 = nn.InstanceNorm2d(out_features)
        self.relu =  nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2

        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.instNorm3 = nn.InstanceNorm2d(out_features)
        self.relu =  nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2

        in_features_c = out_features
        out_features_c = in_features_c*2


        # Residual blocks
        for _ in range(n_residual_blocks):
            # model += [ResidualBlock(in_features)]
            model += [SplitResBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Sigmoid()] # change to tanh, sigmoid for hsv

        self.model = nn.Sequential(*model)
        ### MRI to Cryo
        model_cryo=[]
        for _ in range(n_residual_blocks):
            # model += [ResidualBlock(in_features)]
            model_cryo += [SplitResBlock(in_features_c)]

        # Upsampling
        out_features_c = in_features_c//2
        for _ in range(2):
            model_cryo += [  nn.ConvTranspose2d(in_features_c, out_features_c, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features_c),
                        nn.ReLU(inplace=True) ]
            in_features_c = out_features_c
            out_features_c = in_features_c//2

        # Output layer
        model_cryo += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features_c, output_nc, 7),
                    nn.Sigmoid()] # change to tanh, sigmoid for hsv

        self.model_cryo = nn.Sequential(*model)
            
        self.up_64 = torch.nn.Upsample(size=None, scale_factor=0.25, mode='bilinear')
        self.up_128 = torch.nn.Upsample(size=None, scale_factor=0.5, mode='bilinear')


    def forward(self, x):
        x_128 = self.up_128(x)
        x_64 = self.up_64(x)
        x_out = self.refpad1(x)
        x_out = self.conv1(x_out)
        x_out = self.instNorm(x_out)
        x_out = self.relu(x_out)
        
        

        x_out = self.conv2(x_out)
        x_out = self.instNorm2(x_out)
        x_out = self.relu(x_out)
        

        x_out = self.conv3(x_out)
        x_out = self.instNorm3(x_out)
        x_out = self.relu(x_out) 
        
        x_out_128 = self.conv128(x_128)
        x_out_64 = self.conv64(x_64)
        
        x_out_multiscale = torch.cat([x_out_128, x_out_64], dim=1)
        
        x_out = x_out + x_out_multiscale      

        x_out_mri = self.model(x_out)
        x_out_cryo = self.model_cryo(x_out)
        
              
        return x_out_mri, x_out_cryo



class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 128, 4, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(128, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    

# class Discriminator(nn.Module):#;2 seg training
#     def __init__(self, input_nc):
#         super(Discriminator, self).__init__()

#         # A bunch of convolutions one after another
#         model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(128), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
#                     nn.InstanceNorm2d(256), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         model += [  nn.Conv2d(256, 256, 4, padding=1),
#                     nn.InstanceNorm2d(256), 
#                     nn.LeakyReLU(0.2, inplace=True) ]

#         # FCN classification layer
#         model += [nn.Conv2d(256, 1, 4, padding=1)]

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         x =  self.model(x)
#         # Average pooling and flatten
#         return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
    


""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
""" Full assembly of the parts to form the complete network """



class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)




class GeneratorSegShift(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(GeneratorSegShift, self).__init__()

        ## Different scales convs
        self.conv128 = nn.Sequential(nn.Conv2d(input_nc, 64, 3, 1, padding=1),
                      nn.ReLU(),
                      nn.Conv2d(64, 128, 3, 1, 1), 
                      nn.ReLU(),
                      nn.Conv2d(128, 256, 3, 2, 1), 
                      nn.ReLU())

        self.conv64 = nn.Sequential(nn.Conv2d(input_nc, 64, 3, 1, padding=1),
                            nn.ReLU(),
                            nn.Conv2d(64, 128, 3, 1, 1), 
                            nn.ReLU(),
                            nn.Conv2d(128, 256, 3, 1, 1), 
                            nn.ReLU())
        
        
        # Initial convolution block       
        self.refpad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(input_nc, 128, 7)
        self.instNorm = nn.InstanceNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        model =[]

        # Downsampling
        in_features = 128
        out_features = in_features*2
        self.conv2 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.instNorm2 = nn.InstanceNorm2d(out_features)
        self.relu =  nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2

        self.conv3 = nn.Conv2d(in_features, out_features, 3, stride=2, padding=1)
        self.instNorm3 = nn.InstanceNorm2d(out_features)
        self.relu =  nn.ReLU(inplace=True)
        in_features = out_features
        out_features = in_features*2

        in_features_c = out_features
        out_features_c = in_features_c*2


        # Residual blocks
        for _ in range(n_residual_blocks):
            # model += [ResidualBlock(in_features)]
            model += [SplitResBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Sigmoid()] # change to tanh, sigmoid for hsv

        self.model = nn.Sequential(*model)
        ### MRI to Cryo
        model_cryo=[]
        for _ in range(n_residual_blocks):
            # model += [ResidualBlock(in_features)]
            model_cryo += [SplitResBlock(in_features_c)]

        # Upsampling
        out_features_c = in_features_c//2
        for _ in range(2):
            model_cryo += [  nn.ConvTranspose2d(in_features_c, out_features_c, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features_c),
                        nn.ReLU(inplace=True) ]
            in_features_c = out_features_c
            out_features_c = in_features_c//2

        # Output layer
        # model_cryo += [  nn.ReflectionPad2d(3),
        #             nn.Conv2d(in_features_c, output_nc, 7),
        #             nn.Sigmoid()] # change to tanh, sigmoid for hsv

        # self.model_cryo = nn.Sequential(*model)
            
        self.up_64 = torch.nn.Upsample(size=None, scale_factor=0.25, mode='bilinear')
        self.up_128 = torch.nn.Upsample(size=None, scale_factor=0.5, mode='bilinear')


    def forward(self, x):
        x_128 = self.up_128(x)
        x_64 = self.up_64(x)
        x_out = self.refpad1(x)
        x_out = self.conv1(x_out)
        x_out = self.instNorm(x_out)
        x_out = self.relu(x_out)
        
        

        x_out = self.conv2(x_out)
        x_out = self.instNorm2(x_out)
        x_out = self.relu(x_out)
        

        x_out = self.conv3(x_out)
        x_out = self.instNorm3(x_out)
        x_out = self.relu(x_out) 
        
        x_out_128 = self.conv128(x_128)
        x_out_64 = self.conv64(x_64)
        
        x_out_multiscale = torch.cat([x_out_128, x_out_64], dim=1)
        
        x_out = x_out + x_out_multiscale      

        x_out_mri = self.model(x_out)
        
              
        return x_out_mri