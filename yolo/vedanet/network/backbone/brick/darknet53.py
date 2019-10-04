import os
from collections import OrderedDict
import torch
import torch.nn as nn

from ... import layer as vn_layer


'''
DBL = con + BN + LeakyReLU
a.k.a vn_layer.COnv2dBatchLeaky
'''

class StageBlock(nn.Module):
    '''
    
    Res_Unit = ---> DBL + DBL ---> Add ----
                |                   ^
                |___________________|
                
    '''
    custom_layers = ()
    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
                    vn_layer.Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1),
                    vn_layer.Conv2dBatchLeaky(int(nchannels/2), nchannels, 3, 1)
                )

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):
    '''
    
    Resblock_body = zero padding + DBL + Res_Unit * N
    
    '''
    custom_layers = (StageBlock, StageBlock.custom_layers)
    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(vn_layer.DeformConv2(nchannels, 2*nchannels, 3, stride))
        for ii in range(nblocks - 1):
            blocks.append(StageBlock(2*nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


class HeadBody(nn.Module):
    
    '''
    
    5 * DBL 
    
    '''
    
    custom_layers = ()
    def __init__(self, nchannels, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels/2)
        else:
            half_nchannels = int(nchannels/3)
        in_nchannels = 2 * half_nchannels
        layers = [
                vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
                vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
                vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
                ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x




class Transition(nn.Module):
    
    '''
    
    DBL + UPsampling
    
    '''
    custom_layers = ()
    def __init__(self, nchannels):
        super().__init__()
        half_nchannels = int(nchannels/2)
        layers = [
                vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
                nn.Upsample(scale_factor=2)
                ]

        self.features = nn.Sequential(*layers)

    def forward(self, data):
        x = self.features(data)
        return x


class SPPBody(nn.Module):
    '''

    5 * DBL: (1-3) + spp +(4-5)

    '''

    custom_layers = ()

    def __init__(self, nchannels, first_head=False):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels / 2)
        else:
            half_nchannels = int(nchannels / 3)
        in_nchannels = 2 * half_nchannels
        layers = [
            vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
            vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),
            vn_layer.SPPLayer(3, pool_type='max_pool'), #num_level=3
            vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
        ]
        self.feature = nn.Sequential(*layers)

    def forward(self, data):
        x = self.feature(data)
        return x
    
''' 
class DCNv2(nn.Module):

    

    #Deformable Convolutional Networks
    how to replace regular convolution layers

    


    def __init__(self):
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = vn_layer.DeformConv2(32, 64, 3, padding=1, modulation=True)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x
'''



'''
DBL = con + BN + LeakyReLU
a.k.a vn_layer.COnv2dBatchLeaky
'''


class DCNv2Block(nn.Module):
    '''

    Res_Unit = ---> (DCNv2+BN+LeakyReLU) + (DCNv2+BN+LeakyReLU) ---> Add ----
                |                                                     ^
                |_____________________________________________________|

    '''
    custom_layers = ()

    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
            vn_layer.DeformConv2(nchannels, int(nchannels / 2), 1, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True),
            vn_layer.DeformConv2(int(nchannels / 2), nchannels, 3, 1),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True),
        )

    def forward(self, data):
        return data + self.features(data)


class DCNv2(nn.Module):
    '''

    Resblock_body = zero padding + (DCNv2+BN+LeakyReLU) + Res_Unit * N

    '''
    custom_layers = (DCNv2Block, DCNv2Block.custom_layers)

    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(vn_layer.DCNv22dBatchReLU(nchannels, 2 * nchannels, 3, stride))
        for ii in range(nblocks - 1):
            blocks.append(DCNv2Block(2 * nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


