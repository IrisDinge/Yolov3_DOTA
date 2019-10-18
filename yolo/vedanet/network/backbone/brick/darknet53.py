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
            vn_layer.Conv2dBatchLeaky(nchannels, int(nchannels / 2), 1, 1),
            vn_layer.Conv2dBatchLeaky(int(nchannels / 2), nchannels, 3, 1)
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
        blocks.append(vn_layer.Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, stride))
        for ii in range(nblocks - 1):
            blocks.append(StageBlock(2 * nchannels))
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
            half_nchannels = int(nchannels / 2)
        else:
            half_nchannels = int(nchannels / 3)
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
        half_nchannels = int(nchannels / 2)
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

        '''
        def calc_auto(num, channel):
            lst = [5, 9, 13]
            return sum(map(lambda x: x ** 2, lst[:num])) * channel
        '''

        layer1 = vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1)
        layer2 = vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1)
        layer3 = vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
        layer4 = vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1)
        layer5 = vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
        spp5 = nn.MaxPool2d(kernel_size=5, stride=1)
        spp9 = nn.MaxPool2d(kernel_size=9, stride=1)
        spp13 = nn.MaxPool2d(kernel_size=13, stride=1)



        layer_list = [
            OrderedDict([

                ('1', layer1),
                ('2', layer2),
                ('3', layer3),
            ]),
            
            OrderedDict([
                ('maxpool', spp5),
                ('route', layer3),
                                
                
            ]),

            OrderedDict([

                ('3', layer3),
                ('4', layer4),
                ('5', layer5),
            ]),
            
        ]

        self.feature = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])


    def forward(self, data):
        x = self.feature(data)
        return x





'''
DBL = con + BN + LeakyReLU
a.k.a vn_layer.Conv2dBatchLeaky
'''


class DCNv2Block(nn.Module):
    '''

    Res_Unit = ---> DBL + DBL ---> Add ----
                |                   ^
                |___________________|

    '''
    custom_layers = ()

    def __init__(self, nchannels):
        super(DCNv2Block,self).__init__()
        self.features = nn.Sequential(
            vn_layer.ConvNet(),
            nn.LeakyReLU(0.1),
            vn_layer.ConvNet(),
            nn.LeakyReLU(0.1)
        )

    def forward(self, data):
        return data + self.features(data)



class DCNv2(nn.Module):
    '''

    Resblock_body = zero padding + DBL + Res_Unit * N

    THere are 8 Res_Unit, so far we only replace 2 traditional conv2d layers of DBL in the last Res_Unit

    the last Res_Unit = ---> DBL_DCN(Conv2d+BN+LeakyReLU) + DBL_DCN(COnv2d+BN+LeakyReLU)---> Add ----
                         |                                                                    ^
                         |____________________________________________________________________|


    '''
    custom_layers = (StageBlock, StageBlock.custom_layers, DCNv2Block)

    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(vn_layer.Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, stride))
        for ii in range(nblocks - 2):
            blocks.append(StageBlock(2 * nchannels))
        blocks.append(DCNv2Block(nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


