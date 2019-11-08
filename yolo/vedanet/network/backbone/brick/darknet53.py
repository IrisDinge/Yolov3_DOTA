import os
from collections import OrderedDict
import torch
import torch.nn as nn
from ... import layer as vn_layer
from ... dcnv2.dcn_v2 import DCN
from ... dcnv2.dcn_v2 import DCNPooling

import math

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

    def __init__(self, nchannels, first_head=True):
        super().__init__()
        if first_head:
            half_nchannels = int(nchannels / 2)
        else:
            half_nchannels = int(nchannels / 3)
        in_nchannels = 2 * half_nchannels


        layers1_3 = [

            vn_layer.Conv2dBatchLeaky(nchannels, half_nchannels, 1, 1),
            vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),
            vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)
        ]

        self.DBL3_5 = nn.Sequential(*layers1_3)


        layer_list = [
            OrderedDict([                                               # num, c, w, h
                ('maxpool', nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1), padding=(0, 0), dilation=(1, 1))),     # num x 512 x 9 x 9
                ('spp1',    vn_layer.SPPLayer(level=3))            # padding 6 + (3 - 1) * (-2) = 2
            ]),                                                    # num x 512 x 13 x 13
            OrderedDict([
                ('maxpool', nn.MaxPool2d(kernel_size=(9, 9), stride=(1, 1), padding=(0, 0), dilation=(1, 1))),     # num x 512 x 5 x 5
                ('spp2',    vn_layer.SPPLayer(level=2))             # padding 6 + (2 - 1) * (-2) = 4
                                                                    # num x 512 x 13 x 13
            ]),
            OrderedDict([
                ('maxpool', nn.MaxPool2d(kernel_size=(13, 13), stride=(1, 1), padding=(0, 0), dilation=(1, 1))),     # num x 512 x 1 x 1
                ('spp3',    vn_layer.SPPLayer(level=1))             # padding 6 + (1 - 1) * (-2) = 6
                                                                    # num x 512 x 13 x 13
            ]),
        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])




        layers3_5 = [

            vn_layer.Conv2dBatchReLU(in_nchannels*2, in_nchannels, 1, 1),           # 2048 -> 1024
            vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1),          # layer3
            vn_layer.Conv2dBatchLeaky(half_nchannels, in_nchannels, 3, 1),          # layer4
            vn_layer.Conv2dBatchLeaky(in_nchannels, half_nchannels, 1, 1)           # layer5
        ]

        self.feature = nn.Sequential(*layers3_5)

    def forward(self, data):
        # N, C, W, H = x.size()
        x = self.DBL3_5(data)

        spp1 = self.layers[0](x)
        spp2 = self.layers[1](x)
        #spp3 = self.layers[2](x)

        spp = torch.cat((x, spp2, spp1, x), 1)  # from C dim cat together

        x = self.feature(spp)
        #x = self.feature(data)
        return x




'''
DBL = con + BN + LeakyReLU
a.k.a vn_layer.Conv2dBatchLeaky
'''


class DCNv2Block(nn.Module):
    '''

    Res_Unit = ---> DBL(replaced) + DBL(replaced) ---> Add ----
                |                                  ^
                |__________________________________|

    '''
    custom_layers = ()

    def __init__(self, nchannels):                                                  # nchannels = 256
        super().__init__()

        self.features = nn.Sequential(
            DCN(nchannels, int(nchannels / 2), kernel_size=(1, 1), stride=1, padding=1, deformable_groups=1).cuda(),    # (256, 128, 1, 1)
            DCN(int(nchannels / 2), nchannels, kernel_size=(3, 3), stride=1, padding=0, deformable_groups=1).cuda(),    # (128, 256, 3, 1)
            )

    def forward(self, data):

        data = data.cuda()

        return data + self.features(data)

#def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
#'stage_4', bdkn.Stage(input_channels*(2**1), stage_cfg['stage_4']=9)



class DCNv2(nn.Module):
    '''

    Resblock_body = zero padding + DBL + Res_Unit * N

    There are 8 Res_Unit, so far we only replace 2 traditional conv2d layers of DBL in the last 2 Res_Unit
    It means 4 standard convolutional layers are replaced by deformable convolution layers

    the last 2 Res_Unit = ---> DBL_DCN(Conv2d+BN+LeakyReLU) + DBL_DCN(COnv2d+BN+LeakyReLU)---> Add ----
                         |                                                                    ^
                         |____________________________________________________________________|


    '''
    custom_layers = (StageBlock, StageBlock.custom_layers)

    def __init__(self, nchannels, nblocks, stride=2):                   # nchannels = 128, nblocks = 9
        super().__init__()
        blocks = []
        blocks.append(vn_layer.Conv2dBatchLeaky(nchannels, 2 * nchannels, 3, stride))   #layer[0] DBL
        for ii in range(nblocks - 3):                                   # 9 - 3 = 6
            blocks.append(StageBlock(2 * nchannels))                    # StageBlock(256)
        blocks.append(DCNv2Block(2*nchannels))                          # DCNv2Block(256)
        blocks.append(DCNv2Block(2*nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)


