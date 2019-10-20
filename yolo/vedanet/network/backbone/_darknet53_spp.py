#
#   Darknet YOLOv3 model
#   Copyright EAVISE
#


# modified by mileistone

import os
from collections import OrderedDict, Iterable
import logging
import torch
import torch.nn as nn

from .. import layer as vn_layer
from .brick import darknet53 as bdkn

__all__ = ['Darknet53spp']


class Darknet53spp(nn.Module):
    custom_layers = (bdkn.Stage, bdkn.HeadBody, bdkn.Transition,
                     bdkn.Stage.custom_layers, bdkn.HeadBody.custom_layers, bdkn.Transition.custom_layers,
                     bdkn.SPPBody)

    def __init__(self):
        super().__init__()

        input_channels = 32
        stage_cfg = {'stage_2': 2, 'stage_3': 3, 'stage_4': 9, 'stage_5': 9, 'stage_6': 5}

        # Network
        layer_list = [
            # layer 0
            # first scale, smallest
            OrderedDict([
                ('stage_1', vn_layer.Conv2dBatchLeaky(3, input_channels, 3, 1, 1)),     # DBL(in->3, out->32, kernel_size-> 3, stride->1, padding->1)
                ('stage_2', bdkn.Stage(input_channels, stage_cfg['stage_2'])),          # DBL + res1
                ('stage_3', bdkn.Stage(input_channels*(2**1), stage_cfg['stage_3'])),   # res2
                ('stage_4', bdkn.Stage(input_channels*(2**2), stage_cfg['stage_4'])),   # res8
            ]),

            # layer 1
            # second scale
            OrderedDict([
                ('stage_5', bdkn.Stage(input_channels*(2**3), stage_cfg['stage_5'])),   # res8
            ]),

            # layer 2
            # third scale, largest
            OrderedDict([
                ('stage_6', bdkn.Stage(input_channels*(2**4), stage_cfg['stage_6'])),   # res4
            ]),

            # the following is extra
            # layer 3
            # output third scale, largest
            OrderedDict([
                ('head_body_1', bdkn.SPPBody(input_channels*(2**5), first_head=True)),  # 5DBL:(1-3)DBL + SPP +(4-5)
            ]),

            # layer 4
            OrderedDict([
                ('trans_1', bdkn.Transition(input_channels*(2**4))),                    # DBL+UPSAMPLING
            ]),

            # layer 5
            # output second scale
            OrderedDict([
                ('head_body_2', bdkn.HeadBody(input_channels*(2**4+2**3))),             # 5DBL
            ]),

            # layer 6
            OrderedDict([
                ('trans_2', bdkn.Transition(input_channels*(2**3))),                    # DBL+UPSAMPLING
            ]),

            # layer 7
            # output first scale, smallest
            OrderedDict([
                ('head_body_3', bdkn.HeadBody(input_channels*(2**3+2**2))),             # 5DBL
            ]),

        ]

        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, x):
        features = []
        outputs = []

        stage_4 = self.layers[0](x)                 # DBL + res1 + res2 + No.1 res8
        stage_5 = self.layers[1](stage_4)           # No.2 res8
        stage_6 = self.layers[2](stage_5)           # res4

        head_body_1 = self.layers[3](stage_6)       # 5DBL
        trans_1 = self.layers[4](head_body_1)       # DBL + UPsampling

        concat_2 = torch.cat([trans_1, stage_5], 1)     # DBL + UPsampling cat No.2 res8
        head_body_2 = self.layers[5](concat_2)          # 5DBL
        trans_2 = self.layers[6](head_body_2)           # DBL + UPsampling

        concat_3 = torch.cat([trans_2, stage_4], 1)     # No.1 res8 cat
        head_body_3 = self.layers[7](concat_3)

        # stage 6, stage 5, stage 4
        features = [head_body_1, head_body_2, head_body_3]

        return features 

