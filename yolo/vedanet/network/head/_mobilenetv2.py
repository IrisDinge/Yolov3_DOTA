import os
from collections import OrderedDict
import torch
import torch.nn as nn

from .. import layer as vn_layer

__all__ = ['Mobilenetv2']


class Mobilenetv2(nn.Module):
    def __init__(self, num_classes, in_channels_list, num_anchors_list):
        """ Network initialisation
        """
        super().__init__()
        layer_list = [
            # Sequence 0 : input = large prediction
            OrderedDict([
                ('1_convbatch', vn_layer.Conv2dBatchReLU(32, 64, 3, 1)),
                ('2_conv', nn.Conv2d(in_channels_list[0], num_anchors_list[0]*(5 + num_classes), 1, 1, 0)),
            ]),

            # Sequence 1 : input = Sequence 0 divide
            OrderedDict([
                ('3_convbatch', vn_layer.Conv2dBatchReLU(32, 8, 1, 1)),
                ('4_reorg', vn_layer.Reorg(2)),
            ]),

            # Sequence 2 : input = Sequence 1 and middle
            OrderedDict([
                ('5_convbatch',    vn_layer.Conv2dBatchReLU(96, 24, 1, 1)),
                ('6_conv',        nn.Conv2d(in_channels_list[1], num_anchors_list[1]*(5 + num_classes), 1, 1, 0)),
                ]),

            # Sequence 3: input = mid
            OrderedDict([
                ('7_convbatch',    vn_layer.Conv2dBatchReLU(96, 24, 1, 1)),
                ('8_conv',         vn_layer.Reorg(2)),
                ]),

            # Sequence 4:  inputs = Sequence3 + small
            OrderedDict([
                ('9_convbatch',     vn_layer.Conv2dBatchReLU((4*24)+320, 320, 3, 1)),
                ('10_conv',          nn.Conv2d(in_channels_list[2], num_anchors_list[2]*(5+num_classes), 1, 1, 0)),
                ]),
            ]
        self.layers = nn.ModuleList([nn.Sequential(layer_dict) for layer_dict in layer_list])

    def forward(self, middle_feats):
        '''
        :param middle_feats: stage6, stage5, stage4
        :return:
                                     input
        :Mobilenetv2    small_scale 56^2x24     6,32,3,2
                        middle_scale 14^2x64    6,96,3,1
                        large_scale 7^2x160     6,320,1,1
        '''
        outputs = []

        stage4 = middle_feats[2]
        out1 = self.layer[0](stage4)
        stage5_reorg = self.layers[1](stage4)

        stage5 = middle_feats[1]
        out2 = middle_feats[2](torch.cat((stage5_reorg, stage5), 1))
        stage6_reorg = self.layers[3](middle_feats[1])

        stage6 = middle_feats[0]
        out3 = self.layers[4](torch.cat((stage6_reorg,stage6), 1))

        features = [out3, out2, out1]
        return features
