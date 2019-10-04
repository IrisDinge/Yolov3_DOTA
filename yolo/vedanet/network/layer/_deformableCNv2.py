
import torch.nn as nn

from ._darknet import DeformConv2 as dcnv2

__all__ = ['deformableCNv2']

class deformableCNv2(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a ReLU.
    They are executed in a sequential manner.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Size of the kernel of the convolution
        stride (int or tuple): Stride of the convolution
        padding (int or tuple): padding of the convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(deformableCNv2, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(ii / 2) for ii in kernel_size]
        else:
            self.padding = int(kernel_size / 2)

        # Layer
        self.layers = nn.Sequential(
            dcnv2.DeformConv2(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x
