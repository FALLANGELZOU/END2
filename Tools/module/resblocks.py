import torch
import torch.nn as nn

from Tools.module.common import Conv2dBlock

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm='bn', activation='lrelu', pad_type='zero', kernel_size=3):
        """
        ResBlock V2
        :param in_channel: input channel
        :param out_channel: output channel
        :param norm: norm
        :param activation: activation for residual mapping
        :param pad_type: padding type
        :param kernel_size: only 3, 5, 7
        """
        super(ResBlock, self).__init__()

        padding = 1
        if kernel_size == 5:
            padding = 2
        if kernel_size == 7:
            padding = 3
        self.pre = nn.Identity() if in_channel == out_channel else nn.Conv2d(in_channel, out_channel, (1, 1),
                                                                             bias=False)

        model = []
        model += [Conv2dBlock(out_channel, out_channel, kernel_size, 1, padding,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type, pre_norm_activation=True)]
        model += [Conv2dBlock(out_channel, out_channel, kernel_size, 1, padding,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type, pre_norm_activation=True)]

        self.model = nn.Sequential(*model)
        
        
if __name__ == "__main__":
    model = ResBlock(
        
    )
    pass