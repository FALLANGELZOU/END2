from typing import Callable, List, Tuple
import torch
import torch.nn as nn

from Tools.module.common import ResBlock

def default_build_down_subnet(in_channel, block_idx):
    model = nn.Sequential(
        ResBlock(in_channel, in_channel, 'bn', 'lrelu'),
        ResBlock(in_channel, in_channel * 2, 'bn', 'lrelu'),
        nn.MaxPool2d(kernel_size=(2, 2))
    )
    return model, in_channel * 2
    pass
def default_build_up_subnet(in_channel, skip_channel, block_idx):
    model = nn.Sequential(
        ResBlock(in_channel + skip_channel, in_channel, 'bn', 'lrelu'),
        ResBlock(in_channel, in_channel // 2, 'bn', 'lrelu'),
        nn.Upsample(scale_factor=2)
    )
    return model, in_channel // 2
    pass
class Unet(nn.Module):
    def __init__(self,
                 in_channel: int,
                 num_layer: int,
                 reject_skip: List[int] = [-1],
                 build_down_subnet: Callable[[int, int], Tuple[nn.Module, int]] = default_build_down_subnet,
                 build_up_subnet: Callable[[int, int, int], Tuple[nn.Module, int]] = default_build_up_subnet
                 ):
        """默认的CNN Unet

        Args:
            in_channel (int):  输入通道数
            num_layer (int):  共多少层
            reject_skip (list, optional):  Defaults to [-1]. 哪几层不需要跳跃连接，默认最后一层不需要，即最下面的瓶颈层不需要
            build_down_subnet (Callable[in_channel, block_idx] -> [model, out_channel]): _description_. 下采样网络层构造
            build_up_subnet (Callable[in_channel, skip_channel, block_idx] -> [model, out_channel]): _description_. 上采样网络层构造
        """
        super(Unet, self).__init__()
        self.down_subnet = []
        self.up_subnet = []
        self.down_channels = []
        for idx in range(num_layer):
            subnet, in_channel = build_down_subnet(in_channel, idx)
            self.down_subnet.append(subnet)
            self.down_channels.append(in_channel)
            pass
        self.bottle_channel = self.down_channels[-1] # 最下层channel
        # 设置不需要跳跃连接的层
        for i in reject_skip:
            self.down_channels[i] = 0
        self.up_channels = list(reversed(self.down_channels))
        
        for idx in range(num_layer):
            skip_channel = self.up_channels[idx]
            if skip_channel == 0:
                subnet, in_channel = build_up_subnet(in_channel, 0, idx)
            else:
                subnet, in_channel = build_up_subnet(in_channel, skip_channel, idx)
            self.up_subnet.append(subnet)
            pass
        
        self.down_subnet = nn.ModuleList(self.down_subnet)
        self.up_subnet = nn.ModuleList(self.up_subnet)
        pass 
    
    def forward(self, x):
        skip_feats = []
        for idx, layer in enumerate(self.down_subnet):
            x = self.down_forward(x, layer)
            
            if self.down_channels[idx] != 0:
                skip_feats.append(x)
            else:
                skip_feats.append(None)
            pass
        
        skip_feats = list(reversed(skip_feats))
        
        for idx, layer in enumerate(self.up_subnet):
            skip_feat = skip_feats[idx]
            x = self.up_forward(x, skip_feat, layer)
            pass
        return x
        pass
    
    def down_forward(self, x, layer):
        """下采样过程，传入当前特征图，和对应的卷积块，返回经过网络层的特征图

        Args:
            x (_type_): _description_ 特征图
            layer (_type_): _description_ 神经网络层

        Returns:
            _type_: _description_ 处理后的特征图
        """
        return layer(x)
        pass
    
    def up_forward(self, x, skip_feat, layer):
        """上采样过程

        Args:
            x (_type_): _description_ 特征图
            skip_feat (_type_): _description_ 跳跃连接的特征图，如果为None则没有跳跃连接
            layer (_type_): _description_ 网络层

        Returns:
            _type_: _description_ 处理后的特征图
        """
        if skip_feat is not None:
            x = torch.cat([x, skip_feat], dim=1)
        return layer(x)
        pass
    pass


if __name__ == "__main__":
    model = Unet(3, 3)
    #print(model)
    x = torch.randn(4, 3, 256, 256)
    x = model(x)
    print(x.size())
    pass