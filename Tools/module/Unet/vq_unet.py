
from typing import Callable, List, Tuple
import torch
from torch.nn.modules import Module
from Tools.module.Unet.unet import Unet, default_build_down_subnet, default_build_up_subnet
import torch.nn as nn

from Tools.module.probability.vqvae import VQEmbed

class VQUnet(Unet):
    def __init__(self, 
                 in_channel: int, 
                 num_layer: int, 
                 num_embed: int = 512,
                 reject_skip: List[int] = [-1], 
                 build_down_subnet: Callable[[int, int], Tuple[Module, int]] = default_build_down_subnet, 
                 build_up_subnet: Callable[[int, int, int], Tuple[Module, int]] = default_build_up_subnet):
        super().__init__(in_channel, num_layer, reject_skip, build_down_subnet, build_up_subnet)

        self.codebooks = []
        for idx, channel in enumerate(self.down_channels):
            if channel == 0 and idx != num_layer-1:
                self.codebooks.append(nn.Identity())
                pass
            else:
                if idx == num_layer-1:
                    channel = self.bottle_channel
                self.codebooks.append(VQEmbed(num_embed, channel))
                pass
            pass
        self.codebooks = nn.ModuleList(self.codebooks)
        pass
    
    def forward(self, x):
        skip_feats = []
        vq_loss = 0
        for idx, layer in enumerate(self.down_subnet):
            
            x = self.down_forward(x, layer)
            codebook = self.codebooks[idx]
            
            if self.down_channels[idx] != 0:
                factor = len(self.down_subnet) - 1 - idx
                x, loss = codebook(x)
                vq_loss += (0.5 ** factor) * loss
                skip_feats.append(x)
            else:
                skip_feats.append(None)
            pass
        
        # 最底层
        x, loss = self.codebooks[-1](x)
        vq_loss += loss
        
        skip_feats = list(reversed(skip_feats))
        
        for idx, layer in enumerate(self.up_subnet):
            skip_feat = skip_feats[idx]
            x = self.up_forward(x, skip_feat, layer)
            pass
        return x, vq_loss
        pass
    pass

if __name__ == "__main__":
    device = "cuda:0"
    model = VQUnet(3, 5, 512).to(device)
    #print(model)
    x = torch.randn(4, 3, 256, 256).to(device)
    x, loss = model(x)
    print(x.size())
    
    pass