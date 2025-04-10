import torch
import torch.nn as nn
import random
from Tools.register.registry import Registry

NOISE_LAYER = Registry("noise_layer")

class Noiser(nn.Module):
    def __init__(self, layers = [
        ("diff_jpeg", {"max_quality":90, "min_quality": 50})
    ]):
        super(Noiser, self).__init__()
        self.models = []
        
        for name, kwargs in layers:
            if kwargs is None:
                self.models.append(NOISE_LAYER.get(name)())
            else:
                self.models.append(NOISE_LAYER.get(name)(**kwargs))
            pass
        
        self.models = nn.ModuleList(self.models)
        pass
    def forward(self, x):
        layer = random.choice(self.models)
        # print(layer)
        return layer(x)
        pass
    pass




if __name__ == "__main__":
    model = Noiser()
    
    pass