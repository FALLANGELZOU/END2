import torch
import torch.nn as nn
from kornia.augmentation import RandomAffine, RandomGaussianNoise, ColorJitter, RandomErasing
import torch.nn.functional as F
import math
import random
from Tools.module.noiser.Noiser import NOISE_LAYER
@NOISE_LAYER.register("diff_affine")
class DiffAffine(nn.Module):
    def __init__(self,
                 degree=None,
                 translate=None,
                 scale=None,
                 shear=None
                 ):
        super().__init__()
        if degree == translate == scale == shear == None:
            self.model = nn.Identity()
        else:
            self.model = RandomAffine(
                degrees=degree, 
                translate=translate, 
                scale=scale,
                shear=shear,
                padding_mode=1,
                p=1)
        pass
    
    def forward(self, x):
        return self.model(x)
        pass
    pass

@NOISE_LAYER.register("diff_degree")
class DiffDegree(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.model = RandomAffine(
            degrees=degree,
            p=1,
            padding_mode=1
        )
        pass
    def forward(self, x):
        return self.model(x)
    
    pass

@NOISE_LAYER.register("diff_translate")
class DiffTranslate(nn.Module):
    def __init__(self, translate):
        super().__init__()
        self.model = RandomAffine(
            degrees=0,
            translate=translate,
            p=1,
            padding_mode=1
        )
        pass
    def forward(self, x):
        return self.model(x)
    pass

@NOISE_LAYER.register("diff_resize")
class DiffResize(nn.Module):
    def __init__(self, resize):
        super().__init__()
        self.model = RandomAffine(
            degrees=0,
            scale=resize,
            p=1,
            padding_mode=1
        )
        pass
    def forward(self, x):
        return self.model(x)
    pass


@NOISE_LAYER.register("diff_noise")
class DiffGaussianNoise(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.model = RandomGaussianNoise(mean, std, p=1)
        pass
    def forward(self, x):
        x = self.model(x)
        x = torch.clamp(x, -1., 1.)
        return x
    pass

@NOISE_LAYER.register("diff_color")
class DiffColor(nn.Module):
    def __init__(self, b,c,s,h):
        super().__init__()
        self.model = ColorJitter(b, c, s, h)
        pass
    
    def forward(self, x):
        flag = False
        if x.size(1) == 1:
            x = x.expand(-1, 3, -1, -1)
            flag = True
        x = self.model(x)
        if flag:
            x = torch.mean(x, dim=1, keepdim=True)
        x = torch.clamp(x, -1., 1.)
        return x
    pass

@NOISE_LAYER.register("diff_scale")
class DiffScale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        pass
    
    def forward(self, x):
        b, c, h, w = x.size()
        factor = random.uniform(self.factor, 1)
        x = F.interpolate(x, size=(int(h * factor), int(w * factor)), mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x
        pass
    pass