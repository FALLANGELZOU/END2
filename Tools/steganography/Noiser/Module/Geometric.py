import random
import torch
import torch.nn as nn
from Tools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager
import torch.nn.functional as F
from kornia.augmentation import RandomAffine
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
import numpy as np
import math
from kornia.geometry import rotate, translate, scale, shear

from Tools.util.ImgUtil import img2tensor
from Tools.util.utils import vutils



@NoiseLayerManager.register("Affine")
class Affine(NoiseLayer):
    '''
    随机仿射变换
    '''
    def __init__(self, 
                 degree=0,
                 translate=None,
                 scale=None,
                 shear=None,
                 padding_mode="zeros",
                 same_on_batch=True,
                 sync_cover_img=True
                 ):
        super().__init__()
        self.sync_cover_img = sync_cover_img
        if padding_mode == "zeros":
            padding_mode = 0
        elif padding_mode == "border":
            padding_mode = 1
        elif padding_mode == "reflection":
            padding_mode = 2
        else:
            padding_mode = 0

        self.model = RandomAffine(
                degrees=degree, 
                translate=translate, 
                scale=scale,
                shear=shear,
                padding_mode=padding_mode,
                same_on_batch=same_on_batch,
                p=1
                )
        pass

    def noise(self, img, cover_img):
        img = self.model(img)
        if self.sync_cover_img:
            cover_img = self.model(cover_img, params=self.model._params)
        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        img = self.model(img, params=self.model._params)
        if self.sync_cover_img:
            cover_img = self.model(cover_img, params=self.model._params)
        return img, cover_img       
        pass


@NoiseLayerManager.register("Rotate")
class Rotate(NoiseLayer):
    def __init__(self, 
                 angle=[-30, 30],
                 same_on_batch=False,
                 padding_mode='zeros',
                 sync_cover_img=True
                 ):
        """旋转

        Args:
            angle (list, optional): 旋转角度范围，角度制. Defaults to [-30, 30].
            same_on_batch: 是否一个batch采用相同的变换
            padding_mode: 'zeros' | 'border' | 'reflection'
        """
        super().__init__()
        self.angle = angle
        self.same_on_batch = same_on_batch
        self.now_angle = 0
        self.padding_mode = padding_mode
        self.sync_cover_img = sync_cover_img
        pass

    def noise(self, img, cover_img, repeat=False):
        bs = img.size(0)
        if repeat is False:
            if self.same_on_batch:
                angle = torch.from_numpy(np.random.uniform(*self.angle, size=(1, ))).float().expand(bs).to(img.device)
            else:
                angle = torch.from_numpy(np.random.uniform(*self.angle,size=(bs, ))).float().to(img.device)
            pass
            self.now_angle = angle
        img = rotate(img, self.now_angle, padding_mode=self.padding_mode)
        if self.sync_cover_img:
            cover_img = rotate(cover_img, self.now_angle, padding_mode=self.padding_mode)
        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, True)
    pass

@NoiseLayerManager.register("Translate")
class Translate(NoiseLayer):
    def __init__(self, 
                 translation=[-0.1, 0.1],
                 same_on_batch=False,
                 padding_mode='zeros',
                 sync_cover_img=True
                 ):
        super().__init__()
        self.translation = translation
        self.same_on_batch = same_on_batch
        self.padding_mode = padding_mode
        self.sync_cover_img = sync_cover_img
        self.p = None
        pass

    def noise(self, img, cover_img, repeat=False):
        bs, c, h, w = img.size()
        if repeat is False:
            self.p = _random(*self.translation, size=(bs, 2), same_on_batch=self.same_on_batch).to(img.device)
            self.p = self.p * torch.tensor([h, w]).float().unsqueeze(0).to(img.device)
        img = translate(img, self.p, padding_mode=self.padding_mode)
        if self.sync_cover_img:
            cover_img = translate(cover_img, self.p, padding_mode=self.padding_mode)
        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, True)
    
    pass


@NoiseLayerManager.register("Scale")
class Scale(NoiseLayer):
    def __init__(self, 
                 scale_factor=[0.5, 1.2],
                 same_on_batch=False,
                 padding_mode='zeros',
                 sync_cover_img=True
                 ):
        super().__init__()
        self.scale_factor = scale_factor
        self.same_on_batch = same_on_batch
        self.padding_mode = padding_mode
        self.sync_cover_img = sync_cover_img
        self.p = None
        pass

    def noise(self, img, cover_img, repeat=False):
        bs, c, h, w = img.size()
        if repeat is False:
            self.p = _random(*self.scale_factor, size=(bs, 1), same_on_batch=self.same_on_batch).to(img.device)
        img = scale(img, self.p, padding_mode=self.padding_mode)
        if self.sync_cover_img:
            cover_img = scale(cover_img, self.p, padding_mode=self.padding_mode)
        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, True)
    
    pass



@NoiseLayerManager.register("Shear")
class Shear(NoiseLayer):
    def __init__(self, 
                 shear_factor=[-0.1, 0.1],
                 same_on_batch=False,
                 padding_mode='zeros',
                 sync_cover_img=True
                 ):
        """_summary_

        Args:
            shear_factor (list, optional): 轴旋转的角度，范围[-1, 1]. Defaults to [-0.1, 0.1].
            same_on_batch (bool, optional): _description_. Defaults to False.
            padding_mode (str, optional): _description_. Defaults to 'zeros'.
            sync_cover_img (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.shear_factor = shear_factor
        self.same_on_batch = same_on_batch
        self.padding_mode = padding_mode
        self.sync_cover_img = sync_cover_img
        self.p = None
        pass

    def noise(self, img, cover_img, repeat=False):
        bs, c, h, w = img.size()
        if repeat is False:
            self.p = _random(*self.shear_factor, size=(bs, 2), same_on_batch=self.same_on_batch).to(img.device)
        img = shear(img, self.p, padding_mode=self.padding_mode)
        if self.sync_cover_img:
            cover_img = shear(cover_img, self.p, padding_mode=self.padding_mode)
        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, True)
    
    pass

def _random(low, high, size=None, same_on_batch=True):
    if size is None:
        data = torch.from_numpy(np.random.uniform(low, high, size=(1, )))
    else:
        bs = size[0]
        shape = size[1:]
        if same_on_batch:
            data = torch.from_numpy(np.random.uniform(low, high, size=(1, *shape))).expand(bs, *shape)
            pass
        else:
            data = torch.from_numpy(np.random.uniform(low, high, size=(bs, *shape)))
            pass
        pass
    return data.float()
    pass



if __name__ == "__main__":
    # print(_random(-1, 1, size=(2, 2), same_on_batch=False))
    model = Shear()
    x = img2tensor("/home/light_sun/workspace/ImageST/test/test.png").expand(2, -1, -1, -1)
    x, y = model(x, x)
    vutils().save_image(torch.cat([x, y], dim=0), "./test/affine.png")
    pass