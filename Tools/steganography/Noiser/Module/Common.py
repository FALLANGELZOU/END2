import random
import torch
import torch.nn as nn
from Tools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager
import torch.nn.functional as F
from kornia.augmentation import RandomAffine
from kornia.enhance import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
import numpy as np
import math

@NoiseLayerManager.register("Identity")
class Identity(NoiseLayer):
    '''
    恒等层
    '''
    def __init__(self, ):
        super().__init__()
        self.model = None
        pass

    def noise(self, img, cover_img):
        return img, cover_img
    
    def latest_noise(self, img, cover_img):
        return img, cover_img
    pass



@NoiseLayerManager.register("Color")
class Color(NoiseLayer):
    def __init__(self, 
                 brightness=[-0.4, 0.4], 
                 saturation=[-0.4, 0.4],
                 hue=[-0.1, 0.1],
                 combine=False, 
                 clip_out=True):
        """颜色变换

        Args:
            brightness: 亮度调节，范围[-1,1]，-1代表全黑，1代表全白，0代表不变. Defaults to [-0.5, 0.5].
            saturation: 饱和度调节，范围[-1, 1], -1为灰度图，1为原来的两倍，0不变.
            hue: 色度调节，范围[-1, 1], -1和1代表在HSV空间中色相朝哪个方向旋转，0不变.
            combine: 是否组合亮度，饱和度，色度噪声，否则只选择其中一个
            clip_out: 是否将图像clip到[0, 1]

        """
        super().__init__()
        self.combine = combine
        self.clip_out = clip_out

        self.brightness_factor = brightness
        self.now_brightness_factor = 0

        self.saturation_factor = saturation
        self.now_saturation_factor = 0

        self.hue_factor = hue
        self.now_hue_factor = 0
        pass

    def noise(self, img, cover_img, repeat=False):
        if self.combine:
            img, cover_img = self._adjust_brightness(img, cover_img, repeat)
            img, cover_img = self._adjust_saturation(img, cover_img, repeat)
            img, cover_img = self._adjust_hue(img, cover_img, repeat)
            pass
        else:
            choice = np.random.randint(0, 3)
            if choice == 0:
                img, cover_img = self._adjust_brightness(img, cover_img, repeat)
            elif choice == 1:
                img, cover_img = self._adjust_saturation(img, cover_img, repeat)
            else:
                img, cover_img = self._adjust_hue(img, cover_img, repeat)

        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, True)
        pass

    def _adjust_brightness(self, img, cover_img, repeat=False):
        if repeat is False:
            self.now_brightness_factor = random.uniform(*self.brightness_factor)
        if self.now_brightness_factor >= 0:
            img = adjust_brightness(img, self.now_brightness_factor, self.clip_out)
            pass
        else:
            img = adjust_contrast(img, self.now_brightness_factor+1, self.clip_out)
            pass
        return img, cover_img
        pass

    def _adjust_saturation(self, img, cover_img, repeat=False):
        if img.size(-3) != 3:
            return img, cover_img
        if repeat is False:
            self.now_saturation_factor = random.uniform(*self.saturation_factor)
            pass
        img = adjust_saturation(img, self.now_saturation_factor+1)
        if self.clip_out:
            img = torch.clamp(img, 0, 1)
        return img, cover_img
        pass

    def _adjust_hue(self, img, cover_img, repeat=False):
        if img.size(-3) != 3:
            return img, cover_img
        if repeat is False:
            self.now_hue_factor = random.uniform(*self.hue_factor)
            pass
        img = adjust_hue(img, self.now_hue_factor * math.pi)
        if self.clip_out:
            img = torch.clamp(img, 0, 1)
        return img, cover_img     
        pass
    pass


