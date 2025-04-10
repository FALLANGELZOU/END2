import random
import warnings
import torch
import torch.nn as nn

from Tools.register.registry import Registry
from Tools.util.ImgUtil import img2tensor
from Tools.util.utils import vutils

# 噪声层管理器
NoiseLayerManager = Registry("noise_layer")

class Noiser(nn.Module):
    def __init__(self, params=[("Identity", None),], normalize=False):
        """噪声组合层，图像需要在[0,1]之间
        Args:
            params (list, optional): 噪声层参数, 格式[(名称，{参数})]. Defaults to [("Identity", None),].
            normalize (bool, optional): 是否自动归一化到[0, 1]. Defaults to False.
        """
        super().__init__()
        self.models = []
        self.normalize = normalize
        self.minn = 0
        self.maxx = 1
        for name, kwargs in params:
            if kwargs is None:
                self.models.append(NoiseLayerManager.get(name)())
            else:
                self.models.append(NoiseLayerManager.get(name)(**kwargs))
            pass
        
        self.models = nn.ModuleList(self.models)
        self.latest_layer = None
        pass

    def forward(self, img, cover_img):
        layer = random.choice(self.models)
        self.latest_layer = layer
        img, cover_img = self._normalize(img, cover_img)
        self._check(img, cover_img)
        img, cover_img = layer.noise(img, cover_img)
        img, cover_img = self._denormalize(img, cover_img)
        return img, cover_img
        pass

    def latest_noise(self, img, cover_img):
        if self.latest_layer is None:
            raise Exception("latest noise is not present")
        img, cover_img = self._normalize(img, cover_img)
        img, cover_img = self.latest_layer.latest_noise(img, cover_img)
        img, cover_img = self._denormalize(img, cover_img)        
        return img, cover_img
        pass

    def _check(self, img, cover_img):
        minn = torch.min(img.min(), cover_img.min())
        maxx = torch.max(img.max(), cover_img.max())
      
        if minn < 0 or maxx > 1:
            warnings.warn("The current tensor range is not expected. Expected at [0,1], currently at [{},{}].".format(minn, maxx))
        pass

    def _normalize(self, img, cover_img):
        if self.normalize:
            self.minn = cover_img.min()
            self.maxx = cover_img.max()
            img = (img - self.minn) / (self.maxx - self.minn)
            cover_img = (cover_img - self.minn) / (self.maxx - self.minn)
            pass
        return img, cover_img
        pass

    def _denormalize(self, img, cover_img):
        if self.normalize:
            img = img * (self.maxx - self.minn) + self.minn
            cover_img = cover_img * (self.maxx - self.minn) + self.minn
            pass
        return img, cover_img
        pass
    pass

# https://applenob.github.io/python/register/


class NoiseLayer(nn.Module):
    '''
    噪声层基类
    '''
    def __init__(self, *args):
        super().__init__()
        pass
    
    def forward(self, img, cover_img):
        return self.noise(img, cover_img)
        pass
    
    def noise(self, img, cover_img):
        raise NotImplementedError("You need to implement the noise addition process")
        pass

    def latest_noise(self, img, cover_img):
        raise NotImplementedError("You need to implement the latest noise addition process")
    pass



def get_default_Noiser():
    """获取默认的噪声层
       几何噪声和非几何噪声分开处理

    Returns:
        _type_: _description_
    """

    geo_noiser = Noiser([
            ("Identity", None),
            ("Crop", None),
            ("Rotate", None),
            ("Translate", None),
            ("Scale", None),
            ("Shear", None),
            # ("TPSDistortion", None) # 暂时不加
            ]
    )

    non_geo_noiser = Noiser([
            ("Identity", None),
            ("Color", None),
            ("KorniaJpeg", None),
            ("GaussianFilter", None),
            ("GaussianNoise", None)
            ]


    )
    return geo_noiser, non_geo_noiser

    pass


def combine_default_noiser():
    return Noiser([
            ("Identity", None),
            ("Crop", None),
            ("Rotate", None),
            ("Translate", None),
            ("Scale", None),
            ("Shear", None),
            
            ("Color", None),
            ("KorniaJpeg", None),
            ("GaussianFilter", None),
            ("GaussianNoise", None)
    ])
    pass


if __name__ =="__main__":
    model = Noiser(
        [
            ("Color", None)
        ]
    )
    img = img2tensor("/home/light_sun/workspace/DWSF/DIV2K/DIV2K_train/0004.png")
    img = model(img)
    vutils().save_image(img, "./test.png")
    pass