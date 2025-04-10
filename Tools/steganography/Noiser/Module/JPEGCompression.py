import warnings
import torch
from Tools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager
from kornia.augmentation import RandomJPEG
from kornia.enhance import jpeg_codec_differentiable
import numpy as np

from Tools.util.ImgUtil import img2tensor
from Tools.util.utils import vutils

@NoiseLayerManager.register("KorniaJpeg")
class KorniaJpeg(NoiseLayer):
    '''
    kornia实现的JPEG压缩模拟
    '''
    def __init__(self, min_q=50, max_q=100, same_on_batch=False):
        super().__init__()
        
        warnings.filterwarnings("ignore") # UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
        self.min_q = min_q
        self.max_q = max_q
        self.same_on_batch = same_on_batch
        self.now_q = 100
        pass

    def noise(self, img, cover_img, repeat=False):
        if repeat is False:
            if self.same_on_batch:
                self.now_q = np.random.uniform(self.max_q, self.min_q)
                self.now_q = torch.Tensor([self.now_q]).to(img.device)
            else:
                self.now_q = np.random.rand(img.size(0), ) * (self.max_q - self.min_q) + self.min_q
                self.now_q = torch.Tensor(self.now_q).to(img.device)
        img = jpeg_codec_differentiable(img, self.now_q)
        return img, cover_img
    
    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, True)
    pass

if __name__ == "__main__":
    model = KorniaJpeg()
    x = img2tensor("/home/light_sun/workspace/DWSF/DIV2K/DIV2K_train/0001.png")
    x = model(x)
    vutils().save_image(x, "./test/test.png")
    pass