import torch
import torch.nn as nn

from Tools.module.noiser.Noiser import NOISE_LAYER
from Tools.util.utils import vutils
@NOISE_LAYER.register("diff_binarize")
class DiffBinarization(nn.Module):
    def __init__(self, threshold=0, k=50):
        """可微分二值化，输入需要在[-1, 1]之间
        Args:
            threshold (int, optional): 阈值
        """
        super(DiffBinarization, self).__init__()
        self.td = threshold
        self.k = k
        pass
    def forward(self, x):
        x = torch.clamp(x, -1., 1.)
        x = (1 / (1+torch.exp(-self.k * (x-self.td)))) * 2 - 1
        # binarize_x = torch.where(x>=0, 1, -1)
        return x # x + (binarize_x - x).detach()
        pass
    pass

