import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from Tools.util.ImgUtil import img_normalize, rgb2yuv, yuv2rgb
from Tools.util.utils import vutils
from dataset.INWDataset import INWDataset

class RGB2YUVLayer(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
        pass

    def forward(self, img):
        minn = img.min()
        maxx = img.max()
        if self.normalize:
            img = img_normalize(img, 0, 255)
            pass
        img = rgb2yuv(img)
        if self.normalize:
            img = img_normalize(img, minn, maxx)
            pass
        return img
        pass
    pass

class YUV2RGBLayer(nn.Module):
    def __init__(self, normalize=True):
        super().__init__()
        self.normalize = normalize
        pass

    def forward(self, img):
        minn = img.min()
        maxx = img.max()
        if self.normalize:
            img = img_normalize(img, 0, 255)
            pass
        img = yuv2rgb(img)
        if self.normalize:
            img = img_normalize(img, minn, maxx)
            pass
        return img
        pass
    pass



class HaarTransform(nn.Module):
    '''
    Haar小波变换
    '''
    def __init__(self, channel_in):
        super(HaarTransform, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False
    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac
    
if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    bs = 4
    block_size = 64
    dataset = INWDataset(
        "/mnt/xsj2023/Datasets/DIV2K/DIV2K_train", 
        )
    dataloader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=0)
    yuv_layer = RGB2YUVLayer()
    rgb_layer = YUV2RGBLayer()
    print(1)
    for batch in dataloader:
        image_lr = batch['image_lr']
        image_hr = batch['image_hr']
        img = image_lr
        # img = yuv_layer(img)
        # img = rgb_layer(img)
        print(img.size())
        vutils().save_image(img, "img.png", normalize=True)
        break
        pass
    pass


class ShuffleLayer(nn.Module):
    '''
    训练时，对特征图随机循环左移和循环下移
    '''
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, shift_up=None, shift_left=None, shuffle=True):
        if self.training and shuffle:
            bs, c, h, w = x.size()
            if shift_up is None:
                shift_up = random.randint(0, h)
            if shift_left is None:
                shift_left = random.randint(0, w)
            x = torch.cat((x[:, :, -shift_up:, :], x[:, :, :-shift_up, :]), dim=2)
            x = torch.cat((x[:, :, :, -shift_left:], x[:, :, :, :-shift_left]), dim=3)
            # x = torch.roll(x, shifts=-shift_up, dims=2)
            # x = torch.roll(x, shifts=-shift_left, dims=3)
            return x
            pass
        else:
            return x
        pass
    pass