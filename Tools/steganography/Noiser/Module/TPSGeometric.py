import random
from Tools.steganography.Noiser.Noiser import NoiseLayer, NoiseLayerManager
import torch.nn.functional as F
import itertools
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function, Variable

from Tools.util.ImgUtil import img2tensor
from Tools.util.utils import vutils

# https://github1s.com/WarBean/tps_stn_pytorch/blob/master/mnist_train.py
# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix

class TPSGridGen(nn.Module):

    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate



@NoiseLayerManager.register("TPSDistortion")
class TPSDistortion(NoiseLayer):
    def __init__(self, 
                 num_mesh=8, 
                 img_size=128, 
                 perturbation_rate=[0, 0.5],
                 span_range=1,
                 alpha=0.2,
                 same_on_batch=False,
                 sync_cover_img=True
                 ):
        """基于TPS实现的局部几何变换攻击

        Args:
            num_mesh (int, optional): 控制点个数. Defaults to 8.
            img_size (int, optional): 图像大小. Defaults to 128.
            perturbation_rate (list, optional): 扰动比率, [0,1]之间. Defaults to [0, 0.5].
            span_range (float, optional): _description_. 扰动范围, [0,1]之间.
            alpha (float, optional): 扭曲强度[0, 1]. Defaults to 0.5.
            same_on_batch (bool, optional): 同个batch是否执行相同操作. Defaults to False.
            sync_cover_img (bool, optional): 是否同步cover_img. Defaults to True.
        """
        super().__init__()
        self.num_mesh = num_mesh
        self.img_size = img_size
        self.alpha=alpha
        self.keep_batch_same=same_on_batch
        self.sync_cover_img = sync_cover_img
        self.perturbation_rate = perturbation_rate
        self.max_offset = 2.0  * span_range / (num_mesh)
        r1=r2=span_range
        control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1, 2.0  * r1 / (num_mesh)),
            np.arange(-r2, r2, 2.0  * r2 / (num_mesh)),
        )))
        Y, X = control_points.split(1, dim = 1)
        control_points = torch.cat([X, Y], dim = 1)
        self.register_buffer("control_points", control_points)
        self.tps = TPSGridGen(img_size, img_size, control_points)
      
        pass

    def noise(self, img, cover_img, repeat=False):
        bs = img.size(0)
        if repeat is False:
            control_points = self.control_points
            if self.keep_batch_same:
                control_points = control_points.unsqueeze(0).expand(bs, -1, -1)
                pass
            offset = self.alpha * (torch.rand(self.control_points.size())*2-1) * self.max_offset
            offset = offset.to(img.device)
            p = np.random.uniform(*self.perturbation_rate)
            mask = (torch.rand(offset.size()) < p).float().to(img.device)
            now_offset = offset * mask + control_points
            if self.keep_batch_same is False:
                now_offset = now_offset.unsqueeze(0).expand(bs, -1, -1)  
            self.now_offset = now_offset
        source_coordinate = self.tps(self.now_offset)
        grid = source_coordinate.view(bs, self.img_size, self.img_size, 2)
        img = F.grid_sample(img, grid, align_corners=False)
        if self.sync_cover_img:
            cover_img = F.grid_sample(cover_img, grid, align_corners=False)
        return img, cover_img
        pass
        
    def latest_noise(self, img, cover_img):
        return self.noise(img, cover_img, True)


    def noise_with_grid(self, img, cover_img, repeat=False):
        bs = img.size(0)
        if repeat is False:
            control_points = self.control_points
            if self.keep_batch_same:
                control_points = control_points.unsqueeze(0).expand(bs, -1, -1)
                pass
            offset = self.alpha * (torch.rand(self.control_points.size())*2-1) * self.max_offset
            offset = offset.to(img.device)
            p = np.random.uniform(*self.perturbation_rate)
            mask = (torch.rand(offset.size()) < p).float().to(img.device)
            now_offset = offset * mask + control_points
            if self.keep_batch_same is False:
                now_offset = now_offset.unsqueeze(0).expand(bs, -1, -1)  
            self.now_offset = now_offset
        source_coordinate = self.tps(self.now_offset)
        grid = source_coordinate.view(bs, self.img_size, self.img_size, 2)
        img = F.grid_sample(img, grid, align_corners=False)
        if self.sync_cover_img:
            cover_img = F.grid_sample(cover_img, grid, align_corners=False)
        return img, cover_img, grid
        pass

if __name__ == "__main__":
    # print(_random(-1, 1, size=(2, 2), same_on_batch=False))
    model = TPSDistortion(img_size=256)
    x = img2tensor("/home/light_sun/workspace/ImageST/test/test.png").expand(2, -1, -1, -1)
    x, y = model(x, x)
    vutils().save_image(torch.cat([x, y], dim=0), "./test/affine.png")
    pass