from einops import rearrange
import numpy as np
import torch
import torch.nn as nn

from Tools.util.utils import vutils
import numpy as np
import math
try:
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))
    
    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

    def idct_irfft_impl(V):
        return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
except ImportError:
    # PyTorch 1.6.0 and older versions
    def dct1_rfft_impl(x):
        return torch.rfft(x, 1)
    
    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)

    def idct_irfft_impl(V):
        return torch.irfft(V, 1, onesided=False)



def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def idct_3d(X, norm=None):
    """
    The inverse to 3D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_3d(dct_3d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    x3 = idct(x2.transpose(-1, -3), norm=norm)
    return x3.transpose(-1, -3).transpose(-1, -2)


class LinearDCT(nn.Linear):
    """Implement any DCT as a linear layer; in practice this executes around
    50x faster on GPU. Unfortunately, the DCT matrix is stored, which will 
    increase memory usage.
    :param in_features: size of expected input
    :param type: which dct function in this file to use"""
    def __init__(self, in_features, type, norm=None, bias=False):
        self.type = type
        self.N = in_features
        self.norm = norm
        super(LinearDCT, self).__init__(in_features, in_features, bias=bias)

    def reset_parameters(self):
        # initialise using dct function
        I = torch.eye(self.N)
        if self.type == 'dct1':
            self.weight.data = dct1(I).data.t()
        elif self.type == 'idct1':
            self.weight.data = idct1(I).data.t()
        elif self.type == 'dct':
            self.weight.data = dct(I, norm=self.norm).data.t()
        elif self.type == 'idct':
            self.weight.data = idct(I, norm=self.norm).data.t()
        self.weight.requires_grad = False # don't learn this!


def apply_linear_2d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 2D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 2 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    return X2.transpose(-1, -2)

def apply_linear_3d(x, linear_layer):
    """Can be used with a LinearDCT layer to do a 3D DCT.
    :param x: the input signal
    :param linear_layer: any PyTorch Linear layer
    :return: result of linear layer applied to last 3 dimensions
    """
    X1 = linear_layer(x)
    X2 = linear_layer(X1.transpose(-1, -2))
    X3 = linear_layer(X2.transpose(-1, -3))
    return X3.transpose(-1, -3).transpose(-1, -2)


def split_blocks(img, block_size=8):
    bs, c, h, w = img.size()
    # 使用 unfold 方法切块
    blocks = img.unfold(2, block_size, block_size).unfold(3, block_size, block_size)

    # 调整形状以得到期望的结果
    blocks = blocks.contiguous().view(bs, c, -1, block_size, block_size).contiguous()
    blocks = rearrange(blocks, "bs c n h w -> bs n c h w")
    return blocks
    pass


def merge_blocks(blocks):
    blocks = rearrange(blocks, "bs n c h w -> bs c n h w")
    n = blocks.size(2)
    m = int(math.sqrt(n))
    img = rearrange(blocks, "bs c (x y) h w -> bs c (x h) (y w)", x=m, y=m)
    return img
    pass
def zigzag_indices(n):
    '''
    生成n*n的zigzag编码索引，以及反向索引
    '''
    data = np.arange(n*n).reshape((n, n))
    row = data.shape[0]
    col = data.shape[1]
    num = row * col
    list = np.zeros(num,)
    k = 0
    i = 0
    j = 0
 
    while i < row and j < col and k < num:
        list[k] = data.item(i, j)
        k = k + 1
        # i + j 为偶数, 右上移动. 下面情况是可以合并的, 但是为了方便理解, 分开写
        if (i + j) % 2 == 0:
            # 右边界超出, 则向下
            if (i-1) in range(row) and (j+1) not in range(col):
                i = i + 1
            # 上边界超出, 则向右
            elif (i-1) not in range(row) and (j+1) in range(col):
                j = j + 1
            # 上右边界都超出, 即处于右上顶点的位置, 则向下
            elif (i-1) not in range(row) and (j+1) not in range(col):
                i = i + 1
            else:
                i = i - 1
                j = j + 1
        # i + j 为奇数, 左下移动
        elif (i + j) % 2 == 1:
            # 左边界超出, 则向下
            if (i+1) in range(row) and (j-1) not in range(col):
                i = i + 1
            # 下边界超出, 则向右
            elif (i+1) not in range(row) and (j-1) in range(col):
                j = j + 1
            # 左下边界都超出, 即处于左下顶点的位置, 则向右
            elif (i+1) not in range(row) and (j-1) not in range(col):
                j = j + 1
            else:
                i = i + 1
                j = j - 1
    indices = torch.tensor(list, dtype=torch.long)
    reverse_indices = torch.empty_like(indices)
    reverse_indices[indices] = torch.arange(indices.size(0), dtype=torch.long)
    return indices, reverse_indices
 


def fft_2d(x):
    '''
    返还频率谱和相位谱
    '''
    z = torch.fft.fft2(x)
    z = torch.fft.fftshift(z, dim=[-2,-1])
    # 计算频率谱和相位谱
    magnitude_spectrum = torch.log1p(torch.abs(z))  # 使用对数缩放
    phase_spectrum = torch.angle(z)
    return magnitude_spectrum, phase_spectrum
    pass

def ifft_2d(magnitude_spectrum, phase_spectrum):
    # 重建傅里叶变换的复数表示
    reconstructed_fft_images = torch.polar(torch.exp(magnitude_spectrum), phase_spectrum)
    # 使用ifftshift将频谱移回原位置
    unshifted_fft_images = torch.fft.ifftshift(reconstructed_fft_images, dim=[-2, -1])
    # 计算二维逆傅里叶变换
    reconstructed_images = torch.fft.ifft2(unshifted_fft_images)
    # 取实部作为重建后的图像
    reconstructed_images = reconstructed_images.real
    return reconstructed_images
    pass

class DCT2D_Layer(nn.Module):
    def __init__(self, block_size=8):
        super().__init__()
        self.block_size = block_size
        indices, reverse_indices = zigzag_indices(block_size)
        self.register_buffer("indices", indices)
        self.register_buffer("reverse_indices", reverse_indices)
        pass

    def dct(self, img):
        '''
        返还dct变换的结果 
        输入：[bs, c, h, w]
        输出：[bs, n, c, block_h, block_w]，其中n是分了多少个block
        '''
        blocks = split_blocks(img, self.block_size)
        n = blocks.size(1)
        blocks = rearrange(blocks, "bs n c h w -> (bs n) c h w")
        z = dct_2d(blocks)
        z = rearrange(z, "(bs n) c h w -> bs n c h w",n=n)
        return z.contiguous()
        pass

    def idct(self, z):
        '''
        dct反变换
        输入：[bs, n, c, block_h, block_w]
        输出：[bs, c, h, w]
        '''
        n = z.size(1)
        z = rearrange(z, "bs n c h w -> (bs n) c h w")
        blocks = idct_2d(z)
        blocks = rearrange(blocks, "(bs n) c h w -> bs n c h w", n=n)
        return merge_blocks(blocks).contiguous()
        pass

    def frequencyFeat(self, blocks):
        '''
        生成频谱特征图。按照频率高低排序，拼成c * block_h * block_w个feature
        输入：[bs, n, c, block_h, block_w]
        输出：[bs, c * block_h * block_w, m, m] 其中m是sqrt(n)
        '''
        n = blocks.size(1)
        m = int(math.sqrt(n))
        blocks = rearrange(blocks, "bs n c h w -> bs n c (h w)")
        blocks = blocks[:,:,:, self.indices].contiguous()
        feat = rearrange(blocks, "bs (h w) c c2 -> bs (c c2) h w", h=m, w=m)
        return feat.contiguous()
        pass

    def reverseFeat(self, feat):
        '''
        对频谱特征图进行反变换，让其变回blocks
        输入：[bs, c * c2, feat_h, feat_w]
        输出：[bs, n, c, block_h, block_w]
        '''
        k = feat.size(1)
        c2 = (self.block_size * self.block_size)
        c = k // c2
        
        feat = rearrange(feat, "bs (c c2) h w -> bs (h w) c c2", c=c, c2=c2)
        feat = feat[:,:,:,self.reverse_indices].contiguous()
        blocks = rearrange(feat, "bs n c (h w) -> bs n c h w", h=self.block_size, w=self.block_size)
        return blocks.contiguous()
        pass

    pass
if __name__ == '__main__':
    from PIL import Image
    import random
    import math
    from torchvision import transforms
    image = Image.open("/home/light_sun/workspace/DWSF/DIV2K/DIV2K_train/0010.png")
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((2048, 2048), antialias=True)
    ])
    block_size=64
    model = DCT2D_Layer(block_size)
    image = trans(image).unsqueeze(0)
    vutils().save_image(dct_2d(image), "dct_image.png", normalize=True)
    z = model.dct(image)
    z_img = rearrange(z, "bs n c h w -> (bs n) c h w")
    vutils().save_image(z_img, "z.png", normalize=True, nrow=256//block_size)
    feat = model.frequencyFeat(z)
    print(feat.size())
    tc = feat.size(1)
    tmp_feat = rearrange(feat, "bs (c c2) h w -> (bs c2) c h w", c=3, c2=tc//3)
    vutils().save_image(tmp_feat, "frequencyFeat.png", normalize=True, nrow=block_size)
    vutils().save_image(tmp_feat-torch.mean(tmp_feat, dim=1,keepdim=True), "frequencyFeat_res.png", normalize=True, nrow=block_size)
    z = model.reverseFeat(feat)
    image = model.idct(z)
    vutils().save_image(image, "image.png", normalize=True)

    x, y = fft_2d(image)
    image = ifft_2d(x, y)
    vutils().save_image(x, "magnitude_spectrum.png", normalize=True)
    vutils().save_image(image, "fft_image.png", normalize=True)

    z = model.dct(x)
    z_img = rearrange(z, "bs n c h w -> (bs n) c h w")
    vutils().save_image(z_img, "z.png", normalize=True, nrow=256//block_size)
    feat = model.frequencyFeat(z)
    print(feat.size())
    tc = feat.size(1)
    tmp_feat = rearrange(feat, "bs (c c2) h w -> (bs c2) c h w", c=3, c2=tc//3)
    vutils().save_image(tmp_feat, "fft_frequencyFeat.png", normalize=True, nrow=block_size)
    # image = split_blocks(image, 8)
    # merge_image = merge_blocks(image)
    # vutils().save_image(merge_image, "merge_image.png", normalize=True)
    # z = dct_2d(image)
    # vutils().save_image(z, "z.png", normalize=True)
    # print(z.size())
    # x = idct_2d(z)
    # vutils().save_image(x, "dct_image.png", normalize=True,nrow=256//8)
