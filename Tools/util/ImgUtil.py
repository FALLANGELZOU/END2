
import torch
import numpy as np
from torch import nn
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

from Tools.util.utils import vutils

def clip_psnr(container: torch.tensor, host: torch.Tensor, psnr: float = 35, over_clip=False):
    '''
    固定psnr值来调整嵌入强度
    over_clip: 如果container的PSNR好于设定值，是否覆盖掉，默认不覆盖
    '''
    # container嵌入水印的图片，host原图    
    data_range = host.max() - host.min ()
    target_mse = data_range ** 2 / 10 ** (psnr / 10)
    residual = container - host
    factor = torch.sqrt(target_mse / torch.mean(residual ** 2))
    if factor >= 1:
        if over_clip is False:
            factor = 1
        pass
    return host + residual * factor  



# https://github.com/Orange-OpenSource/Cool-Chic/blob/cdd8d8d8627f67322b8e8722c6ec5b2681382f9a/src/utils/yuv.py#L229
def rgb2yuv(rgb):
    """Convert a 4D RGB tensor [1, 3, H, W] into a 4D YUV444 tensor [1, 3, H, W].
    The RGB and YUV values are in the range [0, 255]

    Args:
        rgb (Tensor): 4D RGB tensor to convert in [0. 255.]

    Returns:
        Tensor: the resulting YUV444 tensor in [0. 255.]
    """
    assert len(rgb.size()) == 4, f'rgb2yuv input must be a 4D tensor [B, 3, H, W]. Data size: {rgb.size()}'
    assert rgb.size()[1] == 3, f'rgb2yuv input must have 3 channels. Data size: {rgb.size()}'

    # Split the [1, 3, H, W] into 3 [1, 1, H, W] tensors
    r, g, b = rgb.split(1, dim=1)

    # Compute the different channels
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b  + 128
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    # Concatenate them into the resulting yuv 4D tensor.
    yuv = torch.cat((y, u, v), dim=1)
    return yuv

def yuv2rgb(yuv):
    """Convert a 4D YUV tensor [1, 3, H, W] into a 4D RGB tensor [1, 3, H, W].
    The RGB and YUV values are in the range [0, 255]

    Args:
        rgb (Tensor): 4D YUV444 tensor to convert in [0. 255.]

    Returns:
        Tensor: the resulting RGB tensor in [0. 255.]
    """
    assert len(yuv.size()) == 4, f'yuv2rgb input must be a 4D tensor [B, 3, H, W]. Data size: {yuv.size()}'
    assert yuv.size()[1] == 3, f'yuv2rgb input must have 3 channels. Data size: {yuv.size()}'

    y, u, v = yuv.split(1, dim=1)
    r = 1.0 * y + -0.000007154783816076815 * u + 1.4019975662231445 * v - 179.45477266423404
    g = 1.0 * y + -0.3441331386566162 * u + -0.7141380310058594 * v + 135.45870971679688
    b = 1.0 * y + 1.7720025777816772 * u + 0.00001542569043522235 * v - 226.8183044444304
    rgb = torch.cat((r, g, b), dim=1)
    return rgb

def img_normalize(img, a=-1, b=1):
    """
    将张量归一化到 [a, b] 范围内。

    参数：
    tensor (torch.Tensor): 输入张量
    a (float): 范围下限
    b (float): 范围上限

    返回：
    torch.Tensor: 归一化后的张量
    """
    # 计算最小值和最大值
    min_val = torch.min(img)
    max_val = torch.max(img)

    # 最小-最大归一化到 [0, 1]
    norm_tensor = (img - min_val) / (max_val - min_val)

    # 缩放到 [a, b]
    scaled_tensor = a + (b - a) * norm_tensor
    
    return scaled_tensor


def img2tensor(img_path, mode='RGB', img_size=(256, 256)):
    '''
    将目标路径的转换为tensor,供快速测试，数值范围[0, 1]
    return: [1, c, h, w]
    '''
    from torchvision import transforms
    from PIL import Image
    if img_size == None:
        trans = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(
                    size=img_size, 
                    antialias=True, 
                    interpolation=transforms.InterpolationMode.BICUBIC
            )
            ])
    img = Image.open(img_path).convert(mode)
    img = trans(img)
    img = torch.clamp(img, 0, 1).unsqueeze(0)
    return img
    pass


def soble_conv(img):
    """可微分soble算子

    Args:
        img (_type_): _description_

    Returns:
        _type_: _description_
    """
    bs, c, h, w = img.size()
    # 用nn.Conv2d定义卷积操作
    conv_op = nn.Conv2d(c, c, 3, bias=False, groups=c, padding=1).to(img.device)
    # print(conv_op.weight.data.size())
    # 定义sobel算子参数
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')
    # 将sobel算子转换为适配卷积操作的卷积核
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    # 给卷积操作的卷积核赋值
    conv_op.weight.data = torch.from_numpy(sobel_kernel).expand(c, 1, 3, 3).to(img.device)
    # 对图像进行卷积操作
    edge_detect = conv_op(img)
    return edge_detect
    pass


if __name__ == "__main__":
    x = img2tensor("/home/light_sun/workspace/ImageST/test/test.png")
    edge = soble_conv(x)
    vutils().save_image(edge, "./test/soble.png")
    pass