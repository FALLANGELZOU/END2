from pytorch_msssim import ssim, ms_ssim
from pytorch_msssim import SSIM as pytorch_ssim
from pytorch_msssim import MS_SSIM as pytorch_ms_ssim




def SSIM(x, y, multi_scale=False):
    """计算SSIM，输入范围[0, 1]
    """
    if multi_scale:
        return ms_ssim(x, y, data_range=1.0, size_average=True)
    else:
        return ssim(x, y, data_range=1.0, size_average=True)
    pass

def ssim_loss(x, y, channel=3, multi_scale=False):
    """计算SSIM loss，输入范围[0, 1]
    """
    if multi_scale:
        ssim_loss = pytorch_ms_ssim(data_range=1.0, size_average=True, channel=channel)
    else:
        ssim_loss = pytorch_ssim(data_range=1.0, size_average=True, channel=channel)
    return 1 - ssim_loss(x, y)
    pass