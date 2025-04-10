import random
import math
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torchvision.transforms.functional as F2
import torch.nn.functional as F
from Tools.util.utils import vutils

def gen_random_msg(n):
    """随机产生n位二进制bit串
    """
    msg = np.random.choice([0, 1], (n, ))
    return torch.from_numpy(msg).float()
    pass


def msg_acc(predict_msg, msg):
    """计算平均准确率
    """
    decoded_rounded = torch.round(predict_msg).clamp(0, 1).detach()
    bit_err = torch.abs(decoded_rounded - msg).sum() / (msg.numel())
    acc = 1 - bit_err
    return acc
    pass