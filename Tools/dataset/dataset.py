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

class LDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()
        pass

    def __len__(self):

        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
    pass

def read_img(img_path, mode="RGB"):
    """使用PIL读取Image
    """
    return Image.open(img_path).convert(mode)


def gen_default_img_transform():
    """产生默认的img转换，范围[0, 1]
    """
    return transforms.Compose([
            transforms.ToTensor()
        ])
    pass

def get_dataset_class():
    """返还pytorch的Dataset类
    """
    return Dataset

def get_transforms():
    """返还pytorch的transforms
    """
    return transforms