# -*- coding: utf-8 -*-
"""
@FILE    :   utils.py
@DATE    :   2023/7/4 20:31
@Author  :   Angel_zou 
@Contact :   ahacgn@gmail.com
@docs    :   
"""
from fnmatch import fnmatch
import math
import random
from shutil import copy
import shutil

import torch
import yaml
from torch.nn import init
import os
import numpy as np
from varname import nameof
import torchvision.utils as _vutils
import torch.nn as nn
def vutils():
    return _vutils

def load_param_dict(model, params):
    """
    动态加载模型
    :param model:
    :param params:
    :return:
    """
    model_dict = model.state_dict()
    params = {k: v for k, v in params.items() if k in model_dict}
    model_dict.update(params)
    model.load_state_dict(model_dict)
    pass


def one_hot(index, num_class):
    """
    转成one hot 形式
    :param index:
    :param num_class:
    :return:
    """
    return torch.nn.functional.one_hot(index, num_class).float()
    pass


def load_yaml(config):
    """
    加载yaml
    :param config:
    :return:
    """
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def weights_init(init_type='kaiming'):
    """
    模型初始化
    :param init_type:
    :return:
    """

    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


def root_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pass

def random_label(shape, label_set):
    """
    随机生成形状为shape的tensor，值从label_set中随机选择
    :param shape: tensor形状
    :param label_set: tensor取值集合
    :return:
    """
    return torch.Tensor(np.random.choice(label_set, shape))
    pass

    pass


def element_division(lst, divisor):
    """
    对列表（list）或元组（tuple）中的每个元素进行除法运算
    :param lst: 列表或元组
    :param divisor: 除数
    :return: 结果列表
    """
    result = []
    for num in lst:
        result.append(num / divisor)
    return result


def element_add(x, y):
    assert len(x) == len(y)
    result = []
    for i in range(len(x)):
        result.append(x[i] + y[i])
        pass
    return result
    pass


def L2RGB(image):
    if len(image.shape) == 3:
        gray_tensor = image
        rgb_tensor = torch.cat((gray_tensor, gray_tensor, gray_tensor), dim=0)
        return rgb_tensor
        pass
    assert len(image.shape) == 4 and image.shape[1] == 1
    gray_tensor = image
    rgb_tensor = torch.cat((gray_tensor, gray_tensor, gray_tensor), dim=1)
    return rgb_tensor
    pass


def RGB2L(image):
    assert len(image.shape) == 4 and image.shape[1] == 3
    rgb_tensor = image
    r, g, b = rgb_tensor[:, 0, :, :].unsqueeze(1), rgb_tensor[:, 1, :, :].unsqueeze(1), rgb_tensor[:, 2, :,
                                                                                        :].unsqueeze(1)
    gray_tensor = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray_tensor
    pass

def RGBA2RGB(img, background=(1.0, 1.0, 1.0), norm=False):
    extr = False
    if len(img.size()) == 3:
        img = img.unsqueeze(0)
        extr = True
        pass
    if norm:
        th_max = torch.max(img)
        th_min = torch.min(img)
        img_rng = th_max - th_min
        if img_rng > 0:
            img = (img - th_min) / img_rng
            img = torch.clamp(img, 0.0, 1.0)
            
    r,g,b,a = img.split(1, dim=1)
    r = r * a + (1-a)*background[0]
    g = g * a + (1-a)*background[1]
    b = b * a + (1-a)*background[2]
    img = torch.cat([r,g,b], dim=1)
    
    if norm:
        if img_rng > 0:
            img = img * img_rng + th_min
        
    if extr:
        img = img.squeeze(0)
    return img

def seed_everything(seed):
    '''
    freeze seed
    :param seed: seed number
    :return: None
    '''
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pass
  
def save_programme(save_path):
    
    def copy_contents(source_item, destination_item, exclude_patterns=None):
        try:
            if exclude_patterns and any((fnmatch(os.path.basename(source_item), pattern) or os.path.split(source_item)[-1] == pattern) for pattern in exclude_patterns):
                return
                pass
            if os.path.isdir(source_item):
                if not os.path.exists(destination_item):
                    os.makedirs(destination_item)

                for item in os.listdir(source_item):
                    source_subitem = os.path.join(source_item, item)
                    destination_subitem = os.path.join(destination_item, item)
                    copy_contents(source_subitem, destination_subitem, exclude_patterns)

            else:
                shutil.copy2(source_item, destination_item)
        except Exception as e:
            print(f"An error occurred when copy files: {e}")
            
    root = root_path()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        pass  
    config = os.path.join(root, ".save_whitelist.yaml")
    if not os.path.exists(config):
        with open(config, 'w') as f:
            f.write("save:\nexclude:\n")
            pass
        pass
    else:
        config_yaml = load_yaml(config)
        save_list = config_yaml['save']
        exclude_list = config_yaml['exclude']
        for item in save_list:
            copy_contents(os.path.join(root, item), os.path.join(save_path, item), exclude_list)
            pass
    pass

def PSNR(x, y):
    """PSNR, db
       the input should be between 0 and 1
       
    Args:
        x (tensor): _description_
        y (tensor): _description_
    Returns:
        _type_: _description_
    """
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return torch.tensor(100).float()
    PIXEL_MAX = 1
    return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    pass
  
def random_float(min, max):
    """
    Return a random number
    :param min:
    :param max:
    :return:
    """
    return np.random.rand() * (max - min) + min
 
def cal_params(model, only_trainable=True, need_print=False):
        if only_trainable:
            params =  sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
            pass
        else:
            params = sum(p.numel() for p in model.parameters()) / 1e6
        if need_print:
            print("params:{:.2f}M".format(params))
            pass
        return params
        pass
    
def model_freeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False
        pass
    pass

def model_unfreeze(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
        pass
    pass




if __name__ == "__main__":
    # save_programme("F:\python\workSpace\InvertedFont\output\T1\InvertedFont")
    current_directory = os.path.dirname(os.path.abspath(__file__))
    print(current_directory)
    pass
