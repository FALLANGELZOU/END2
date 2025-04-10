# -*- coding: utf-8 -*-
"""
@FILE    :   __init__.py.py
@DATE    :   2023/7/4 20:27
@Author  :   Angel_zou 
@Contact :   ahacgn@gmail.com
@docs    :   
"""
import torch
import torchvision

# PyTorch version as a tuple of 2 ints. Useful for comparison.
TORCH_VERSION = tuple(int(x) for x in torch.__version__.split(".")[:2])
TORCHVISION_VERSION = tuple(int(x) for x in torchvision.__version__.split(".")[:2])

