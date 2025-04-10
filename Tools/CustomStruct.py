
from typing import Any
import torch.nn as nn

class CustomDict(dict):
    def __missing__(self, key):
        return None
    
    def __getitem__(self, __key: Any) -> Any:
        try:
            res = super().__getitem__(__key)
        except:
            res = None
            pass
        return res
    
class Flatten(nn.Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

class FeatureListAdapter(nn.Module):
    """
    用来过滤特征列表
    当keep_list=False，并且只选取一个特征的时候直接返回特征，否则返回特征列表
    """
    def __init__(self, idxs=[0], keep_list=False):
        super(FeatureListAdapter, self).__init__()
        self.idxs = idxs
        self.keep_list = keep_list
        pass
    def forward(self, x):
        if len(self.idxs) == 1:
            if self.keep_list:
                return x
            return x[self.idxs[0]]
        return [x[i] for i in self.idxs]
        pass
    pass
