import torch
def h(A, B):
    """二维平面上两点集的豪多斯距离
    https://zhuanlan.zhihu.com/p/351921396#%E4%B8%80.%20Hausdorff%20distance%E4%BB%8B%E7%BB%8D
    Args:
        A (_type_): [n, 2] n个2D点集
        B (_type_): [n, 2] n个2D点集
    """
    distance = torch.cdist(A, B)
    min_distance, _ = distance.min(dim=1)
    max_min_distance = min_distance.max()
    return max_min_distance
    pass

def H(A, B):
    """双向豪多斯距离

    Args:
        A (_type_): _description_
        B (_type_): _description_

    Returns:
        _type_: _description_
    """
    return torch.max(h(A, B), h(B, A))
    pass