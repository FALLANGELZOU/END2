import numpy as np
from torch import Tensor
import torch.nn as nn
import collections
import torch
import torch.nn.functional as F
# https://zhuanlan.zhihu.com/p/408474710
# https://changyaochen.github.io/welford/
def welford_update(count, mean, M2, currValue):
    count += 1
    delta = currValue - mean
    mean += delta / count
    delta2 = currValue - mean
    M2 += delta * delta2
    return (count, mean, M2)

class Welford(nn.Module):
    def __init__(self):
        super(Welford, self).__init__()
        # 这个式子貌似只能保证一个数慢慢算
        pass
    pass


# 合并两组数的均值和方差
# https://blog.csdn.net/u014250897/article/details/106195771
# https://www.cnblogs.com/mercurysun/p/17226989.html
import math
import numpy as np

class DequeMeanVar(nn.Module):
    def __init__(self, size=5000):
        """
        存在队列中的值会导致内存溢出吗
        """
        super(DequeMeanVar, self).__init__()
        self.deque = collections.deque()
        self.size = size
        self.mean = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.var = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.m = 0

    def forward(self, x: Tensor):
        n = x.numel()
        var, mean = torch.var_mean(x)
        new_m, total_mean, total_var = self.merge_mean_var(n, mean, var, self.m, self.mean, self.var)
        self.m = new_m
        self.mean.data = total_mean.clone().detach()
        self.var.data = total_var.clone().detach()
        self.deque.append((n, mean.clone().detach(), var.clone().detach()))
        if len(self.deque) > self.size:
            n, mean, var = self.deque.popleft()
            self.m, self.mean.data, self.var.data = self.split_mean_var(n, mean, var, self.m, self.mean, self.var)
        return total_mean, total_var
    
    
    def merge_mean_var(self, n, mean1, var1, m, mean2, var2):
        mean = (n * mean1 + m * mean2) / (m + n)
        var = (n * (var1 + mean1**2) + m * (var2 + mean2**2))/(m + n) - mean**2
        return m+n, mean, var
    
    def split_mean_var(self, n, mean1, var1, m, mean, var):
        m -= n
        mean2 = (m+n)/ m * mean - n / m * mean1
        var2 = ((m + n) * (var + mean ** 2) - n * (var1 + mean1 ** 2)) / m - mean2 ** 2
        return m, mean2, var2
        pass
    pass

if __name__ == '__main__':
    model = DequeMeanVar(500)
    
    for i in range(10000):
        mean, var = model(torch.randn(5, 10, requires_grad=True))
        print(mean, var)
        loss = F.mse_loss(mean, torch.zeros_like(mean).fill_(0.5)) + F.mse_loss(var, torch.zeros_like(var).fill_(1/12))
        loss.backward()
    
        pass
    print(model.mean, model.var)




