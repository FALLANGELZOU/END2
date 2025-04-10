from torch import nn
import torch
import numpy as np
import math
class TimeEmbedding(nn.Module):
    """时序序列的embedding
    """
    def __init__(self, in_dim, out_dim, act="cos"):
        """初始化

        Args:
            in_dim : 输入维度
            out_dim (_type_): 输出维度
            act (str, optional): 使用的模式 Defaults to "cos".
        """
        super(TimeEmbedding, self).__init__()
        if act == 'cos':
            self.emb = CosineActivation(in_dim, out_dim)
        if act == 'sin':
            self.emb = SineActivation(in_dim, out_dim)
        pass
    
    def forward(self, x):
        return self.emb(x)
    
    pass

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)




    
if __name__ == "__main__":
    x = torch.randn(4, 10, 4)
    model = TimeEmbedding(4, 128)
    print(model(x).size())
    