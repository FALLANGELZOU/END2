import math

from torch import nn
import torch
import torch.nn.functional as F
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=-1e9):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        # 2. apply masking (opt)
        if mask is not None:
            # print(score.size())
            score = score.masked_fill(mask==0, e)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)
        
        # 4. multiply with Value
        v = score @ v
        
        return v, score
    
class ConvMultiHeadAttention(nn.Module):
    """使用1d卷积增强上下文建模的attention
    """
    def __init__(self, d_model, n_head, kernel_size = 3):
        """

        Args:
            d_model (_type_): 序列中每个元素的维度
            n_head (_type_): 注意力头的个数
            kernel_size (int, optional): 1d卷积的核大小 Defaults to 3.
        """
        super(ConvMultiHeadAttention, self).__init__()
        if kernel_size == 3:
            pad = 1
        if kernel_size == 1:
            pad = 0
        if kernel_size == 5:
            pad = 5
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        h = self.n_head
        self.w_q = nn.Conv1d(d_model, d_model, kernel_size, 1, pad)  
        self.w_k = nn.Conv1d(d_model, d_model, kernel_size, 1 ,pad)  
        self.w_v = nn.Conv1d(d_model, d_model, 1, 1, 0)  
        
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        """计算attention
        
        本质上是说，我计算一个v的加权线性组合，

        对于q中的每个元素a，通过k和q计算v中哪些元素和a最相关
        
        然后加权求和

        最后的输出中，每个元素都是v的线性组合，与q元素一一对应
        
        Args:
            q (_type_): _description_
            k (_type_): _description_
            v (_type_): _description_ 通常k与v要保持一致
            mask (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: 与q形状相同的v的线性组合
        """
        # print(self.w_q.weight.data)
        b, s, d = q.size()
        h = self.n_head
        d = d // h
        q = q.transpose(1, 2) # [b, dim, seq]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k, v = self.w_q(q).view(b, d, h, s), self.w_k(k).view(b, d, h, s), self.w_v(v).view(b, d, h, s)
        q = q.view(b, d, self.n_head, s).transpose(1, 2).transpose(2, 3) # [b, h, s, d]
        k = k.view(b, d, self.n_head, s).transpose(1, 2).transpose(2, 3)
        v = v.view(b, d, self.n_head, s).transpose(1, 2).transpose(2, 3)
        out, attention = self.attention(q, k, v, mask=mask)

        out = self.concat(out)
        out = self.w_concat(out)

        return out


    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
class MultiHeadAttention(nn.Module):
    """普通的attention
    """
    def __init__(self, d_model, n_head):
        """
        Args:
            d_model (_type_): 序列中元素的维度
            n_head (_type_): 注意力头的个数
        """
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
    
    

class GroupNorm(nn.Module):
    '''
    A nice alternative to batch normalization. It is more suitable for small
    batch size, and it is also more stable than batch normalization. Generally
    good for vision problems with intensive memory usage.
    '''
    def __init__(self, channels) -> None:
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        return self.gn(x)

class Swish(nn.Module):
    """
    An activation function attained by Neural Architecture Search (NAS) with
    a little bit of modification. It is a smooth approximation of ReLU. Its
    most distinctive property is that has a non-monotonic "bump" and has the
    following properties:
    - Non-monotonicity
    - Unboundedness
    - Smoothness
    """
    def forward(self, x):
        return x * torch.sigmoid(x)
    
class NonLocalBlock(nn.Module):
    '''卷积网络的attention机制
    Non-local block, used for long-range dependencies. It is a generalization of
    the self-attention mechanism. See,
    Wang, Xiaolong, et al. "Non-local neural networks."
    Proceedings of the IEEE conference on computer vision and
    pattern recognition. 2018. 
    '''
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels
        
        self.group_norm = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        # self.projection = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x):
        hidden = self.group_norm(x)
        q = self.q(hidden)
        k = self.k(hidden)
        v = self.v(hidden)
        
        # some reshaping for matrix multiplication
        b, c, h, w = q.shape
        
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        
        attn = torch.bmm(q, k) # batch matrix multiplication for attention
        attn = attn * (int(c) ** (-0.5)) # scaling factor
        print(torch.isnan(attn).any())
        attn = F.softmax(attn, dim=2) # softmax along the last dimension to get the probabilies as attention weights
        attn = attn.permute(0, 2, 1) # transpose for matrix multiplication
        
        a = torch.bmm(v, attn)
        a = a.reshape(b, c, h, w)
        
        return x + a # residual connection for baseline performance guarantee