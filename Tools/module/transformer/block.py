import torch
import torch.nn as nn
import math
import copy
from Tools.module.attention import ConvMultiHeadAttention, MultiHeadAttention

def get_src_mask(mask):
    """生成transformer的encoder mask

    Args:
        mask (_type_): [bs, len]

    Returns:
        _type_: _description_
    """
    mask = mask.float()
    return mask.unsqueeze(1).unsqueeze(2)
    pass

def get_trg_mask(mask):
    """生成transformer的decoder mask

    Args:
        mask (_type_): [bs, len]

    Returns:
        _type_: _description_
    """
    trg_pad_mask = mask.unsqueeze(1).unsqueeze(2)
    trg_len = mask.size(1)
    trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(trg_pad_mask.device)
    trg_mask = trg_pad_mask & trg_sub_mask
    return trg_mask.float()
    pass

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu1 = nn.GELU()
        self.relu2 = nn.GELU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu2(x)
        return x
    
class Encoder(nn.Module):
    """transformer encoder
    """
    def __init__(self, 
                d_model, 
                num_layers,
                drop_prob,
                ffn_hidden=512,
                n_head=8, 
                build_attention = lambda d_model, n_head, *args : MultiHeadAttention(d_model, n_head),
                build_ffn = lambda d_model, hidden, drop_prob, *args : PositionwiseFeedForward(d_model, hidden, drop_prob),
                *args
                ):
        """transformer encoder初始化

        Args:
            d_model (_type_): 序列中每个元素的维度
            ffn_hidden (_type_): ffn中隐藏层的元素个数
            n_head (_type_): 多注意头的个数
            num_layers (_type_): 共多少层
            drop_prob (_type_): dropout的概率
            build_attention (_type_, optional): 构建attention
            build_ffn (_type_, optional): 构建ffn
        """
        super(Encoder, self).__init__()
        self.model =nn.ModuleList([
            EncoderLayer(
                d_model,
                drop_prob,
                build_attention(d_model, n_head, args),
                build_ffn(d_model, ffn_hidden, drop_prob, args)
            ) for _ in range(num_layers)
        ])
        pass
    def forward(self, x, src_mask):
        for layer in self.model:
            x = layer(x, src_mask)
            pass
        return x
        pass
    pass
class EncoderLayer(nn.Module):

    def __init__(self, 
                 d_model, 
                 drop_prob,
                 attention,
                 ffn):
        super(EncoderLayer, self).__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = ffn
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask=None):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)
             
        # 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)
        return x


class Decoder(nn.Module):
    """transformer encoder
    """
    def __init__(self, 
                d_model, 
                num_layers,
                drop_prob,
                ffn_hidden=512,
                n_head=8, 
                build_attention = lambda d_model, n_head, *args : MultiHeadAttention(d_model, n_head),
                build_ffn = lambda d_model, hidden, drop_prob, *args : PositionwiseFeedForward(d_model, hidden, drop_prob),
                *args
                ):
        """transformer decoder初始化

        Args:
            d_model (_type_): 序列中每个元素的维度
            ffn_hidden (_type_): ffn中隐藏层的元素个数
            n_head (_type_): 多注意头的个数
            num_layers (_type_): 共多少层
            drop_prob (_type_): dropout的概率
            build_attention (_type_, optional): 构建attention
            build_ffn (_type_, optional): 构建ffn
        """
        super(Decoder, self).__init__()
        self.model =nn.ModuleList([
            DecoderLayer(
                d_model,
                drop_prob,
                build_attention(d_model, n_head, args),
                build_attention(d_model, n_head, args),
                build_ffn(d_model, ffn_hidden, drop_prob, args)
            ) for _ in range(num_layers)
        ])
        pass
    def forward(self, dec, enc, trg_mask, src_mask):
        """前向传播

        Args:
            dec (_type_): 目标
            enc (_type_): 编码器编码的特征
            trg_mask (_type_): 目标的mask
            src_mask (_type_): 编码器输入的mask
        """
        for layer in self.model:
            dec = layer(dec, enc, trg_mask, src_mask)
            pass
        return dec
        pass
    pass

class DecoderLayer(nn.Module):

    def __init__(self, 
                 d_model, 
                 drop_prob,
                 attention,
                 dec_attention,
                 ffn
                 ):
        super(DecoderLayer, self).__init__()
        self.self_attention = dec_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = attention
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = ffn
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        
        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc is not None:
            # 3. compute encoder - decoder attention
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=src_mask)
            
            # 4. add and norm
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        # 5. positionwise feed forward network
        _x = x
        x = self.ffn(x)
        
        # 6. add and norm
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        return x
  