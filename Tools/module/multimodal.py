import torch
import torch.nn as nn

from Tools.module.attention import ConvMultiHeadAttention, MultiHeadAttention
from Tools.module.common import MLP
from Tools.util.utils import cal_params

"""
用于多模态任务的模块
包括但不限于多模态融合，多模态特征对齐等
"""




class MultimodalFusionAttention(nn.Module):
    """多模态特征融合
    
    使用attention的方式将一个模态的特征注入到另一个模态

    """
    def __init__(self, d_model, n_head, dropout=0.1, mode='linear', num_fusion_layer=4):
        super(MultimodalFusionAttention, self).__init__()
        if mode == 'linear':
            self.att1 = MultiHeadAttention(d_model, n_head)
        elif mode == 'conv':
            self.att1 = ConvMultiHeadAttention(d_model, n_head)
        else:
            raise "未知attention类型"
        self.fusion = MLP(d_model * 2, d_model, num_hidden_layers=num_fusion_layer)
        self.dropout = nn.Dropout(dropout)
        pass
    
    def forward(self, x, y):
        """将模态y注入到x

        Args:
            x : 主模态[batch, seq1, dim]
            y : 补充模态[batch, seq2, dim]
        """

        y_att = self.att1(x, y, y, None)
        x = self.fusion(torch.cat([x, y_att], dim=-1))  
        x = self.dropout(x)
        return x
        pass
    pass


class MultimodalAlignFusionBlock(nn.Module):
    """多模态特征融合，先对齐再融合
    
    使用attention的方式将一个模态的特征注入到另一个模态

    """
    def __init__(self, d_model, n_head, dropout=0.02, mode='linear', num_fusion_layer=4, norm='ln', act='lrelu'):
        super(MultimodalAlignFusionBlock, self).__init__()
        if mode == 'linear':
            self.att1 = MultiHeadAttention(d_model, n_head)
            self.att2 = MultiHeadAttention(d_model, n_head)
        elif mode == 'conv':
            self.att1 = ConvMultiHeadAttention(d_model, n_head)
            self.att2 = ConvMultiHeadAttention(d_model, n_head)
        else:
            raise "未知attention类型"
        self.align = MLP(d_model, d_model, num_hidden_layers=num_fusion_layer, norm=norm, act=act)
        self.align2 = MLP(d_model, d_model, num_hidden_layers=num_fusion_layer, norm=norm, act=act)        
        self.dropout = nn.Dropout(dropout)
        pass
    
    def forward(self, x, y):
        """将模态y注入到x

        Args:
            x : 主模态[batch, seq1, dim]
            y : 补充模态[batch, seq2, dim]
        """

        n_y = self.att1(x, y, y, None)
        n_y = self.align(n_y)  
        n_x = self.att2(x + n_y, x, x, None)
        n_x = self.align(n_x)
        out = x + n_x + n_y
        out = self.dropout(out)
        return out, n_y
        pass
    pass


class MultimodalAlignFusion(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_head, 
                 dropout=0.02, 
                 mode='linear', 
                 num_fusion_layer = 4, 
                 num_block = 2,
                 norm='ln',
                 act="lrelu"
                 ):
        super(MultimodalAlignFusion, self).__init__()
        self.model = nn.ModuleList([
            MultimodalAlignFusionBlock(d_model, n_head, dropout, mode, num_fusion_layer, norm, act)
            for _ in range(num_block)
        ])
        pass
    
    def forward(self, x, y):
        for layer in self.model:
            x, y = layer(x, y)
            pass
        return x
        pass
    pass
if __name__ == "__main__":
    x = torch.randn(1, 4, 256)
    y = torch.randn(1, 8, 256)
    model = MultimodalAlignFusion(256, 4)
    x = model(x, y)
    print(x.size())
    print("{:.2f}M".format(cal_params(model)))
    pass