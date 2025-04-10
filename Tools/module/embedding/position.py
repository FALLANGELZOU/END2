import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """最普通的位置编码
    """
    def __init__(self, d_model, max_len=2000, batch_first=True):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term) # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # 1,51,512    -->  [51, 1, d_model]
        self.register_buffer('pe', pe)
        self.batch_first = batch_first
        pass
    
    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        if self.batch_first:
            x = x.permute(1, 0, 2)
            pass
        pos = self.pe[:x.size(0), :]  # [x_len, batch_size, d_model]
        if self.batch_first:
            pos = pos.permute(1, 0, 2)
        return pos