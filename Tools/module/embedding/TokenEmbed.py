import torch
import torch.nn as nn
class TokenEmbed(nn.Module):
    def __init__(self, num_embed, embed_dim):
        super(TokenEmbed, self).__init__()
        self.embed = nn.Embedding(num_embed, embed_dim)
        pass
    def forward(self, x):
        return self.embed(x)
    pass


if __name__ == "__main__":
    model = TokenEmbed(2, 256)
    x = torch.Tensor([
        [0, 1],
        [1, 1],
        [0, 0]
    ]).int()
    x = model(x)
    print(x)
    pass