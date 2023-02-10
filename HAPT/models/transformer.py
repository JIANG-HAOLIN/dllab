import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange






# classes
class seq2token(nn.Module):
    def __init__(self, patch_length, patch_dim, token_dim):
        super(seq2token, self).__init__()
        self.patch_length = patch_length
        self.patch_dim = patch_dim
        self.token_dim = token_dim

        self.mapping = nn.Linear(patch_dim, token_dim)
    def forward(self, x):
        x = rearrange(x,'batch channel (num_patch patch_length) -> batch num_patch (patch_length channel)',patch_length=self.patch_length) # (B, C通道数 * p * p, L一张图片几个patch)
        x = self.mapping(x)
        return x


class input_preprocess(nn.Module):
    def __init__(self, patch_length, patch_dim, token_dim):
        super(input_preprocess, self).__init__()
        self.seq2token = seq2token(patch_length=patch_length, patch_dim=patch_dim, token_dim=token_dim)
    def forward(self, x):
        x = self.seq2token(x)

        token_dim = x.shape[-1]
        num_patches = x.shape[-2]

        omega = torch.arange(token_dim // 2, device=x.device) / (token_dim // 2 - 1)
        omega = 1. / (10**4 ** omega).view(1,-1)

        n = torch.arange(num_patches, device=x.device).view(-1,1)*omega
        n_sine = n.sin()
        n_cosine = n.cos()

        x = x.view(x.shape[0],-1,x.shape[-1]) + torch.cat((n_sine, n_cosine), dim=1).type(x.dtype)
        return x

class ff(nn.Module):
    def __init__(self, token_dim, hidden_size):
        super().__init__()
        self.ff = nn.ModuleList()
        self.layernorm = nn.LayerNorm(token_dim)
        self.mapping = nn.Linear(token_dim,hidden_size)
        self.act = nn.GELU()
        self.back_mapping = nn.Linear(hidden_size, token_dim)
        self.ff.append(self.layernorm)
        self.ff.append(self.mapping)
        self.ff.append(self.act)
        self.ff.append(self.back_mapping)

    def forward(self, x):
        for layer in self.ff:
            x = layer(x)
        return x

class Attention(nn.Module):
    def __init__(self, token_dim, num_heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * num_heads
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.coefficient = torch.rsqrt(torch.Tensor([dim_head]))
        self.layernorm = nn.LayerNorm(token_dim)
        self.WQ = nn.Linear(token_dim, inner_dim, bias = False)
        self.WK = nn.Linear(token_dim, inner_dim, bias = False)
        self.WV = nn.Linear(token_dim, inner_dim, bias = False)
        self.mapping = nn.Linear(inner_dim, token_dim, bias = False)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.layernorm(x)
        Q = self.WQ(x).view(batch_size,self.num_heads,-1,self.dim_head)
        K = self.WK(x).view(batch_size,self.num_heads,-1,self.dim_head)
        V = self.WV(x).view(batch_size,self.num_heads,-1,self.dim_head)
        QK = torch.einsum('bhni,bhim->bhnm',Q,K.permute(0,1,-1,-2))* self.coefficient
        attention = F.softmax(QK,dim=-1)
        out = torch.einsum('bhni,bhim->bhnm',attention,V).view(batch_size,-1,self.inner_dim)
        out = self.mapping(out)
        return out



class Transformer(nn.Module):
    def __init__(self, token_dim, num_blocks, num_heads, dim_head, hidden_size):
        super().__init__()
        self.attention_blocks = nn.ModuleList([])
        for block in range(num_blocks):
            self.attention_blocks.append(nn.ModuleList([
                Attention(token_dim=token_dim, num_heads = num_heads, dim_head = dim_head),
                ff(token_dim=token_dim, hidden_size=hidden_size)
            ]))
    def forward(self, x):
        for attention, feedfoward in self.attention_blocks:
            x = attention(x) + x
            x = feedfoward(x) + x
        x = torch.mean(x, dim=1)
        return x

class Encoder(nn.Module):
    def __init__(self, *, sequence_length=250, patch_length=10, num_classes=12, token_dim=512, num_blocks=6, num_heads=8, hidden_size=1024, channels = 6, dim_head = 64):
        super().__init__()
        patch_dim = channels * patch_length
        self.input_preprocess = input_preprocess(patch_length=patch_length, patch_dim=patch_dim, token_dim=token_dim)
        self.transformer = Transformer(token_dim=token_dim,
                                       num_blocks=num_blocks,
                                       num_heads=num_heads,
                                       dim_head=dim_head,
                                       hidden_size=hidden_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, num_classes),
            # nn.Softmax(dim=-1)
        )


    def forward(self, x):
        x = self.input_preprocess(x)
        x = self.transformer(x)
        x = self.classifier(x)
        return x



# if __name__ == '__main__':
#
#     v = Encoder(
#         sequence_length = 250,
#         patch_length = 10,
#         num_classes = 12,
#         token_dim = 512,
#         num_blocks = 6,
#         num_heads = 8,
#         hidden_size = 1024,
#         channels = 6,
#         dim_head = 64
#     )
#
#     time_series = torch.randn(32, 6, 250)
#     logits = v(time_series) # (4, 1000)
#     print(logits.shape,torch.sum(logits[0]))