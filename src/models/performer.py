from models.base import TorchModel
import performer_pytorch
from performer_pytorch import FastAttention


class PerformerAttention(TorchModel):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.attention = FastAttention(hidden_dim // num_heads)

    def forward(self, k, q, v, attn_mask=None, key_padding_mask=None, need_weights=False):
        bs, length, dim = k.shape
        head_dim = dim // self.num_heads

        k = k.reshape(bs, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(bs, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(bs, length, self.num_heads, head_dim).permute(0, 2, 1, 3)

        v = self.attention(k, q, v)
        v = v.permute(0, 2, 1, 3).reshape(bs, length, dim)
        
        return v


class Performer(TorchModel, config_name='performer'):
    def __init__(self, **config):
        super().__init__()
        self._performer = performer_pytorch.Performer(**config)

    def forward(self, x):
        x = self._performer(x)
        return x
