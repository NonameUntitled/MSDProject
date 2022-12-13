from .base import TorchModel
from ..infinite_former import long_term_attention


class InfinityFormerAttention(TorchModel):
    def __init__(self, **config):
        super().__init__()
        hidden_dim = config['head_size']
        num_heads = config['n_heads']

        assert hidden_dim % num_heads == 0

        self.num_heads = num_heads
        self.attention = long_term_attention.LongTermAttention(**config)

    def forward(self, k, q, v, attn_mask=None, key_padding_mask=None, need_weights=False):
        k = k.permute(1, 0, 2)
        q = q.permute(1, 0, 2)

        v = self.attention(k, q)[0].permute(1, 0, 2)
        
        return v


class Informer(TorchModel, config_name='informer'):
    pass
