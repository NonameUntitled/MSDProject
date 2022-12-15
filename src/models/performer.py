from models.base import TorchModel
from models.baseline import BaselineProjector, BaselineEncoder

import torch.nn as nn

from performer_pytorch import FastAttention


class PerformerAttention(nn.MultiheadAttention):

    def __init__(
            self,
            embed_dim,
            num_heads,
            nb_features,
            dropout=0.,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            device=device,
            dtype=dtype
        )
        assert embed_dim % num_heads == 0
        self.attention = FastAttention(
            dim_heads=embed_dim // num_heads,
            nb_features=nb_features
        )

    def forward(
            self, query, key, value, key_padding_mask=None,
            need_weights=True, attn_mask=None,
            average_attn_weights=True
    ):
        bs, length, dim = key.shape
        head_dim = dim // self.num_heads

        k = key.reshape(bs, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        q = query.reshape(bs, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = value.reshape(bs, length, self.num_heads, head_dim).permute(0, 2, 1, 3)

        v = self.attention(k, q, v)
        v = v.permute(0, 2, 1, 3).reshape(bs, length, dim)

        return v


class PerformerAttentionLayer(nn.TransformerEncoderLayer):

    def __init__(
            self,
            d_model,
            nhead,
            nb_features,
            dim_feedforward=None,
            dropout=0.0,
            activation=nn.ReLU,
            layer_norm_eps=1e-5,
            batch_first=False,
            norm_first=False,
            device=None,
            dtype=None,
            **kwargs
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward or 4 * d_model,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype
        )
        assert d_model % nhead == 0

        self.self_attn = PerformerAttention(embed_dim=d_model, num_heads=nhead, nb_features=nb_features)


class Performer(TorchModel, config_name='performer'):

    def __init__(self, projector, encoder):
        super().__init__()
        self._projector = projector
        self._encoder = encoder

    @classmethod
    def create_from_config(cls, config, **kwargs):
        projector = BaselineProjector.create_from_config(config['projector'], **kwargs)
        encoder = BaselineEncoder.create_from_config(
            config['encoder'],
            input_dim=projector.output_dim,
            attention_layer_cls=PerformerAttentionLayer,
            **kwargs
        )
        return cls(projector, encoder)

    def encoder_only(self, embeddings):
        mask = embeddings.new_ones(embeddings.shape[:-1]).bool()
        embeddings, mask = self._encoder(embeddings, mask)
        return embeddings

    def forward(self, inputs):
        embeddings, mask = self._projector(inputs)
        embeddings, mask = self._encoder(embeddings, mask)
        return embeddings
