import math
import torch
import torch.nn.functional as F
from torch import nn


class CustomMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.last_attn_weights = None

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None,
                need_weights=True, average_attn_weights=True):
        output, attn_weights = self.mha(
            query, key, value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=average_attn_weights
        )
        self.last_attn_weights = attn_weights
        return output, attn_weights


class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", layer_norm_eps=1e-5, norm_first=False):
        super().__init__()
        self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = getattr(F, activation)
        self.norm_first = norm_first

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = self.norm1(x)
            x2, weights = self.self_attn(x, x, x, attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask,
                                         need_weights=True)
            x = x + self.dropout1(x2)
            x = x + self._ff_block(self.norm2(x))
        else:
            x2, weights = self.self_attn(x, x, x, attn_mask=src_mask,
                                         key_padding_mask=src_key_padding_mask,
                                         need_weights=True)
            x = self.norm1(x + self.dropout1(x2))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer] + [
            CustomTransformerEncoderLayer(
                encoder_layer.self_attn.mha.embed_dim,
                encoder_layer.self_attn.mha.num_heads,
                dim_feedforward=encoder_layer.linear1.out_features,
                dropout=encoder_layer.dropout.p,
                activation='gelu' if isinstance(encoder_layer.activation, nn.GELU) else 'relu',
                norm_first=encoder_layer.norm_first
            ) for _ in range(num_layers - 1)
        ])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=False):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

    @property
    def attention_maps(self):
        return [layer.self_attn.last_attn_weights for layer in self.layers]


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)