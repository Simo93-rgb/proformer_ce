import time
import math
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import dataset
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tempfile import TemporaryDirectory
from typing import Tuple

# This code is a modification of the original codebase found at 
# https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py

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


class TransformerModel(nn.Module):

    def __init__(self, ntoken, opt, taxonomy=None):
        super().__init__()
        self.ntokens = ntoken
        self.opt = opt
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(opt["d_model"], dropout=opt["pos_enc_dropout"], max_len=opt["bptt"])
        encoder_layers = TransformerEncoderLayer(opt["d_model"], opt["nhead"], opt["d_hid"], opt["dropout"], activation="relu", norm_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, opt["nlayers"], enable_nested_tensor=True)
        self.embedding = nn.Embedding(ntoken, opt["d_model"])
        self.d_model = opt["d_model"]
        
        self.pos_emb = nn.Embedding(opt["bptt"], opt["d_model"])

        if opt["use_taxonomy"]:
            self.taxonomy = taxonomy
            self.tax_encoder = nn.Linear(taxonomy.size(1), opt["d_model"], bias=False)
            # self.tax_encoder = nn.Linear(opt["d_model"]+taxonomy.size(1), opt["d_model"])

        self.linear = nn.Sequential(nn.Linear(opt["d_model"], ntoken))
        self.init_weights()

    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        if self.opt["use_taxonomy"]:
            self.tax_encoder.weight.data.uniform_(-initrange, initrange)
        for l in self.linear:
            if isinstance(l, nn.Linear):
                l.bias.data.zero_()
                l.weight.data.uniform_(-initrange, initrange)
    

    def create_masked_attention_matrix(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    
    def forward(self, src, src_mask=None):
        """
        args:
            src: [seq_len, batch_size]
            src_mask: [seq_len, seq_len]

        returns:
            [seq_len, batch_size, ntoken]
        """
        # pos = torch.stack([torch.arange(src.size(0)) for i in range(src.size(1))]).T.to(self.opt["device"])
        # p_emb  = self.pos_emb(pos)
    
        if self.opt["use_taxonomy"]:
            tax_pe = self.tax_encoder(F.normalize(self.taxonomy[src], 2, dim=0))
            # tax_pe = F.normalize(self.taxonomy[src], 2, dim=0)
        
        src = self.embedding(src) * math.sqrt(self.d_model)

        if self.opt["use_taxonomy"]:
            src = (src + F.dropout(tax_pe, 0.01))
            # src = self.tax_encoder(torch.cat([src, tax_pe], dim=2))
        
        if self.opt["use_pe"]:
            src = self.pos_encoder(src)
            # src = src + F.dropout(p_emb, 0.1)

        output = self.transformer_encoder(src, src_mask, is_causal=False)
        self.last_hidden_states = output.clone()
        output = self.linear(output)

        return output