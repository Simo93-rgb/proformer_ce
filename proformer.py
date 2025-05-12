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

import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    FeedForward is a feed-forward neural network layer used in the Transformer model.

    Attributes:
        linear1 (nn.Linear): First linear transformation layer.
        dropout (nn.Dropout): Dropout layer after the first linear transformation.
        linear2 (nn.Linear): Second linear transformation layer.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initializes the FeedForward layer.

        Args:
            d_model (int): Dimension of the input.
            d_ff (int): Dimension of the feed-forward network.
            dropout (float): Dropout rate.
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Forward pass of the FeedForward layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the feed-forward network.
        """
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))

class TransformerModel(nn.Module):
    """
    TransformerModel is a neural network model based on the Transformer architecture.

    Attributes:
        last_hidden_states (Tensor): Stores the last hidden states of the model.
        ntokens (int): Number of tokens in the vocabulary.
        opt (dict): Dictionary containing model hyperparameters.
        model_type (str): Type of the model, set to 'Transformer'.
        pos_encoder (PositionalEncoding): Positional encoding layer.
        transformer_encoder (TransformerEncoder): Transformer encoder layer.
        embedding (nn.Embedding): Embedding layer for input tokens.
        d_model (int): Dimension of the model.
        pos_emb (nn.Embedding): Positional embedding layer.
        taxonomy (Tensor, optional): Taxonomy tensor if use_taxonomy is enabled.
        tax_encoder (nn.Linear, optional): Linear layer for taxonomy encoding.
        linear (nn.Sequential): Linear layer for output.
    """

    def __init__(self, vocab, ntoken, opt, taxonomy=None):
        """
        Initializes the TransformerModel.

        Args:
            ntoken (int): Number of tokens in the vocabulary.
            opt (dict): Dictionary containing model hyperparameters.
            taxonomy (Tensor, optional): Taxonomy tensor if use_taxonomy is enabled.
        """
        super().__init__()
        self.cls_logits = None
        self.last_hidden_states = None
        self.vocab = vocab
        self.ntokens = ntoken
        self.opt = opt
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(opt["d_model"], dropout=opt["pos_enc_dropout"], max_len=opt["bptt"])
        encoder_layers = TransformerEncoderLayer(opt["d_model"], opt["nhead"], opt["d_hid"], opt["dropout"],
                                                 activation="gelu", norm_first=True)  # Changed to gelu
        self.transformer_encoder = TransformerEncoder(encoder_layers, opt["nlayers"], enable_nested_tensor=True)
        self.embedding = nn.Embedding(ntoken, opt["d_model"])
        self.d_model = opt["d_model"]

        self.pos_emb = nn.Embedding(opt["bptt"], opt["d_model"])

        if opt["use_taxonomy"] and taxonomy is not None:
            self.taxonomy = taxonomy
            self.tax_encoder = nn.Linear(taxonomy.size(1), opt["d_model"], bias=False)

        self.norm = nn.LayerNorm(opt["d_model"])  # Added layer norm before final projection
        self.linear = nn.Linear(opt["d_model"], ntoken)  # Simplified to single Linear layer
        self.init_weights()
        self.classifier = nn.Sequential(
            nn.Linear(opt["d_model"], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )


    # def init_weights(self):
    #     """
    #     Initializes the weights of the model.
    #     """
    #     initrange = 0.1
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     if self.opt["use_taxonomy"]:
    #         self.tax_encoder.weight.data.uniform_(-initrange, initrange)
    #     for l in self.linear:
    #         if isinstance(l, nn.Linear):
    #             l.bias.data.zero_()
    #             l.weight.data.uniform_(-initrange, initrange)

    def init_weights(self):
        # Use He initialization
        nn.init.kaiming_uniform_(self.embedding.weight.data, nonlinearity='relu')
        if self.opt["use_taxonomy"]:
            nn.init.kaiming_uniform_(self.tax_encoder.weight.data, nonlinearity='relu')

        # Initialize linear layer
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def create_masked_attention_matrix(self, sz):
        """
        Creates a masked attention matrix.

        Args:
            sz (int): Size of the attention matrix.

        Returns:
            Tensor: Masked attention matrix.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        """
        Forward pass of the model.
        """
        # Calculate token embeddings and scale
        x = self.embedding(src) * math.sqrt(self.d_model)

        # Add taxonomy embeddings if enabled
        if self.opt["use_taxonomy"] and hasattr(self, 'taxonomy'):
            tax_emb = self.taxonomy[src]
            tax_pe = self.tax_encoder(F.normalize(tax_emb, p=2, dim=-1))
            x = x + F.dropout(tax_pe, 0.01)

        # Apply positional encoding if enabled
        if self.opt["use_pe"]:
            x = self.pos_encoder(x)

        # Pass through transformer encoder
        output = self.transformer_encoder(x, src_mask, is_causal=False)

        # IMPORTANTE: Salva l'output del transformer come hidden states
        self.last_hidden_states = output.clone()

        # Normalizza e applica la proiezione lineare
        output = self.norm(output)
        output_logits = self.linear(output)

        # Trova le posizioni dei token <mask>
        mask_positions = (src == self.vocab["<mask>"])

        # Estrai gli hidden states solo per le posizioni mascherate
        if torch.any(mask_positions):
            masked_embeddings = self.last_hidden_states[mask_positions]
            cls_logits = self.classifier(masked_embeddings).squeeze()
            # Gestisce il caso di un singolo elemento
            if cls_logits.dim() == 0:
                self.cls_logits = cls_logits.unsqueeze(0)
            else:
                self.cls_logits = cls_logits
        else:
            # Crea un tensore vuoto ma con dimensioni corrette
            self.cls_logits = torch.zeros((0,), device=src.device)

        return output_logits



        '''OLD CODE: forward() method'''

        # def forward(self, src, src_mask=None):
        #     """
        #     Forward pass of the model.
        #
        #     Args:
        #         src (Tensor): Input tensor.
        #         src_mask (Tensor, optional): Source mask tensor.
        #
        #     Returns:
        #         Tensor: Output tensor.
        #     """
        #
        #     # Calculate token embeddings and scale
        # src = self.embedding(src) * math.sqrt(self.d_model)
        # if self.opt["use_taxonomy"]:
        #     src = src.long()
        #     tax_pe = self.tax_encoder(F.normalize(self.taxonomy[src], 2, dim=0))
        #     src = (src + F.dropout(tax_pe, 0.01))
        #
        # if self.opt["use_pe"]:
        #     src = self.pos_encoder(src)
        #
        # output = self.transformer_encoder(src, src_mask, is_causal=False)
        # self.last_hidden_states = output.clone()
        #
        # # Apply layer norm before final projection
        # output = self.norm(output)
        # output = self.linear(output)

        # return output