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

from proformer_utils.custom_classes import PositionalEncoding, CustomTransformerEncoderLayer, CustomTransformerEncoder


# This code is a modification of the original codebase found at
# https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py

def create_special_attention_mask(src, vocab, device):
    """
    Crea una maschera di attenzione specializzata che permette ai token di prestare
    attenzione in modo selettivo ad altri token in base alla loro importanza semantica.

    Args:
        src (torch.Tensor): Tensore di input contenente gli indici dei token [batch_size, seq_len]
        vocab (torchtext.vocab.Vocab): Vocabolario del modello
        device (str/torch.device): Dispositivo su cui allocare la maschera

    Returns:
        torch.Tensor: Maschera di attenzione di forma [batch_size, seq_len, seq_len]
    """
    # Determina la dimensione del batch e la lunghezza della sequenza
    batch_size, seq_len = src.shape

    # Crea una maschera di base (tutti possono prestare attenzione a tutti)
    base_mask = torch.ones(batch_size, seq_len, seq_len, device=device)

    # Ottieni gli indici dei token speciali
    pad_idx = vocab["<pad>"] if "<pad>" in vocab.get_stoi() else -1
    mask_idx = vocab["<mask>"] if "<mask>" in vocab.get_stoi() else -1
    cls0_idx = vocab["<cls0>"] if "<cls0>" in vocab.get_stoi() else -1
    cls1_idx = vocab["<cls1>"] if "<cls1>" in vocab.get_stoi() else -1

    # Identifica le posizioni dei token speciali
    pad_positions = (src == pad_idx)
    cls0_positions = (src == cls0_idx)
    cls1_positions = (src == cls1_idx)
    mask_positions = (src == mask_idx)
    class_positions = cls0_positions | cls1_positions | mask_positions

    # Amplifica l'attenzione verso i token di classe (fattore 2.0)
    for i in range(batch_size):
        # I token di padding non ricevono né forniscono attenzione
        for j in range(seq_len):
            if pad_positions[i, j]:
                base_mask[i, j, :] = 0  # Il padding non presta attenzione a nulla
                base_mask[i, :, j] = 0  # Nessuno presta attenzione al padding

            # I token di classe ricevono maggiore attenzione
            if class_positions[i, j]:
                # Aumenta l'attenzione dai token normali ai token di classe
                for k in range(seq_len):
                    if not (pad_positions[i, k] or class_positions[i, k]):
                        base_mask[i, k, j] = 2.0  # Normale -> Classe: attenzione amplificata

                # Aumenta l'attenzione tra token di classe
                for k in range(seq_len):
                    if class_positions[i, k] and k != j:
                        base_mask[i, j, k] = 1.5  # Classe -> Classe: attenzione moderatamente amplificata

    return base_mask

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
        self.mask_positions = None
        self.cls_logits = None
        self.last_hidden_states = None
        self.vocab = vocab
        self.ntokens = ntoken
        self.opt = opt
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(opt["d_model"], dropout=opt["pos_enc_dropout"], max_len=opt["bptt"])

        # Sostituiamo il TransformerEncoder standard con la nostra versione personalizzata
        encoder_layer = CustomTransformerEncoderLayer(
            opt["d_model"], opt["nhead"], opt["d_hid"], opt["dropout"],
            activation="gelu", norm_first=True
        )
        self.transformer_encoder = CustomTransformerEncoder(encoder_layer, opt["nlayers"])

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

    def init_weights(self):
        # Use He initialization
        nn.init.kaiming_uniform_(self.embedding.weight.data, nonlinearity='relu')
        if self.opt["use_taxonomy"]:
            nn.init.kaiming_uniform_(self.tax_encoder.weight.data, nonlinearity='relu')

        # Initialize linear layer
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

    def get_attention_maps(self):
        """
        Restituisce le mappe di attenzione di tutti i livelli del transformer

        Returns:
            list: Lista di tensori di attenzione, uno per ogni livello
        """
        return self.transformer_encoder.attention_maps

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
        # Applica la maschera di attenzione speciale se richiesta
        if self.opt.get("use_special_attention", True):
            src_mask = create_special_attention_mask(src, self.vocab, src.device)

        # Continua con il resto della forward pass esistente
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)

        if self.opt.get("use_taxonomy", False) and hasattr(self, 'taxonomy'):
            # Codice esistente per la tassonomia...
            pass

        output = self.transformer_encoder(src, mask=src_mask)
        self.last_hidden_states = output

        # Identifica le posizioni dei token <mask> per la classificazione
        self.mask_positions = (src == self.vocab["<mask>"]).bool()
        mask_hidden = output[self.mask_positions]

        if mask_hidden.size(0) > 0:
            self.cls_logits = self.classifier(mask_hidden).squeeze(-1)
        else:
            self.cls_logits = torch.tensor([], device=src.device)

        output = self.norm(output)
        output_logits = self.linear(output)

        return output_logits

    # def forward(self, src, src_mask=None):
    #     """
    #     Forward pass of the model.
    #     """
    #
    #     # Calculate token embeddings and scale
    #     x = self.embedding(src) * math.sqrt(self.d_model)
    #
    #     # Add taxonomy embeddings if enabled
    #     if self.opt["use_taxonomy"] and hasattr(self, 'taxonomy'):
    #         tax_emb = self.taxonomy[src]
    #         tax_pe = self.tax_encoder(F.normalize(tax_emb, p=2, dim=-1))
    #         x = x + F.dropout(tax_pe, 0.01)
    #
    #     # Apply positional encoding if enabled
    #     if self.opt["use_pe"]:
    #         x = self.pos_encoder(x)
    #
    #     # Pass through transformer encoder
    #     output = self.transformer_encoder(x, src_mask, is_causal=False)
    #
    #     # IMPORTANTE: Salva l'output del transformer come hidden states
    #     self.last_hidden_states = output.clone()
    #
    #     # Normalizza e applica la proiezione lineare
    #     output = self.norm(output)
    #     output_logits = self.linear(output)
    #
    #     # Trova le posizioni dei token <mask>
    #     mask_positions = (src == self.vocab["<mask>"])
    #     self.mask_positions = mask_positions
    #     # Estrai gli hidden states solo per le posizioni mascherate
    #     if torch.any(mask_positions):
    #         masked_embeddings = self.last_hidden_states[mask_positions]
    #         cls_logits = self.classifier(masked_embeddings).squeeze()
    #         # Gestisce il caso di un singolo elemento
    #         if cls_logits.dim() == 0:
    #             self.cls_logits = cls_logits.unsqueeze(0)
    #         else:
    #             self.cls_logits = cls_logits
    #     else:
    #         # Crea un tensore vuoto ma con dimensioni corrette
    #         self.cls_logits = torch.zeros((0,), device=src.device)
    #
    #     return output_logits