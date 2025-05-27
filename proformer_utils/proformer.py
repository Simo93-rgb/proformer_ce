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
from typing import Tuple, List, Dict, Any, Optional, Union

from proformer_utils.custom_classes import PositionalEncoding, CustomTransformerEncoderLayer, CustomTransformerEncoder


def check_logits(output_logits: torch.Tensor) -> None:
    """
    Check logits values for debugging purposes.

    Args:
        output_logits: The output logits tensor to check
    """
    try:
        # Check for NaN or infinite values
        with torch.no_grad():
            has_nan = torch.isnan(output_logits).any()
            has_inf = torch.isinf(output_logits).any()

            # Print statistics if values are valid
            if not has_nan and not has_inf:
                logits_mean = output_logits.mean().item()
                logits_std = output_logits.std().item()
                logits_min = output_logits.min().item()
                logits_max = output_logits.max().item()
                print(
                    f"Logits stats: mean={logits_mean:.2f}, std={logits_std:.2f}, min={logits_min:.2f}, max={logits_max:.2f}")
            else:
                print(f"WARNING! Found NaN={has_nan} or Inf={has_inf} values in logits")
    except Exception as e:
        print(f"Error checking logits: {e}")


class TransformerModel(nn.Module):
    """
    TransformerModel is a neural network model based on the Transformer architecture.

    Attributes:
        last_hidden_states (Tensor): Stores the last hidden states of the model
        ntokens (int): Number of tokens in the vocabulary
        opt (dict): Dictionary containing model hyperparameters
        model_type (str): Type of the model, set to 'Transformer'
        pos_encoder (PositionalEncoding): Positional encoding layer
        transformer_encoder (TransformerEncoder): Transformer encoder layer
        embedding (nn.Embedding): Embedding layer for input tokens
        d_model (int): Dimension of the model
        pos_emb (nn.Embedding): Positional embedding layer
        taxonomy (Tensor, optional): Taxonomy tensor if use_taxonomy is enabled
        tax_encoder (nn.Linear, optional): Linear layer for taxonomy encoding
        norm (nn.LayerNorm): Layer normalization before final projection
        linear (nn.Linear): Linear projection layer for output
        classifier (nn.Sequential): Sequential layers for classification
    """

    def __init__(self, vocab: Any, ntoken: int, opt: Dict[str, Any], taxonomy: Optional[torch.Tensor] = None):
        """
        Initialize the TransformerModel.

        Args:
            vocab: Vocabulary object
            ntoken: Number of tokens in the vocabulary
            opt: Dictionary containing model hyperparameters
            taxonomy: Optional taxonomy tensor if use_taxonomy is enabled
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

        # Replace standard TransformerEncoder with custom version
        try:
            encoder_layer = CustomTransformerEncoderLayer(
                opt["d_model"], opt["nhead"], opt["d_hid"], opt["dropout"],
                activation="gelu", norm_first=True
            )
            self.transformer_encoder = CustomTransformerEncoder(encoder_layer, opt["nlayers"])
        except Exception as e:
            print(f"Error initializing transformer encoder: {e}")
            # Fallback to standard transformer encoder if custom fails
            encoder_layer = TransformerEncoderLayer(
                opt["d_model"], opt["nhead"], opt["d_hid"], opt["dropout"],
                activation="gelu"
            )
            self.transformer_encoder = TransformerEncoder(encoder_layer, opt["nlayers"])

        self.embedding = nn.Embedding(ntoken, opt["d_model"])
        self.d_model = opt["d_model"]

        self.pos_emb = nn.Embedding(opt["bptt"], opt["d_model"])

        if opt["use_taxonomy"] and taxonomy is not None:
            self.taxonomy = taxonomy
            self.tax_encoder = nn.Linear(taxonomy.size(1), opt["d_model"], bias=False)

        self.norm = nn.LayerNorm(opt["d_model"])
        self.linear = nn.Linear(opt["d_model"], ntoken)

        # Initialize classifier for masked token prediction
        self.classifier = nn.Sequential(
            nn.Linear(opt["d_model"], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.init_weights()

    def init_weights(self) -> None:
        """
        Initialize model weights using He initialization.
        """
        try:
            # Use He initialization for embedding
            nn.init.kaiming_uniform_(self.embedding.weight.data, nonlinearity='relu')

            # Initialize taxonomy encoder if used
            if self.opt["use_taxonomy"] and hasattr(self, 'tax_encoder'):
                nn.init.kaiming_uniform_(self.tax_encoder.weight.data, nonlinearity='relu')

            # Initialize linear layer
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
            nn.init.zeros_(self.linear.bias)

            # Initialize classifier layers
            for name, param in self.classifier.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif 'bias' in name:
                    nn.init.zeros_(param)

        except Exception as e:
            print(f"Error initializing weights: {e}")

    def get_attention_maps(self) -> List[torch.Tensor]:
        """
        Returns attention maps from all transformer layers.

        Returns:
            List of attention tensor maps, one for each layer
        """
        try:
            if hasattr(self.transformer_encoder, 'attention_maps'):
                return self.transformer_encoder.attention_maps
            else:
                print("Warning: Transformer encoder does not provide attention maps")
                return []
        except Exception as e:
            print(f"Error retrieving attention maps: {e}")
            return []

    def create_masked_attention_matrix(self, sz: int) -> torch.Tensor:
        """
        Creates a masked attention matrix for causal attention.

        Args:
            sz: Size of the attention matrix

        Returns:
            Masked attention matrix
        """
        try:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            return mask
        except Exception as e:
            print(f"Error creating masked attention matrix: {e}")
            # Return identity matrix as fallback
            return torch.eye(sz)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                debugging: bool = False) -> torch.Tensor:
        """
        Forward pass of the model with special attention logic.

        Args:
            src: Source tensor of token indices [batch_size, seq_len]
            src_mask: Optional attention mask tensor
            debugging: Whether to enable debugging output

        Returns:
            Output logits tensor
        """
        try:
            batch_size, seq_len = src.shape

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

            # Create standard attention mask if not provided
            if src_mask is None:
                src_mask = torch.ones(seq_len, seq_len, device=src.device)

            # Pass through transformer encoder
            output = self.transformer_encoder(x, src_mask, is_causal=False)

            # Save transformer output as hidden states
            self.last_hidden_states = output.clone()

            # Normalize and apply linear projection
            output = self.norm(output)
            output_logits = self.linear(output)

            # Find mask token positions
            mask_positions = (src == self.vocab["<mask>"])
            self.mask_positions = mask_positions

            # Extract hidden states only for masked positions
            if torch.any(mask_positions):
                masked_embeddings = self.last_hidden_states[mask_positions]
                cls_logits = self.classifier(masked_embeddings).squeeze()
                # Handle single element case
                if cls_logits.dim() == 0:
                    self.cls_logits = cls_logits.unsqueeze(0)
                else:
                    self.cls_logits = cls_logits
            else:
                # Create empty tensor with correct dimensions
                self.cls_logits = torch.zeros((0,), device=src.device)

            if debugging:
                check_logits(output_logits)

            return output_logits

        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Return zeros as fallback
            return torch.zeros((batch_size, seq_len, self.ntokens), device=src.device)

    def _create_integrated_attention_mask(self, src: torch.Tensor,
                                          batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Creates an integrated attention mask that combines causal logic
        with special attention for class tokens.

        Args:
            src: Source tensor of token indices
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Integrated attention mask
        """
        try:
            device = src.device

            # Part 1: Create base causal mask
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1
            causal_mask = causal_mask.float().masked_fill(~causal_mask, float('-inf')).masked_fill(causal_mask, 0.0)

            # Part 2: Identify special token positions
            pad_idx = self.vocab["<pad>"] if "<pad>" in self.vocab.get_stoi() else -1
            mask_idx = self.vocab["<mask>"] if "<mask>" in self.vocab.get_stoi() else -1
            cls0_idx = self.vocab["<cls0>"] if "<cls0>" in self.vocab.get_stoi() else -1
            cls1_idx = self.vocab["<cls1>"] if "<cls1>" in self.vocab.get_stoi() else -1

            # Identify special token positions for the entire batch
            pad_positions = (src == pad_idx)
            cls0_positions = (src == cls0_idx)
            cls1_positions = (src == cls1_idx)
            mask_positions = (src == mask_idx)
            class_positions = cls0_positions | cls1_positions | mask_positions

            # Part 3: Apply modifications for each batch element
            batch_masks = []

            for batch_idx in range(batch_size):
                # Copy of causal mask for this batch element
                current_mask = causal_mask.clone()

                # Apply special rules for this batch element
                for i in range(seq_len):
                    for j in range(seq_len):
                        # Rule 1: Padding tokens neither receive nor provide attention
                        if pad_positions[batch_idx, i] or pad_positions[batch_idx, j]:
                            current_mask[i, j] = float('-inf')

                        # Rule 2: Amplify attention towards class tokens
                        elif class_positions[batch_idx, j] and not (
                                pad_positions[batch_idx, i] or class_positions[batch_idx, i]):
                            # Normal -> Class: maintain causal structure but amplify if allowed
                            if current_mask[i, j] != float('-inf'):
                                # Reduce attention penalty for preferential attention
                                current_mask[i, j] = min(current_mask[i, j], -0.1)

                        # Rule 3: Moderate attention between class tokens
                        elif class_positions[batch_idx, i] and class_positions[batch_idx, j] and i != j:
                            if current_mask[i, j] != float('-inf'):
                                current_mask[i, j] = min(current_mask[i, j], -0.05)

                batch_masks.append(current_mask)

            # Return appropriate mask format based on batch size
            return batch_masks[0] if batch_size == 1 else self._handle_batch_attention_masks(batch_masks)

        except Exception as e:
            print(f"Error creating integrated attention mask: {e}")
            # Return a simple mask as fallback
            return torch.ones(seq_len, seq_len, device=src.device)

    def _handle_batch_attention_masks(self, batch_masks: List[torch.Tensor]) -> torch.Tensor:
        """
        Handles different attention masks for each batch element.

        Args:
            batch_masks: List of attention masks, one for each batch element

        Returns:
            Attention mask compatible with the transformer
        """
        try:
            # Convert list of masks to a tensor
            stacked_masks = torch.stack(batch_masks)  # [batch_size, seq_len, seq_len]

            # Check if encoder supports batch masks
            if isinstance(self.transformer_encoder, CustomTransformerEncoder):
                # If it supports multiple masks, return 3D tensor
                return stacked_masks
            else:
                # Otherwise, use a standard mask for compatibility
                seq_len = batch_masks[0].size(0)
                return torch.ones(seq_len, seq_len, device=batch_masks[0].device)

        except Exception as e:
            print(f"Error handling batch attention masks: {e}")
            # Return a simple mask as fallback
            seq_len = batch_masks[0].size(0) if batch_masks else 1
            device = batch_masks[0].device if batch_masks else torch.device('cpu')
            return torch.ones(seq_len, seq_len, device=device)