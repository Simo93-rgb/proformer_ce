import math
from typing import Optional, Tuple, List, Union
import torch
import torch.nn.functional as F
from torch import nn


class CustomMultiheadAttention(nn.Module):
    """
    Custom multihead attention module that stores attention weights for visualization.

    Attributes:
        mha: PyTorch MultiheadAttention module
        last_attn_weights: Stores the last computed attention weights
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize the custom multihead attention module.

        Args:
            embed_dim: Dimension of embedding vectors
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        try:
            self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
            self.last_attn_weights = None
        except Exception as e:
            print(f"Error initializing CustomMultiheadAttention: {e}")
            raise

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                average_attn_weights: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through multihead attention with attention weight storage.

        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            attn_mask: Optional attention mask
            key_padding_mask: Optional mask for padding keys
            need_weights: Whether to return attention weights
            average_attn_weights: Whether to average attention weights across heads

        Returns:
            Tuple containing output tensor and attention weights
        """
        try:
            output, attn_weights = self.mha(
                query, key, value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=average_attn_weights
            )
            self.last_attn_weights = attn_weights
            return output, attn_weights
        except Exception as e:
            print(f"Error in CustomMultiheadAttention.forward: {e}")
            # Return empty tensors in case of error
            device = query.device if query is not None else torch.device('cpu')
            return torch.zeros_like(query), torch.zeros((1, 1, 1), device=device)


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom transformer encoder layer with customizable normalization and activation.

    Attributes:
        self_attn: Custom multihead attention module
        linear1: First linear transformation in feed-forward network
        dropout: Dropout module for feed-forward network
        linear2: Second linear transformation in feed-forward network
        norm1: Layer normalization for attention sublayer
        norm2: Layer normalization for feed-forward sublayer
        dropout1: Dropout for attention sublayer
        dropout2: Dropout for feed-forward sublayer
        activation: Activation function
        norm_first: Whether to apply normalization before or after sublayers
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu",
                 layer_norm_eps: float = 1e-5, norm_first: bool = False):
        """
        Initialize the custom transformer encoder layer.

        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Dimension of feed-forward network
            dropout: Dropout probability
            activation: Activation function name
            layer_norm_eps: Epsilon for layer normalization
            norm_first: Whether to apply normalization before sublayers
        """
        super().__init__()
        try:
            self.self_attn = CustomMultiheadAttention(d_model, nhead, dropout=dropout)

            # Feed-forward network
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)

            # Set activation function
            if hasattr(F, activation):
                self.activation = getattr(F, activation)
            else:
                print(f"Warning: Activation {activation} not found, using relu instead")
                self.activation = F.relu

            self.norm_first = norm_first
        except Exception as e:
            print(f"Error initializing CustomTransformerEncoderLayer: {e}")
            raise

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward block of the transformer layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor after feed-forward transformation
        """
        try:
            x = self.linear2(self.dropout(self.activation(self.linear1(x))))
            return self.dropout2(x)
        except Exception as e:
            print(f"Error in feed-forward block: {e}")
            return x

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the transformer encoder layer.

        Args:
            src: Source tensor
            src_mask: Optional source attention mask
            src_key_padding_mask: Optional source key padding mask

        Returns:
            Output tensor after transformation
        """
        try:
            x = src

            if self.norm_first:
                # Pre-normalization architecture
                normalized_x = self.norm1(x)
                x2, weights = self.self_attn(normalized_x, normalized_x, normalized_x,
                                             attn_mask=src_mask,
                                             key_padding_mask=src_key_padding_mask)
                x = x + self.dropout1(x2)
                x = x + self._ff_block(self.norm2(x))
            else:
                # Post-normalization architecture
                x2, weights = self.self_attn(x, x, x,
                                             attn_mask=src_mask,
                                             key_padding_mask=src_key_padding_mask)
                x = self.norm1(x + self.dropout1(x2))
                x = self.norm2(x + self._ff_block(x))

            return x
        except Exception as e:
            print(f"Error in CustomTransformerEncoderLayer.forward: {e}")
            return src


class CustomTransformerEncoder(nn.Module):
    """
    Custom transformer encoder that stacks multiple encoder layers.

    Attributes:
        layers: List of transformer encoder layers
        num_layers: Number of encoder layers
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        """
        Initialize the custom transformer encoder.

        Args:
            encoder_layer: Transformer encoder layer to be stacked
            num_layers: Number of layers to stack
        """
        super().__init__()
        try:
            # Create multiple copies of the encoder layer
            self.layers = nn.ModuleList([encoder_layer] + [
                CustomTransformerEncoderLayer(
                    encoder_layer.self_attn.mha.embed_dim,
                    encoder_layer.self_attn.mha.num_heads,
                    dim_feedforward=encoder_layer.linear1.out_features,
                    dropout=encoder_layer.dropout.p,
                    activation="gelu" if encoder_layer.activation == F.gelu else "relu",
                    norm_first=encoder_layer.norm_first
                )
                for _ in range(num_layers - 1)
            ])
            self.num_layers = num_layers
        except Exception as e:
            print(f"Error initializing CustomTransformerEncoder: {e}")
            raise

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Forward pass through all transformer encoder layers.

        Args:
            src: Source tensor
            mask: Optional attention mask
            src_key_padding_mask: Optional key padding mask
            is_causal: Whether the attention is causal

        Returns:
            Output tensor after transformation through all layers
        """
        try:
            output = src

            # Pass input through each layer sequentially
            for layer in self.layers:
                output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

            return output
        except Exception as e:
            print(f"Error in CustomTransformerEncoder.forward: {e}")
            return src

    @property
    def attention_maps(self) -> List[torch.Tensor]:
        """
        Get attention weights from all layers for visualization.

        Returns:
            List of attention weight tensors from each layer
        """
        try:
            return [layer.self_attn.last_attn_weights for layer in self.layers]
        except Exception as e:
            print(f"Error retrieving attention maps: {e}")
            return []


class PositionalEncoding(nn.Module):
    """
    Positional encoding module for transformer models.

    Adds positional information to input embeddings using sine and cosine functions.

    Attributes:
        dropout: Dropout module
        pe: Pre-computed positional encoding buffer
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        """
        Initialize positional encoding module.

        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length to pre-compute
        """
        super().__init__()
        try:
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding matrix
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

            # Initialize positional encoding buffer
            pe = torch.zeros(max_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            pe[:, 0, 1::2] = torch.cos(position * div_term)

            # Register buffer to ensure it's saved and moved with the module
            self.register_buffer('pe', pe)
        except Exception as e:
            print(f"Error initializing PositionalEncoding: {e}")
            raise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape [seq_len, batch_size, embedding_dim]

        Returns:
            Tensor with positional encoding added
        """
        try:
            # Add positional encoding and apply dropout
            x = x + self.pe[:x.size(0)]
            return self.dropout(x)
        except Exception as e:
            print(f"Error in PositionalEncoding.forward: {e}")
            return x