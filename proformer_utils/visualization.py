import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np
import torch
from typing import List, Optional, Union, Tuple, Dict, Any
from proformer_utils.tensor_utils import create_attention_mask
from config import MODELS_DIR
from proformer_utils.dataloader import Dataloader


def save_attention_maps(
        model: torch.nn.Module,
        data_loader: Dataloader,
        opt: Dict[str, Any],
        output_dir: str = f"{MODELS_DIR}/attention_maps"
) -> None:
    """
    Save model's attention maps for a single batch of data.

    Args:
        model: The transformer model
        data_loader: The data loader instance
        opt: Configuration options dictionary
        output_dir: Directory where attention maps will be saved
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            # Get a single batch
            data, targets = data_loader.get_batch_from_list(data_loader.test_data, 0)
            attn_mask = create_attention_mask(data.size(0), opt["device"])

            # Perform forward pass
            _ = model(data, attn_mask)

            # Get attention maps
            attention_maps = model.get_attention_maps()
            if not attention_maps:
                print("No attention maps available in the model")
                return

            # Get input tokens using the correct vocabulary
            input_tokens = [data_loader.vocab.get_itos()[idx] for idx in data[0].tolist()]

            total_maps = len(attention_maps) * attention_maps[0].shape[0]
            print(f"Generating {total_maps} attention maps...")

            # Save maps for each layer and head
            max_tokens = 20
            input_tokens = input_tokens[:max_tokens]
            for layer_idx in range(len(attention_maps)):
                num_heads = attention_maps[layer_idx].shape[0]
                for head_idx in range(num_heads):
                    attn_map = attention_maps[layer_idx][head_idx][:max_tokens, :max_tokens]
                    output_file = f"{output_dir}/attn_layer{layer_idx}_head{head_idx}.png"
                    plot_attention_map_with_blocks(
                        [torch.tensor(attn_map)],  # Pass only the cropped map
                        output_file,
                        tokens=input_tokens,
                        layer_idx=0,
                        head_idx=0,
                        title=f"Attention Map"
                    )

            print(f"Attention maps saved to {output_dir}")
    except Exception as e:
        print(f"Error generating attention maps: {e}")


def plot_attention_map(
        attention_maps: List[torch.Tensor],
        output_file: str,
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        title: str = "Attention Map"
) -> None:
    """
    Visualize and save attention maps for a given layer and head,
    using numerical indices in the heatmap and a separate legend.

    Args:
        attention_maps: List of attention tensors
        output_file: Path to save the output file
        tokens: List of tokens to label the axes (optional)
        layer_idx: Index of the layer to visualize (default: 0)
        head_idx: Index of the attention head to visualize (default: 0)
        title: Title of the plot
    """
    try:
        if layer_idx >= len(attention_maps):
            print(f"Error: layer index {layer_idx} out of range. Available: {len(attention_maps)} layers.")
            return

        attn = attention_maps[layer_idx]
        if len(attn.shape) == 2:
            attn = attn.unsqueeze(0)  # Add head dimension for consistency

        if head_idx >= attn.shape[0]:
            print(f"Error: head index {head_idx} out of range. Available: {attn.shape[0]} heads.")
            return

        attn = attn[head_idx].detach().cpu().numpy()

        # Create heatmap
        create_attention_heatmap(
            attn,
            output_file.replace(".png", "_heatmap.png"),
            layer_idx,
            head_idx,
            title
        )

        # Create separate legend if tokens are provided
        if tokens is not None:
            create_token_legend(tokens, output_file.replace(".png", "_legend.png"))

    except Exception as e:
        print(f"Error plotting attention map: {e}")


def create_attention_heatmap(
        attention_matrix: np.ndarray,
        output_file: str,
        layer_idx: int,
        head_idx: int,
        title: str
) -> None:
    """
    Create and save a heatmap visualization of attention weights.

    Args:
        attention_matrix: Numpy array of attention weights
        output_file: Path to save the output file
        layer_idx: Index of the layer
        head_idx: Index of the attention head
        title: Title of the plot
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attention_matrix, cmap='viridis')

        # Show numerical indices on the heatmap
        ax.set_xticks(np.arange(attention_matrix.shape[1]))
        ax.set_yticks(np.arange(attention_matrix.shape[0]))
        ax.set_xticklabels(np.arange(1, attention_matrix.shape[1] + 1))  # Start from 1 for clarity
        ax.set_yticklabels(np.arange(1, attention_matrix.shape[0] + 1))

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        ax.set_title(f"{title} - Layer {layer_idx}, Head {head_idx}")
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query Index")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"Error creating attention heatmap: {e}")


def create_token_legend(tokens: List[str], output_file: str) -> None:
    """
    Create and save a legend mapping indices to token values.

    Args:
        tokens: List of token strings
        output_file: Path to save the legend image
    """
    try:
        plt.figure(figsize=(8, len(tokens) // 2))  # Adjust height based on number of tokens
        plt.axis('off')
        patches = [mpatches.Patch(color='none', label=f"{i + 1}: {token}")
                   for i, token in enumerate(tokens)]  # Start index from 1
        plt.legend(handles=patches, loc='center', fontsize='small')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()

    except Exception as e:
        print(f"Error creating token legend: {e}")


def plot_attention_map_with_blocks(
        attention_maps: List[torch.Tensor],
        output_file: str,
        tokens: Optional[List[str]] = None,
        token_groups: Optional[List[int]] = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        title: str = "Attention Map"
) -> None:
    """
    Visualize and save attention maps with block separators between token groups.

    Args:
        attention_maps: List of attention tensors
        output_file: Path to save the output file
        tokens: List of tokens to label the axes (optional)
        token_groups: List of integers indicating token groups (e.g., [0,0,1,1,2,2])
        layer_idx: Index of the layer to visualize
        head_idx: Index of the attention head to visualize
        title: Title of the plot
    """
    try:
        if layer_idx >= len(attention_maps):
            print(f"Error: layer index {layer_idx} out of range. Available: {len(attention_maps)} layers.")
            return

        attn = attention_maps[layer_idx]
        if len(attn.shape) == 2:
            attn = attn.unsqueeze(0)  # Add head dimension for consistency

        if head_idx >= attn.shape[0]:
            print(f"Error: head index {head_idx} out of range. Available: {attn.shape[0]} heads.")
            return

        attn = attn[head_idx].detach().cpu().numpy()

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attn, cmap='viridis')

        # Add block separators if token groups are provided
        if token_groups is not None:
            add_group_separators(ax, token_groups)

        # Add token labels if provided
        configure_axis_labels(ax, tokens)

        ax.set_title(f"{title} - Layer {layer_idx}, Head {head_idx}")
        ax.set_xlabel("Key Index")
        ax.set_ylabel("Query Index")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

    except Exception as e:
        print(f"Error plotting attention map with blocks: {e}")


def add_group_separators(ax: plt.Axes, token_groups: List[int]) -> None:
    """
    Add separator lines between different token groups in the plot.

    Args:
        ax: Matplotlib axes object
        token_groups: List of integers indicating token groups
    """
    try:
        group_changes = [i for i in range(1, len(token_groups)) if token_groups[i] != token_groups[i - 1]]
        for pos in group_changes:
            ax.axhline(pos - 0.5, color='red', linewidth=1)
            ax.axvline(pos - 0.5, color='red', linewidth=1)

    except Exception as e:
        print(f"Error adding group separators: {e}")


def configure_axis_labels(ax: plt.Axes, tokens: Optional[List[str]]) -> None:
    """
    Configure axis labels for the attention map plot.

    Args:
        ax: Matplotlib axes object
        tokens: List of tokens to use as labels (optional)
    """
    try:
        if tokens is not None:
            ax.set_xticks(np.arange(len(tokens)))
            ax.set_yticks(np.arange(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    except Exception as e:
        print(f"Error configuring axis labels: {e}")