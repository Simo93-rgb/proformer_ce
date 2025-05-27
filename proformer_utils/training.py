import torch
import pandas as pd
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Union
from proformer_utils.dataloader import Dataloader
from proformer_utils.evaluation import calculate_classification_metrics
from proformer_utils.tensor_utils import create_attention_mask, filter_padding


def prepare_batch_data(loader: Dataloader, batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare batch data from the dataloader.

    Args:
        loader: Dataloader instance with data
        batch_idx: Index of the current batch

    Returns:
        Tuple containing (input data, target data)
    """
    try:
        data, targets = loader.get_batch_from_list(loader.train_data, batch_idx)
        return data, targets
    except Exception as e:
        print(f"Error preparing batch data: {e}")
        device = next(iter(loader.train_data)).device if loader.train_data else torch.device('cpu')
        return torch.tensor([], device=device), torch.tensor([], device=device)


def calculate_sequence_loss(model: torch.nn.Module, output_flat: torch.Tensor,
                            targets: torch.Tensor) -> torch.Tensor:
    """
    Calculate sequence prediction loss, ignoring padding tokens.

    Args:
        model: The model being trained
        output_flat: Flattened output logits from the model
        targets: Target token indices

    Returns:
        Sequence loss tensor
    """
    try:
        # Create mask for non-padding tokens
        pad_idx = model.vocab["<pad>"]
        non_pad_mask = targets != pad_idx

        # Calculate loss only on non-padding tokens
        if torch.any(non_pad_mask):
            return F.cross_entropy(
                output_flat[non_pad_mask],
                targets[non_pad_mask]
            )
        else:
            return torch.tensor(0.0, device=targets.device, requires_grad=True)
    except Exception as e:
        print(f"Error calculating sequence loss: {e}")
        return torch.tensor(0.0, device=targets.device, requires_grad=True)


def calculate_classification_loss(model: torch.nn.Module, loader: Dataloader,
                                  batch_idx: int, debugging: bool = False) -> torch.Tensor:
    """
    Calculate classification loss for masked token prediction.

    Args:
        model: The model being trained
        loader: Dataloader instance
        batch_idx: Index of the current batch
        debugging: Whether to print debug information

    Returns:
        Classification loss tensor
    """
    try:
        # Initialize with zero loss
        device = next(model.parameters()).device
        cls_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Check if model has classification logits and they're not empty
        if not hasattr(model, 'cls_logits') or model.cls_logits.numel() == 0:
            return cls_loss

        # Get labels for the current batch
        batch_labels = loader.get_labels_for_batch_type("train", batch_idx)
        mask_positions = model.mask_positions

        if batch_labels.numel() == 0 or mask_positions is None:
            return cls_loss

        # Get labels specifically for masked positions
        batch_labels, has_mask = loader.get_masked_batch_labels("train", batch_idx, mask_positions)

        # Debug info if requested
        if debugging and batch_labels.size(0) == model.cls_logits.size(0) and batch_labels.numel() > 0:
            print(f"Batch {batch_idx}: labels shape={batch_labels.shape}, cls_logits shape={model.cls_logits.shape}")
            print(f"Label values: {batch_labels}")
            print(f"Logits values: {model.cls_logits.detach().cpu().numpy()}")

        # Calculate classification loss if dimensions match
        if batch_labels.size(0) == model.cls_logits.size(0) and batch_labels.numel() > 0:
            # Use positive weight in BCE loss (implemented in calculate_classification_metrics)
            cls_loss_value, _, _, _, _, _ = calculate_classification_metrics(model.cls_logits, batch_labels)
            cls_loss = torch.tensor(cls_loss_value, device=device, requires_grad=True)

        return cls_loss
    except Exception as e:
        print(f"Error calculating classification loss: {e}")
        return torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)


def update_model_parameters(model: torch.nn.Module, loss: torch.Tensor,
                            optimizer: torch.optim.Optimizer,
                            gradient_clip: float = 0.5) -> None:
    """
    Perform backpropagation and update model parameters.

    Args:
        model: The model being trained
        loss: Combined loss tensor
        optimizer: Optimizer instance
        gradient_clip: Maximum gradient norm
    """
    try:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
    except Exception as e:
        print(f"Error updating model parameters: {e}")


def train(model: torch.nn.Module, opt: Dict[str, Any], loader: Dataloader,
          optimizer: torch.optim.Optimizer, debugging: bool = False) -> float:
    """
    Train the model for one epoch, ignoring padding tokens in the loss calculation.

    Args:
        model: The model to train
        opt: Configuration parameters
        loader: Data loader
        optimizer: Optimizer instance
        debugging: Print logit values for debugging

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    cls_total_loss = 0.0
    batches_processed = 0

    for batch_idx, batch in enumerate(loader.train_data):
        try:
            # Get data and targets
            data, targets = prepare_batch_data(loader, batch_idx)
            if data.numel() == 0 or targets.numel() == 0:
                continue

            # Create attention mask and forward pass
            attn_mask = create_attention_mask(data.shape[0], opt.get("device", "cpu"))
            output = model(data, attn_mask)
            output_flat = output.view(-1, model.ntokens)

            # Calculate sequence and classification losses
            seq_loss = calculate_sequence_loss(model, output_flat, targets)
            cls_loss = calculate_classification_loss(model, loader, batch_idx, debugging)

            # Combined loss and parameter update
            loss = seq_loss + cls_loss
            update_model_parameters(model, loss, optimizer, opt.get("gradient_clip", 0.5))

            # Accumulate losses
            total_loss += loss.item()
            if isinstance(cls_loss, torch.Tensor) and cls_loss.requires_grad:
                cls_total_loss += cls_loss.item()

            batches_processed += 1

        except Exception as e:
            print(f"Error during training of batch {batch_idx}: {e}")
            continue

    # Avoid division by zero
    return total_loss / max(batches_processed, 1)


def train_with_validation(model: torch.nn.Module, opt: Dict[str, Any], loader: Dataloader,
                          optimizer: torch.optim.Optimizer, validation_func: callable,
                          num_epochs: int = 1) -> Tuple[float, Dict[str, Any]]:
    """
    Train the model for multiple epochs with validation after each epoch.

    Args:
        model: The model to train
        opt: Configuration parameters
        loader: Data loader
        optimizer: Optimizer instance
        validation_func: Function to call for validation
        num_epochs: Number of epochs to train

    Returns:
        Tuple containing (final_train_loss, validation_metrics)
    """
    try:
        best_val_metric = float('-inf')
        best_val_results = {}
        train_losses = []

        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = train(model, opt, loader, optimizer)
            train_losses.append(train_loss)

            # Validate
            val_results = validation_func(model, loader.valid_data, loader, opt)

            # Track best validation results
            current_metric = val_results.get('primary_metric', 0.0)
            if current_metric > best_val_metric:
                best_val_metric = current_metric
                best_val_results = val_results

            print(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.4f}, val_metric={current_metric:.4f}")

        return train_losses[-1], best_val_results
    except Exception as e:
        print(f"Error in train_with_validation: {e}")
        return 0.0, {}