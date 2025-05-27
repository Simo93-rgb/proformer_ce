import math
import torch
import torch.nn.functional as F
import pickle
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List, Union
from proformer_utils.dataloader import Dataloader
from proformer_utils.params import bpi_params, parse_params
from proformer_utils.visualization import plot_attention_map
from proformer_utils.proformer import TransformerModel
from taxonomy import TaxonomyEmbedding
from config import DATA_DIR, MODELS_DIR
from proformer_utils.tensor_utils import create_attention_mask, filter_padding, get_ranked_metrics
from proformer_utils.metrics import compute_classification_metrics


def calculate_sequence_loss(model: torch.nn.Module, data: torch.Tensor,
                            targets: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Calculate the sequence prediction loss.

    Args:
        model: The model to evaluate
        data: Input data tensor
        targets: Expected target tensor
        attn_mask: Attention mask tensor

    Returns:
        Tuple containing loss value, flattened output and filtered targets
    """
    seq_len = data.size(0)
    output = model(data, attn_mask)
    output_flat = output.view(-1, model.ntokens)
    targets_filtered, output_flat_filtered = filter_padding(targets, output_flat)
    loss = seq_len * F.cross_entropy(output_flat_filtered, targets_filtered).item()

    return loss, output_flat_filtered, targets_filtered


def calculate_ranked_metrics(output_flat: torch.Tensor, targets: torch.Tensor,
                             accs: Dict[int, float]) -> Dict[int, float]:
    """
    Calculate ranked accuracy metrics (top-k).

    Args:
        output_flat: Model output logits
        targets: Expected targets
        accs: Dictionary with k keys and initialized values

    Returns:
        Updated dictionary with accuracy values
    """
    return get_ranked_metrics(accs, output_flat, targets)


def calculate_classification_metrics(cls_logits: torch.Tensor,
                                     batch_labels: torch.Tensor) -> Tuple[float, int, int, int, int, int]:
    """
    Calculate classification metrics with weighted loss to handle class imbalance.

    Args:
        cls_logits: Classification logits
        batch_labels: True labels

    Returns:
        Tuple containing (loss, true positives, false positives, true negatives, false negatives, num_samples)
    """
    if cls_logits.numel() == 0 or batch_labels.numel() == 0:
        return 0.0, 0, 0, 0, 0, 0

    # Calculate class distribution
    positive_mask = (batch_labels == 1)
    negative_mask = (batch_labels == 0)
    num_positive = torch.sum(positive_mask).item()
    num_negative = torch.sum(negative_mask).item()

    # Calculate weight for positive class based on distribution
    try:
        pos_weight_value = num_negative / num_positive if num_positive > 0 and num_negative > 0 else 3.0
        # Limit weight to a reasonable value
        pos_weight_value = min(max(pos_weight_value, 1.0), 5.0)
    except ZeroDivisionError:
        pos_weight_value = 3.0  # Default if calculation fails

    # Apply weight to loss
    pos_weight = torch.tensor([pos_weight_value], device=cls_logits.device)
    cls_loss = F.binary_cross_entropy_with_logits(
        cls_logits,
        batch_labels,
        pos_weight=pos_weight
    )

    # Calculate metrics
    with torch.no_grad():
        cls_preds = (torch.sigmoid(cls_logits) > 0.5).float()
        tp = torch.sum((cls_preds == 1) & (batch_labels == 1)).item()
        fp = torch.sum((cls_preds == 1) & (batch_labels == 0)).item()
        tn = torch.sum((cls_preds == 0) & (batch_labels == 0)).item()
        fn = torch.sum((cls_preds == 0) & (batch_labels == 1)).item()

    return cls_loss.item(), tp, fp, tn, fn, batch_labels.size(0)


def initialize_prediction_file(data_type: str) -> Optional[str]:
    """
    Initialize a file for saving model predictions.

    Args:
        data_type: Type of data being evaluated ("test" or "valid")

    Returns:
        Path to the created file or None if not needed
    """
    if data_type != "test":
        return None

    file_path = f"{MODELS_DIR}/predictions_class_{data_type}.txt"
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"MODEL PREDICTIONS - {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n")
            f.write(f"{'=' * 80}\n\n")
        return file_path
    except IOError as e:
        print(f"Error creating prediction file: {e}")
        return None


def process_batch_data(model: torch.nn.Module, data: torch.Tensor,
                       targets: torch.Tensor, attn_mask: torch.Tensor,
                       loader: Dataloader, batch_idx: int,
                       data_type: str) -> Tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a batch of data for evaluation.

    Args:
        model: The model to evaluate
        data: Input data tensor
        targets: Expected target tensor
        attn_mask: Attention mask tensor
        loader: Dataloader instance
        batch_idx: Current batch index
        data_type: Type of data being evaluated

    Returns:
        Tuple containing sequence loss, output, targets, classification logits and batch labels
    """
    # Sequence loss
    seq_loss, output_flat, targets_filtered = calculate_sequence_loss(model, data, targets, attn_mask)

    # Get mask positions and batch labels
    mask_positions = model.mask_positions  # (batch_size, seq_len-1)

    batch_labels = torch.tensor([], device=data.device)
    has_mask = torch.tensor([], dtype=torch.bool, device=data.device)

    try:
        batch_labels, has_mask = loader.get_masked_batch_labels(data_type, batch_idx, mask_positions)
    except (ValueError, IndexError) as e:
        print(f"Error retrieving batch labels: {e}")

    cls_logits = model.cls_logits if hasattr(model, 'cls_logits') else torch.tensor([], device=data.device)

    return seq_loss, output_flat, targets_filtered, cls_logits, batch_labels


def evaluate(model: torch.nn.Module, eval_data: List, loader: Dataloader,
             opt: Dict[str, Any], data_type: str = "valid") -> Tuple[float, Dict[int, float], Dict[str, float]]:
    """
    Evaluate the model on validation or test set, calculating metrics only on masked positions.

    Args:
        model: The model to evaluate
        eval_data: List of evaluation data batches
        loader: Dataloader instance
        opt: Options dictionary
        data_type: Type of data being evaluated ("valid" or "test")

    Returns:
        Tuple containing (loss, accuracy metrics dictionary, classification metrics dictionary)
    """
    model.eval()
    total_loss = 0.
    cls_total_loss = 0.
    accs = {1: 0., 3: 0., 5: 0.}
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    cls_samples = 0
    num_batches = 0

    # Initialize prediction file if needed
    prediction_file = initialize_prediction_file(data_type)

    with torch.no_grad():
        for batch_idx in range(len(eval_data)):
            try:
                data, targets = loader.get_batch_from_list(eval_data, batch_idx)
                attn_mask = create_attention_mask(data.shape[0], opt["device"])

                # Process batch data
                seq_loss, output_flat, targets_filtered, cls_logits, batch_labels = process_batch_data(
                    model, data, targets, attn_mask, loader, batch_idx, data_type
                )

                total_loss += seq_loss

                # Ranked metrics
                accs = calculate_ranked_metrics(output_flat, targets_filtered, accs)

                # Skip if no masked tokens
                if cls_logits.numel() == 0 or batch_labels.numel() == 0:
                    continue

                # Save predictions to file if in test mode
                if prediction_file:
                    save_predictions_to_file(model, data, loader, batch_idx, prediction_file, threshold=0.5)

                cls_loss, tp, fp, tn, fn, samples = calculate_classification_metrics(cls_logits, batch_labels)

                cls_total_loss += cls_loss
                true_positives += tp
                false_positives += fp
                true_negatives += tn
                false_negatives += fn
                cls_samples += samples
                num_batches += 1

            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                continue

        # Normalization
        for k in accs:
            accs[k] = accs[k] / num_batches if num_batches > 0 else 0
        loss = total_loss / num_batches if num_batches > 0 else 0

        # Final classification metrics
        cls_metrics = calculate_final_classification_metrics(
            true_positives, false_positives, true_negatives, false_negatives,
            cls_total_loss, num_batches, cls_samples
        )

        print(f"cls_samples: {cls_samples}, cls_metrics: {cls_metrics}")
    return loss, accs, cls_metrics


def calculate_final_classification_metrics(true_positives: int, false_positives: int,
                                           true_negatives: int, false_negatives: int,
                                           cls_total_loss: float, num_batches: int,
                                           cls_samples: int) -> Dict[str, float]:
    """
    Calculate final classification metrics after processing all batches.

    Args:
        true_positives: Total true positives
        false_positives: Total false positives
        true_negatives: Total true negatives
        false_negatives: Total false negatives
        cls_total_loss: Total classification loss
        num_batches: Number of processed batches
        cls_samples: Number of classification samples

    Returns:
        Dictionary with classification metrics
    """
    cls_metrics = {}
    if cls_samples > 0:
        try:
            preds = torch.tensor([1] * true_positives + [1] * false_positives +
                                 [0] * true_negatives + [0] * false_negatives)
            labels = torch.tensor([1] * true_positives + [0] * false_positives +
                                  [0] * true_negatives + [1] * false_negatives)
            cls_metrics = compute_classification_metrics(preds=preds, labels=labels)
            cls_metrics['loss'] = cls_total_loss / num_batches if num_batches > 0 else 0
        except Exception as e:
            print(f"Error calculating classification metrics: {e}")
            cls_metrics = {'loss': 0, 'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

    return cls_metrics


def format_sequence_context(tokens: List[str], mask_position: int) -> str:
    """
    Format a sequence context with masked position for readable output.

    Args:
        tokens: List of tokens in the sequence
        mask_position: Position of the mask token

    Returns:
        Formatted context string
    """
    # Get context (5 tokens before and after)
    start = max(0, mask_position - 5)
    end = min(len(tokens), mask_position + 6)
    context = tokens[start:mask_position] + ["[MASK]"] + tokens[mask_position + 1:end]
    return " ".join(context)


def format_prediction_result(pred_class: int, true_class: int) -> str:
    """
    Format the prediction result with correct/incorrect indicator.

    Args:
        pred_class: Predicted class
        true_class: True class

    Returns:
        Formatted result string
    """
    if int(true_class) == pred_class:
        return "  Result: ✓ CORRECT"
    else:
        return "  Result: ✗ INCORRECT"


def save_predictions_to_file(model: torch.nn.Module, data: torch.Tensor, loader: Dataloader,
                             batch_idx: int, file_path: str, threshold: float = 0.5) -> None:
    """
    Save class predictions to a formatted text file.

    Args:
        model: The transformer model
        data: Batch of input data
        loader: Dataloader for accessing original data
        batch_idx: Current batch index
        file_path: Output file path
        threshold: Threshold for binary classification
    """
    device = next(model.parameters()).device

    try:
        # Check if model has classification logits
        if not hasattr(model, 'cls_logits') or model.cls_logits.numel() == 0:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"No predictions available for batch {batch_idx} (no <mask> tokens found)\n")
            return

        # Convert logits to probabilities and classes
        probs = torch.sigmoid(model.cls_logits)
        predicted_classes = (probs >= threshold).int()

        # Get original tokens for context visualization
        token_dict = {i: token for i, token in enumerate(loader.vocab.get_itos())}

        # Find positions of <mask> tokens in each sequence
        mask_positions = model.mask_positions  # [batch_size, seq_len]

        # Get real labels if available
        batch_labels = torch.tensor([], device=device)
        has_mask = torch.tensor([], dtype=torch.bool, device=device)
        try:
            batch_labels, has_mask = loader.get_masked_batch_labels("test", batch_idx, mask_positions)
        except Exception as e:
            print(f"Error retrieving labels: {e}")

        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'=' * 80}\n")
            f.write(f"PREDICTIONS FOR BATCH {batch_idx}\n")
            f.write(f"{'=' * 80}\n\n")

            pred_counter = 0
            # For each sequence in the batch that contains at least one <mask> token
            for seq_idx, seq in enumerate(data):
                seq_mask_positions = torch.where(mask_positions[seq_idx])[0]

                if len(seq_mask_positions) == 0:
                    continue

                tokens = [token_dict.get(idx.item(), "<UNK>") for idx in seq]
                f.write(f"Sequence {seq_idx}:\n")
                f.write(f"Tokens: {' '.join(tokens)}\n\n")

                # For each <mask> token in the sequence
                for pos_idx, pos in enumerate(seq_mask_positions):
                    # Calculate index in prediction list
                    if pred_counter >= len(probs):
                        break

                    prob = probs[pred_counter].item()
                    pred_class = predicted_classes[pred_counter].item()

                    # Format and write context
                    context = format_sequence_context(tokens, pos)

                    f.write(f"  Mask token position: {pos}\n")
                    f.write(f"  Context: {context}\n")
                    f.write(f"  Class 1 probability: {prob:.4f}\n")
                    f.write(f"  Predicted class: {pred_class}\n")

                    # Add true label if available
                    if batch_labels.numel() > pred_counter:
                        true_class = batch_labels[pred_counter].item()
                        f.write(f"  True class: {int(true_class)}\n")
                        f.write(f"{format_prediction_result(pred_class, true_class)}\n")

                    f.write("\n")
                    pred_counter += 1

            f.write(f"\nSummary: {pred_counter} predictions in batch {batch_idx}\n")
            f.write(f"{'=' * 80}\n\n")

    except Exception as e:
        print(f"Error saving predictions to file: {e}")
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(f"Error processing batch {batch_idx}: {str(e)}\n")
        except IOError:
            pass