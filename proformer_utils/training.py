import torch
import pandas as pd
from typing import Tuple, Dict, Any
from proformer_utils.dataloader import Dataloader
from proformer_utils.evaluation import calculate_classification_metrics
from proformer_utils.tensor_utils import create_attention_mask, filter_padding
import torch.nn.functional as F


def train(model: torch.nn.Module, opt: Dict[str, Any], loader: Dataloader, optimizer: torch.optim.Optimizer) -> float:
    """
    Addestra il modello per una epoca, ignorando i token di padding nella loss.

    Args:
        model: Il modello da addestrare
        opt: Parametri di configurazione
        loader: Data loader
        optimizer: Ottimizzatore

    Returns:
        float: Loss media di training
    """
    model.train()
    total_loss = 0.0
    cls_total_loss = 0.0
    batches_processed = 0

    for batch_idx, batch in enumerate(loader.train_data):
        try:
            # Ottieni dati e target
            data, targets = loader.get_batch_from_list(loader.train_data, batch_idx)
            if data.numel() == 0 or targets.numel() == 0:
                continue

            # Crea maschera di attenzione e forward pass
            attn_mask = create_attention_mask(data.shape[0], opt["device"])
            output = model(data, attn_mask)
            output_flat = output.view(-1, model.ntokens)

            # Filtra i token di padding nella loss
            pad_idx = model.vocab["<pad>"]
            non_pad_mask = targets != pad_idx

            # Calcola la loss solo sui token non-padding
            if torch.any(non_pad_mask):
                seq_loss = F.cross_entropy(
                    output_flat[non_pad_mask],
                    targets[non_pad_mask]
                )
            else:
                seq_loss = torch.tensor(0.0, device=data.device, requires_grad=True)

            # Gestione loss di classificazione
            cls_loss = torch.tensor(0.0, device=data.device, requires_grad=True)
            if hasattr(model, 'cls_logits') and model.cls_logits.numel() > 0:
                # Ottieni le label per il batch corrente
                batch_labels = loader.get_labels_for_batch_type("train", batch_idx)
                mask_positions = model.mask_positions

                if batch_labels.numel() > 0 and mask_positions is not None:
                    batch_labels, has_mask = loader.get_masked_batch_labels("train", batch_idx, mask_positions)
                    if batch_labels.size(0) == model.cls_logits.size(0) and batch_labels.numel() > 0:
                        # Uso di pos_weight nella BCE loss (implementato in calculate_classification_metrics)
                        _, tp, fp, tn, fn, _ = calculate_classification_metrics(model.cls_logits, batch_labels)
                        cls_loss = torch.tensor(0.0, device=data.device, requires_grad=True)

            # Loss totale e backpropagation
            loss = seq_loss + cls_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.get("gradient_clip", 0.5))
            optimizer.step()

            # Accumula le loss
            total_loss += loss.item()
            if isinstance(cls_loss, torch.Tensor) and cls_loss.requires_grad:
                cls_total_loss += cls_loss.item()

            batches_processed += 1

        except Exception as e:
            print(f"Errore durante il training del batch {batch_idx}: {e}")
            continue

    # Evita divisione per zero
    return total_loss / max(batches_processed, 1)

# def train(model: torch.nn.Module, opt: Dict[str, Any], loader: Dataloader, optimizer: torch.optim.Optimizer) -> float:
#     """
#     Train the model for one epoch.
#
#     Args:
#         model (torch.nn.Module): The model to train.
#         opt (dict): Configuration options.
#         loader (Dataloader): Data loader.
#         optimizer (torch.optim.Optimizer): Optimizer.
#
#     Returns:
#         float: Average training loss.
#     """
#     model.train()
#     total_loss = 0.
#     cls_total_loss = 0.
#     for batch_idx, batch in enumerate(loader.train_data):
#         data, targets = loader.get_batch_from_list(loader.train_data, batch_idx)
#         attn_mask = create_attention_mask(data.shape[0], opt["device"])
#         output = model(data, attn_mask)
#         output_flat = output.view(-1, model.ntokens)
#         targets, output_flat = filter_padding(targets, output_flat)
#         seq_loss = F.cross_entropy(output_flat, targets)
#         cls_loss = 0.0
#         if hasattr(model, 'cls_logits') and model.cls_logits.numel() > 0:
#             batch_labels = loader.get_batch_labels(batch_idx)
#             if batch_labels.size(0) == model.cls_logits.size(0):
#                 cls_loss = F.binary_cross_entropy_with_logits(model.cls_logits, batch_labels) * 0.5
#         loss = seq_loss + cls_loss
#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), opt.get("gradient_clip", 0.5))
#         optimizer.step()
#         total_loss += loss.item()
#         if isinstance(cls_loss, torch.Tensor):
#             cls_total_loss += cls_loss.item()
#     return total_loss / (batch_idx + 1)