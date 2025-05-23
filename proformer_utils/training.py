import torch
import pandas as pd
from typing import Tuple, Dict, Any
from proformer_utils.dataloader import Dataloader
from proformer_utils.tensor_utils import create_attention_mask, filter_padding
import torch.nn.functional as F


def train(model: torch.nn.Module, opt: Dict[str, Any], loader: Dataloader, optimizer: torch.optim.Optimizer) -> float:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        opt (dict): Configuration options.
        loader (Dataloader): Data loader.
        optimizer (torch.optim.Optimizer): Optimizer.

    Returns:
        float: Average training loss.
    """
    model.train()
    total_loss = 0.
    cls_total_loss = 0.
    for batch_idx, batch in enumerate(loader.train_data):
        data, targets = loader.get_batch_from_list(loader.train_data, batch_idx)
        attn_mask = create_attention_mask(data.shape[0], opt["device"])
        output = model(data, attn_mask)
        output_flat = output.view(-1, model.ntokens)
        targets, output_flat = filter_padding(targets, output_flat)
        seq_loss = F.cross_entropy(output_flat, targets)
        cls_loss = 0.0
        if hasattr(model, 'cls_logits') and model.cls_logits.numel() > 0:
            batch_labels = loader.get_batch_labels(batch_idx)
            if batch_labels.size(0) == model.cls_logits.size(0):
                cls_loss = F.binary_cross_entropy_with_logits(model.cls_logits, batch_labels) * 0.5
        loss = seq_loss + cls_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.get("gradient_clip", 0.5))
        optimizer.step()
        total_loss += loss.item()
        if isinstance(cls_loss, torch.Tensor):
            cls_total_loss += cls_loss.item()
    return total_loss / (batch_idx + 1)