import math
import torch

import torch
import torch.nn.functional as F
import pickle
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from proformer_utils.dataloader import Dataloader
from proformer_utils.params import bpi_params, parse_params
from proformer_utils.visualization import plot_attention_maps
from proformer_utils.proformer import TransformerModel
from taxonomy import TaxonomyEmbedding
from config import DATA_DIR, MODELS_DIR
from proformer_utils.tensor_utils import create_attention_mask, filter_padding, get_ranked_metrics
from proformer_utils.metrics import compute_classification_metrics





def calculate_sequence_loss(model: torch.nn.Module, data: torch.Tensor, 
                           targets: torch.Tensor, attn_mask: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Calcola la loss per la predizione della sequenza.
    
    Args:
        model: Il modello da valutare
        data: I dati di input
        targets: I target attesi
        attn_mask: La maschera di attenzione
        
    Returns:
        Tuple con loss, output flattened e targets filtrati
    """
    seq_len = data.size(0)
    output = model(data, attn_mask)
    output_flat = output.view(-1, model.ntokens)
    targets_filtered, output_flat_filtered = filter_padding(targets, output_flat)
    loss = seq_len * F.cross_entropy(output_flat_filtered, targets_filtered).item()
    
    return loss, output_flat_filtered, targets_filtered

def calculate_ranked_metrics(output_flat: torch.Tensor, targets: torch.Tensor, accs: Dict[int, float]) -> Dict[int, float]:
    """
    Calcola le metriche di accuratezza ranked (top-k).
    
    Args:
        output_flat: Output del modello (logits)
        targets: Target attesi
        accs: Dizionario con le chiavi k e valori inizializzati
        
    Returns:
        Dizionario aggiornato con le accuratezze
    """
    return get_ranked_metrics(accs, output_flat, targets)

def calculate_classification_metrics(model: torch.nn.Module, batch_labels: torch.Tensor) -> Tuple[float, int, int, int, int, int]:
    """
    Calcola le metriche di classificazione binaria.
    
    Args:
        model: Il modello con i cls_logits
        batch_labels: Le label vere del batch
        
    Returns:
        Tuple con loss, true positives, false positives, true negatives, false negatives, numero di campioni
    """
    if not hasattr(model, 'cls_logits') or model.cls_logits.numel() == 0 or model.cls_logits.dim() == 0:
        return 0.0, 0, 0, 0, 0, 0
        
    if batch_labels.size(0) != model.cls_logits.size(0):
        return 0.0, 0, 0, 0, 0, 0
        
    cls_loss = F.binary_cross_entropy_with_logits(model.cls_logits, batch_labels)
    cls_preds = (torch.sigmoid(model.cls_logits) > 0.5).float()
    
    tp = ((cls_preds == 1) & (batch_labels == 1)).sum().item()
    fp = ((cls_preds == 1) & (batch_labels == 0)).sum().item()
    tn = ((cls_preds == 0) & (batch_labels == 0)).sum().item()
    fn = ((cls_preds == 0) & (batch_labels == 1)).sum().item()
    
    return cls_loss.item(), tp, fp, tn, fn, batch_labels.size(0)

def evaluate(model: torch.nn.Module, eval_data: torch.Tensor, loader: Dataloader, 
            opt: Dict[str, Any]) -> Tuple[float, Dict[int, float], Dict[str, float]]:
    """
    Evaluate the model on validation or test data.

    Args:
        model (torch.nn.Module): The model to evaluate.
        eval_data (torch.Tensor): Evaluation data.
        loader (Dataloader): Data loader.
        opt (dict): Configuration options.

    Returns:
        tuple: (loss, ranked accuracies, classification metrics)
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
    
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, opt["bptt"]):
            data, targets = loader.get_batch(eval_data, i)
            attn_mask = create_attention_mask(data.size(0), opt["device"])
            
            # Calcola la loss di sequenza
            seq_loss, output_flat, targets_filtered = calculate_sequence_loss(model, data, targets, attn_mask)
            total_loss += seq_loss
            
            # Calcola le metriche di accuratezza ranked
            accs = calculate_ranked_metrics(output_flat, targets_filtered, accs)
            
            # Calcola le metriche di classificazione
            batch_labels = loader.get_batch_labels(i // opt["bptt"])
            cls_loss, tp, fp, tn, fn, samples = calculate_classification_metrics(model, batch_labels)
            
            cls_total_loss += cls_loss
            true_positives += tp
            false_positives += fp
            true_negatives += tn
            false_negatives += fn
            cls_samples += samples
            num_batches += 1
        
        # Normalizza i risultati
        for k in accs:
            accs[k] = accs[k] / num_batches if num_batches > 0 else 0
        
        loss = total_loss / (len(eval_data) - 1) if len(eval_data) > 1 else 0
        
        # Calcola le metriche finali di classificazione
        cls_metrics = {}
        if cls_samples > 0:
            preds = torch.tensor([1]*true_positives + [1]*false_positives + [0]*true_negatives + [0]*false_negatives)
            labels = torch.tensor([1]*true_positives + [0]*false_positives + [0]*true_negatives + [1]*false_negatives)
            cls_metrics = compute_classification_metrics(preds=preds, labels=labels)
            cls_metrics['loss'] = cls_total_loss / num_batches if num_batches > 0 else 0
            
    return loss, accs, cls_metrics