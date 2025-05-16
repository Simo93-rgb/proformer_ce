import time
import math
import random
import argparse
import torch
import torch.nn.functional as F
import pickle
from datetime import datetime
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from proformer_utils.dataloader import Dataloader
from proformer_utils.params import bpi_params
from proformer_utils.proformer import TransformerModel
from taxonomy import TaxonomyEmbedding
from config import DATA_DIR, MODELS_DIR

"""
This module implements training, evaluation, and execution of the Proformer model
for process mining and prediction tasks. It includes functionality for hyperparameter
configuration, model training with transformer architecture, and performance evaluation.
"""

def parse_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse command-line arguments for Proformer configuration.

    Args:
        params (dict): Default parameters.

    Returns:
        dict: Dictionary containing all configuration parameters.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=params.get("device", "cuda:0"))
    parser.add_argument("--test_split_size", type=int, default=params["test_split_size"])
    parser.add_argument("--pad", action="store_true", default=params["pad"])
    parser.add_argument("--bptt", type=int, default=params["bptt"])
    parser.add_argument("--split_actions", action="store_true", default=params["split_actions"])
    parser.add_argument("--batch_size", type=int, default=params["batch_size"])
    parser.add_argument("--pos_enc_dropout", type=float, default=params["pos_enc_dropout"])
    parser.add_argument("--d_model", type=int, default=params["d_model"])
    parser.add_argument("--nhead", type=int, default=params["nhead"])
    parser.add_argument("--nlayers", type=int, default=params["nlayers"])
    parser.add_argument("--dropout", type=float, default=params["dropout"])
    parser.add_argument("--d_hid", type=int, default=params["d_hid"])
    parser.add_argument("--epochs", type=int, default=params["epochs"])
    parser.add_argument("--lr", type=float, default=params["lr"])
    parser.add_argument("--gamma_scheduler", type=float, default=params["gamma_scheduler"])
    parser.add_argument("--use_l2_data", action="store_true", default=params["use_l2_data"])
    parser.add_argument("--use_taxonomy", action="store_true", default=params["use_taxonomy"])
    parser.add_argument("--use_pe", action="store_true", default=params["use_pe"])
    parser.add_argument("--taxonomy_emb_type", type=str, default=params["taxonomy_emb_type"])
    parser.add_argument("--taxonomy_emb_size", type=int, default=params["taxonomy_emb_size"])
    parser.add_argument("--weight_decay", type=float, default=params["weight_decay"])
    parser.add_argument("--gradient_clip", type=float, default=params["gradient_clip"])
    parser.add_argument("--early_stopping_patience", type=int, default=params['early_stopping_patience'])
    parser.add_argument("--early_stopping_min_delta", type=float, default=params['early_stopping_min_delta'])
    args = parser.parse_args()
    opt = vars(args)

    return opt

def create_attention_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create a causal attention mask for transformer input.

    Args:
        seq_len (int): Sequence length.
        device (torch.device): Device to allocate the mask.

    Returns:
        torch.Tensor: Attention mask tensor.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device)) == 1
    mask = mask.float().masked_fill(~mask, float('-inf')).masked_fill(mask, 0.0)
    return mask

def filter_padding(targets: torch.Tensor, output_flat: torch.Tensor, pad_tokens: Tuple[int, ...] = (1, 8)) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove padding tokens from targets and output.

    Args:
        targets (torch.Tensor): Target tensor.
        output_flat (torch.Tensor): Output tensor.
        pad_tokens (tuple): Padding token values.

    Returns:
        tuple: Filtered targets and outputs.
    """
    pad_mask = ~torch.isin(targets, torch.tensor(pad_tokens, device=targets.device))
    return targets[pad_mask], output_flat[pad_mask, :]

def compute_classification_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute precision, recall, f1, and accuracy for binary classification.

    Args:
        preds (torch.Tensor): Predicted labels (0/1).
        labels (torch.Tensor): True labels (0/1).

    Returns:
        dict: Dictionary with precision, recall, f1, accuracy.
    """
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    total = labels.size(0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0
    return dict(precision=precision, recall=recall, f1=f1, accuracy=accuracy)

def save_hidden_states(hidden_states: torch.Tensor, path: str = "models/hidden_states.csv") -> None:
    """
    Save hidden states as a CSV file.

    Args:
        hidden_states (torch.Tensor): Hidden states tensor.
        path (str): Output CSV path.
    """
    hs_cpu = hidden_states.cpu()
    df = pd.DataFrame(hs_cpu.reshape(-1, hs_cpu.shape[-1]))
    df.to_csv(path, index=False)

def get_ranked_metrics(accs: Dict[int, float], out: torch.Tensor, t: torch.Tensor) -> Dict[int, float]:
    """
    Calculate ranked metrics (accuracy@k) for model predictions.

    Args:
        accs (dict): Dictionary with k values as keys and accumulated accuracies as values.
        out (torch.Tensor): Model output logits.
        t (torch.Tensor): Target labels.

    Returns:
        dict: Updated accuracies dictionary with new values.
    """
    ks = list(accs.keys())
    out = torch.softmax(out, dim=1).topk(max(ks), dim=1).indices
    all = []
    for i, el in enumerate(out[:, :max(ks)]):
        all.append(torch.isin(el, t[i]))
    all = torch.vstack(all)
    for k in ks:
        accs[k] += all[:, :k].int().sum() / t.size(0)
    return accs

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
    for batch, i in enumerate(range(0, loader.train_data.size(0) - 1, opt["bptt"])):
        data, targets = loader.get_batch(loader.train_data, i)
        attn_mask = create_attention_mask(data.size(0), opt["device"])
        output = model(data, attn_mask)
        output_flat = output.view(-1, model.ntokens)
        targets, output_flat = filter_padding(targets, output_flat)
        seq_loss = F.cross_entropy(output_flat, targets)
        cls_loss = 0.0
        if hasattr(model, 'cls_logits') and model.cls_logits.numel() > 0:
            batch_labels = loader.get_batch_labels(i // opt["bptt"])
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
    return total_loss / (batch + 1)

def evaluate(model: torch.nn.Module, eval_data: torch.Tensor, loader: Dataloader, opt: Dict[str, Any]) -> Tuple[float, Dict[int, float], Dict[str, float]]:
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
    with torch.no_grad():
        for batch, i in enumerate(range(0, eval_data.size(0) - 1, opt["bptt"])):
            data, targets = loader.get_batch(eval_data, i)
            attn_mask = create_attention_mask(data.size(0), opt["device"])
            seq_len = data.size(0)
            output = model(data, attn_mask)
            output_flat = output.view(-1, model.ntokens)
            targets, output_flat = filter_padding(targets, output_flat)
            total_loss += seq_len * F.cross_entropy(output_flat, targets).item()
            accs = get_ranked_metrics(accs, output_flat, targets)
            if hasattr(model, 'cls_logits') and model.cls_logits.numel() > 0 and model.cls_logits.dim() > 0:
                batch_labels = loader.get_batch_labels(i // opt["bptt"])
                if batch_labels.size(0) == model.cls_logits.size(0):
                    cls_loss = F.binary_cross_entropy_with_logits(model.cls_logits, batch_labels)
                    cls_total_loss += cls_loss.item()
                    cls_preds = (torch.sigmoid(model.cls_logits) > 0.5).float()
                    true_positives += ((cls_preds == 1) & (batch_labels == 1)).sum().item()
                    false_positives += ((cls_preds == 1) & (batch_labels == 0)).sum().item()
                    true_negatives += ((cls_preds == 0) & (batch_labels == 0)).sum().item()
                    false_negatives += ((cls_preds == 0) & (batch_labels == 1)).sum().item()
                    cls_samples += batch_labels.size(0)
        for k in accs:
            accs[k] = accs[k] / (batch + 1)
        loss = total_loss / (len(eval_data) - 1)
        cls_metrics = {}
        if cls_samples > 0:
            cls_metrics = compute_classification_metrics(
                preds=(torch.tensor([1]*true_positives + [0]*false_positives + [0]*true_negatives + [1]*false_negatives)),
                labels=(torch.tensor([1]*true_positives + [0]*false_positives + [0]*true_negatives + [1]*false_negatives))
            )
            cls_metrics['loss'] = cls_total_loss / (batch + 1)
    return loss, accs, cls_metrics

def main(opt: Optional[Dict[str, Any]] = None, load_vocab: bool = False) -> Tuple[float, float, Dict[int, float], int, Dict[int, float], Optional[Dict[str, float]]]:
    """
    Main function to train and evaluate the Proformer model.

    Args:
        opt (dict, optional): Configuration parameters. If None, params are parsed from command line.
        load_vocab (bool): If True, load vocabulary from file.

    Returns:
        tuple: (best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics)
    """
    random.seed(datetime.now().timestamp())
    if opt is None:
        print("-- PARSING CMD ARGS --")
        opt = parse_params(bpi_params)
    print(opt)
    loader = Dataloader(filename=opt["dataset"], opt=opt)
    loader.get_dataset(num_test_ex=opt["test_split_size"])
    with open(f"{MODELS_DIR}/vocab.pkl", "wb") as f:
        pickle.dump(loader.vocab, f)
    if opt["use_taxonomy"]:
        tax = TaxonomyEmbedding(
            vocab=loader.vocab,
            filename=opt["taxonomy"],
            opt=opt
        )
        model = TransformerModel(loader.vocab, len(loader.vocab), opt=opt, taxonomy=tax.embs).to(opt["device"])
    else:
        vocab_dim = len(loader.vocab)
        model = TransformerModel(loader.vocab, vocab_dim, opt).to(opt["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"], weight_decay=opt.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=opt["gamma_scheduler"])
    best_val_acc = -float('inf')
    patience = opt.get("early_stopping_patience", 15)
    min_delta = opt.get("early_stopping_min_delta", 0.0001)
    counter = 0
    test_cls_metrics = None
    for epoch in range(1, opt["epochs"]+1):
        epoch_start_time = time.time()
        train_loss = train(model, opt, loader, optimizer)
        valid_loss, valid_accs, valid_cls_metrics = evaluate(model, loader.valid_data, loader, opt)
        valid_ppl = math.exp(valid_loss)
        save_hidden_states(model.last_hidden_states, "../models/hidden_states.csv")
        elapsed = time.time() - epoch_start_time
        if (epoch % 10) == 0:
            print('-' * 104)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {valid_loss:5.2f} | valid ppl {valid_ppl:7.2f} | '
                  f'acc@1 {valid_accs[1]:.4f} | '
                  f'acc@3 {valid_accs[3]:.4f} |')
            print('-' * 104)
            if valid_cls_metrics:
                print(f"| Classification metrics: Accuracy: {valid_cls_metrics.get('accuracy', 0):.4f} | "
                      f"F1: {valid_cls_metrics.get('f1', 0):.4f} | "
                      f"Precision: {valid_cls_metrics.get('precision', 0):.4f} | "
                      f"Recall: {valid_cls_metrics.get('recall', 0):.4f} |")
        if valid_accs[1] > best_val_acc + min_delta:
            counter = 0
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_valid_accs = valid_accs
            best_val_acc = valid_accs[1]
            test_loss, test_accs, test_cls_metrics = evaluate(model, loader.test_data, loader, opt)
            test_ppl = math.exp(test_loss)
            print(f"| Performance on test: Test ppl: {test_ppl:5.2f} | "
                  f"test acc@1: {test_accs[1]:.4f} | test acc@3: {test_accs[3]:.4f}" + " " * 21 + "|")
            print("-"*104)
            torch.save(model, f"{MODELS_DIR}/proformer-base.bin")
        else:
            counter += 1
            if epoch % 10 == 0:
                print(f"| No improvement for {counter} epochs. Best acc@1: {best_val_acc:.4f} |")
        if counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs without improvement")
            break
        scheduler.step()
    print("\n" + "=" * 50)
    print("FINAL EVALUATION ON TEST SET:")
    final_test_loss, final_test_accs, final_test_cls_metrics = evaluate(model, loader.test_data, loader, opt)
    final_test_ppl = math.exp(final_test_loss)
    print(
        f"| Test ppl: {final_test_ppl:5.2f} | test acc@1: {final_test_accs[1]:.4f} | test acc@3: {final_test_accs[3]:.4f} |")
    if final_test_cls_metrics:
        print(f"| Classification metrics: Accuracy: {final_test_cls_metrics.get('accuracy', 0):.4f} | "
              f"F1: {final_test_cls_metrics.get('f1', 0):.4f} | "
              f"Precision: {final_test_cls_metrics.get('precision', 0):.4f} | "
              f"Recall: {final_test_cls_metrics.get('recall', 0):.4f} |")
    print("=" * 50)
    return best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics

if __name__ == "__main__":
    opt = parse_params(bpi_params)
    # opt["dataset"] = "data/aggregated_case_tuple.csv"
    # opt["dataset"] = "data/aggregated_case_detailed.csv"
    opt["dataset"] = f"{DATA_DIR}/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv"

    best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics = main(opt=opt)
    print(f"Best epoch: {best_epoch} \t loss: {best_valid_loss} \t best accs: {best_valid_accs}")
    if test_cls_metrics:
        print(f"Test classification metrics: Accuracy: {test_cls_metrics['accuracy']:.4f}, "
              f"F1: {test_cls_metrics['f1']:.4f}, "
              f"Precision: {test_cls_metrics['precision']:.4f}, "
              f"Recall: {test_cls_metrics['recall']:.4f}")