import time
import math
import random
import argparse
import torch
import torch.nn.functional as F
from proformer.dataloader import Dataloader
from proformer import TransformerModel
from proformer.params import bpi_params
from taxonomy import TaxonomyEmbedding
import pickle
from datetime import datetime
import pandas as pd
import numpy as np
import os

"""
This module implements training, evaluation, and execution of the Proformer model
for process mining and prediction tasks. It includes functionality for hyperparameter
configuration, model training with transformer architecture, and performance evaluation.
"""


def parse_params(params):
    """
    Parse command-line arguments for Proformer configuration.
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    parser = argparse.ArgumentParser()

    # Aggiorna gli argomenti con i valori da bpi_params
    parser.add_argument("--device", type=str, default=params.get("device", "cuda:0"))
    parser.add_argument("--test_split_size", type=int, default=params["test_split_size"],
                        help="Number of examples to use for valid and test")
    parser.add_argument("--pad", action="store_true", default=params["pad"])
    parser.add_argument("--bptt", type=int, default=params["bptt"], help="Max len of sequences")
    parser.add_argument("--split_actions", action="store_true", default=params["split_actions"],
                        help="Splits multiple actions if in one (uses .split('_se_'))")
    parser.add_argument("--batch_size", type=int, default=params["batch_size"], help="Regulates the batch size")
    parser.add_argument("--pos_enc_dropout", type=float, default=params["pos_enc_dropout"],
                        help="Regulates dropout in pe")
    parser.add_argument("--d_model", type=int, default=params["d_model"])
    parser.add_argument("--nhead", type=int, default=params["nhead"])
    parser.add_argument("--nlayers", type=int, default=params["nlayers"])
    parser.add_argument("--dropout", type=float, default=params["dropout"])
    parser.add_argument("--d_hid", type=int, default=params["d_hid"])
    parser.add_argument("--epochs", type=int, default=params["epochs"])
    parser.add_argument("--lr", type=float, default=params["lr"])
    parser.add_argument("--gamma_scheduler", type=float, default=params["gamma_scheduler"])
    parser.add_argument("--use_l2_data", action="store_true", default=params["use_l2_data"],
                        help="Uses data from level 2 dataset")
    parser.add_argument("--use_taxonomy", action="store_true", default=params["use_taxonomy"],
                        help="Introduces weights based on a taxonomy of the tokens")
    parser.add_argument("--use_pe", action="store_true", default=params["use_pe"])
    parser.add_argument("--taxonomy_emb_type", type=str, default=params["taxonomy_emb_type"])
    parser.add_argument("--taxonomy_emb_size", type=int, default=params["taxonomy_emb_size"])
    parser.add_argument("--weight_decay", type=float, default=params["weight_decay"])  # Added weight decay
    parser.add_argument("--gradient_clip", type=float, default=params["gradient_clip"],
                        help="Add gradient clipping for stability")  # Gradient clipping

    # Early stopping parameters
    parser.add_argument("--early_stopping_patience", type=int, default=params['early_stopping_patience'],
                        help="Stop after this many epochs without improvement")
    parser.add_argument("--early_stopping_min_delta", type=float, default=params['early_stopping_min_delta'],
                        help="Minimum change to count as improvement")

    args = parser.parse_args()
    opt = vars(args)

    # Gestione file system
    data_path = os.getenv('DATA_PATH', './data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    return opt


def get_ranked_metrics(accs, out, t):
    """
    Calculate ranked metrics (accuracy@k) for model predictions.

    Args:
        accs (dict): Dictionary with k values as keys and accumulated accuracies as values
        out (torch.Tensor): Model output logits
        t (torch.Tensor): Target labels

    Returns:
        dict: Updated accuracies dictionary with new values
    """
    ks = list(accs.keys())
    # Get top-k predictions with highest probability
    out = torch.softmax(out, dim=1).topk(max(ks), dim=1).indices

    # Create a tensor of boolean masks indicating if target appears in top-k predictions
    all = []
    for i, el in enumerate(out[:,:max(ks)]):
        all.append(torch.isin(el, t[i]))

    all = torch.vstack(all)
    # Calculate accuracy for each k value
    for k in ks:
        accs[k] += all[:,:k].int().sum() / t.size(0)

    return accs


def train(model, opt, loader, optimizer):
    batch = 0
    model.train()
    total_loss = 0.
    cls_total_loss = 0.

    for batch, i in enumerate(range(0, loader.train_data.size(0) - 1, opt["bptt"])):
        # Get batch data and targets
        data, targets = loader.get_batch(loader.train_data, i)
        attn_mask = model.create_masked_attention_matrix(data.size(0)).to(opt["device"])

        # Forward pass
        output = model(data, attn_mask)
        output_flat = output.view(-1, model.ntokens)

        # Create mask to ignore padding tokens (1 and 8)
        pad_mask = (targets != 1) & (targets != 8)
        targets = targets[pad_mask]
        output_flat = output_flat[pad_mask, :]

        weights = torch.ones(model.ntokens).to(opt["device"])

        # Loss per la predizione della sequenza
        seq_loss = F.cross_entropy(output_flat, targets, weight=weights)

        # Loss per la classificazione
        cls_loss = 0.0
        if hasattr(model, 'cls_logits') and model.cls_logits.numel() > 0:
            batch_labels = loader.get_batch_labels(i // opt["bptt"])
            if batch_labels.size(0) == model.cls_logits.size(0):
                cls_loss = F.binary_cross_entropy_with_logits(model.cls_logits, batch_labels)
                cls_loss = cls_loss * 0.5  # Peso della loss di classificazione

        # Loss combinata
        loss = seq_loss + cls_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if isinstance(cls_loss, torch.Tensor):
            cls_total_loss += cls_loss.item()

    return total_loss / (batch + 1)


def evaluate(model, eval_data, loader, opt):
    model.eval()
    total_loss = 0.
    cls_total_loss = 0.
    accs = {1: 0., 3: 0., 5: 0.}

    # Metriche per la classificazione
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    cls_samples = 0

    with torch.no_grad():
        for batch, i in enumerate(range(0, eval_data.size(0) - 1, opt["bptt"])):
            data, targets = loader.get_batch(eval_data, i)
            attn_mask = model.create_masked_attention_matrix(data.size(0)).to(opt["device"])

            seq_len = data.size(0)
            output = model(data, attn_mask)
            output_flat = output.view(-1, model.ntokens)

            # Filtra i token di padding
            pad_mask = (targets != 1) & (targets != 8)
            targets = targets[pad_mask]
            output_flat = output_flat[pad_mask, :]

            # Calcola la loss di sequenza
            total_loss += seq_len * F.cross_entropy(output_flat, targets).item()
            accs = get_ranked_metrics(accs, output_flat, targets)

            # Calcola le metriche di classificazione
            if hasattr(model, 'cls_logits') and model.cls_logits.numel() > 0 and model.cls_logits.dim() > 0:
                batch_labels = loader.get_batch_labels(i // opt["bptt"])
                if batch_labels.size(0) == model.cls_logits.size(0):
                    cls_loss = F.binary_cross_entropy_with_logits(model.cls_logits, batch_labels)
                    cls_total_loss += cls_loss.item()

                    # Calcola la matrice di confusione
                    cls_preds = (torch.sigmoid(model.cls_logits) > 0.5).float()
                    true_positives += ((cls_preds == 1) & (batch_labels == 1)).sum().item()
                    false_positives += ((cls_preds == 1) & (batch_labels == 0)).sum().item()
                    true_negatives += ((cls_preds == 0) & (batch_labels == 0)).sum().item()
                    false_negatives += ((cls_preds == 0) & (batch_labels == 1)).sum().item()
                    cls_samples += batch_labels.size(0)

        # Normalizza le metriche
        for k in accs:
            accs[k] = accs[k] / (batch + 1)
        loss = total_loss / (len(eval_data) - 1)

        # Calcola le metriche di classificazione finali
        cls_metrics = {}
        if cls_samples > 0:
            precision = true_positives / (true_positives + false_positives) if (
                                                                                           true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (
                                                                                        true_positives + false_negatives) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_positives + true_negatives) / cls_samples

            cls_metrics = {
                'loss': cls_total_loss / (batch + 1),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }

    return loss, accs, cls_metrics


def main(opt=None, load_vocab=False):
    """
    Main function to train and evaluate the Proformer model.

    Args:
        opt (dict, optional): Configuration parameters. If None, params are parsed from command line

    Returns:
        tuple: (best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs)
    """
    random.seed(datetime.now().timestamp())

    if opt is None:
        print("-- PARSING CMD ARGS --")
        opt = parse_params(bpi_params)
    print(opt)

    # -- Add optional params here --
    #
    # opt["d_model"] = 32
    # opt["d_hid"] = 32
    # opt["nlayers"] = 1
    # opt["nhead"] = 1
    # opt["use_l2_data"] = False
    # opt["test_split_size"] = 7000
    # opt["epochs"] = 100
    # opt["use_taxonomy"] = False
    # opt["use_pe"] = False
    # opt["bptt"] = 175
    # opt["lr"] = 3e-3
    # opt["taxonomy_emb_type"] = "laplacian"
    # opt["taxonomy_emb_size"] = 8
    #
    # ------------------------------


    # Initialize data loader and create dataset splits
    loader = Dataloader(filename=opt["dataset"], opt=opt)
    loader.get_dataset(num_test_ex=opt["test_split_size"])

    # Save the vocabulary
    with open("models/vocab.pkl", "wb") as f:
        pickle.dump(loader.vocab, f)

    if opt["use_taxonomy"]:
        tax = TaxonomyEmbedding(
            vocab=loader.vocab,
            filename=opt["taxonomy"],
            opt=opt
        )
        # Initialize model with taxonomy
        model = TransformerModel(len(loader.vocab), opt, taxonomy=tax.embs).to(opt["device"])
    else:
        # Initialize model
        vocab_dim = len(loader.vocab)
        model = TransformerModel(loader.vocab, vocab_dim, opt).to(opt["device"])

    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=opt["gamma_scheduler"])
    best_val_acc = -float('inf')

    # Initialize early stopping variables
    patience = opt.get("early_stopping_patience", 15)  # Default to 15 if not specified
    min_delta = opt.get("early_stopping_min_delta", 0.0001) # Default to 0.0001 if not specified
    counter = 0
    test_cls_metrics = None
    # Training loop
    for epoch in range(1, opt["epochs"]+1):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss = train(model, opt, loader, optimizer)
        # Evaluate on validation set
        valid_loss, valid_accs, valid_cls_metrics = evaluate(model, loader.valid_data, loader, opt)
        valid_ppl = math.exp(valid_loss)

        # Extract and save hidden states (trace embeddings)
        print ("last_hidden_ststes - size: ", model.last_hidden_states.shape)
        print ("last_hidden_ststes - H_size: ", model.last_hidden_states.shape[0])
        print ("last_hidden_ststes - V_size: ", model.last_hidden_states.shape[1])
        print (model.last_hidden_states)

        # Move tensor from GPU to CPU and save as CSV
        mytensor = model.last_hidden_states.cpu()
        DF = pd.DataFrame(np.reshape(mytensor, (model.last_hidden_states.shape[0]*model.last_hidden_states.shape[1], -1)))
        DF = DF.transpose()
        DF.to_csv("models/hidden_states.csv")

        elapsed = time.time() - epoch_start_time

        # Print progress every 10 epochs
        if (epoch % 10) == 0:
            print('-' * 104)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {valid_loss:5.2f} | valid ppl {valid_ppl:7.2f} | '
                  f'acc@1 {valid_accs[1]:.4f} | '
                  f'acc@3 {valid_accs[3]:.4f} |')
            print('-' * 104)
            # Aggiungi questa parte quando stampi i risultati
            if valid_cls_metrics:
                print(f"| Classification metrics: Accuracy: {valid_cls_metrics['accuracy']:.4f} | "
                      f"F1: {valid_cls_metrics['f1']:.4f} | "
                      f"Precision: {valid_cls_metrics['precision']:.4f} | "
                      f"Recall: {valid_cls_metrics['recall']:.4f} |")

        # Save best model based on validation accuracy
        if valid_accs[1] > best_val_acc + min_delta:
            # Model improved
            counter = 0
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_valid_accs = valid_accs
            best_val_acc = valid_accs[1]

            # Evaluate on test set when we find a better model
            test_loss, test_accs, test_cls_metrics = evaluate(model, loader.test_data, loader, opt)
            test_ppl = math.exp(test_loss)
            print(f"| Performance on test: Test ppl: {test_ppl:5.2f} | "
                  f"test acc@1: {test_accs[1]:.4f} | test acc@3: {test_accs[3]:.4f}" + " " * 21 + "|")
            print("-"*104)

            # Save the best model checkpoint
            torch.save(model, "models/proformer-base.bin")
        else:
            # No improvement, increment counter
            counter += 1
            if epoch % 10 == 0:
                print(f"| No improvement for {counter} epochs. Best acc@1: {best_val_acc:.4f} |")
                # Check for early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs without improvement")
            break
        # Update learning rate
        scheduler.step()
    print("\n" + "=" * 50)
    print("VALUTAZIONE FINALE SUL SET DI TEST:")
    final_test_loss, final_test_accs, final_test_cls_metrics = evaluate(model, loader.test_data, loader, opt)
    final_test_ppl = math.exp(final_test_loss)
    print(
        f"| Test ppl: {final_test_ppl:5.2f} | test acc@1: {final_test_accs[1]:.4f} | test acc@3: {final_test_accs[3]:.4f} |")
    if final_test_cls_metrics:
        print(f"| Classification metrics: Accuracy: {final_test_cls_metrics['accuracy']:.4f} | "
              f"F1: {final_test_cls_metrics['f1']:.4f} | "
              f"Precision: {final_test_cls_metrics['precision']:.4f} | "
              f"Recall: {final_test_cls_metrics['recall']:.4f} |")
    print("=" * 50)

    return best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics

if __name__ == "__main__":
    opt = parse_params(bpi_params)
    # opt["dataset"] = "data/aggregated_case_tuple.csv"
    # opt["dataset"] = "data/aggregated_case_detailed.csv"
    opt["dataset"] = "data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv"

    best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics = main(opt=opt)
    print(f"Best epoch: {best_epoch} \t loss: {best_valid_loss} \t best accs: {best_valid_accs}")
    if test_cls_metrics:
        print(f"Test classification metrics: Accuracy: {test_cls_metrics['accuracy']:.4f}, "
              f"F1: {test_cls_metrics['f1']:.4f}, "
              f"Precision: {test_cls_metrics['precision']:.4f}, "
              f"Recall: {test_cls_metrics['recall']:.4f}")