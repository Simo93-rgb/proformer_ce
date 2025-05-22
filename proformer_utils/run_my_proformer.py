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
from proformer_utils.params import bpi_params, parse_params
from proformer_utils.visualization import plot_attention_maps
from proformer_utils.proformer import TransformerModel
from taxonomy import TaxonomyEmbedding
from config import DATA_DIR, MODELS_DIR
from proformer_utils.evaluation import evaluate, calculate_classification_metrics
from proformer_utils.tensor_utils import filter_padding, save_hidden_states, create_attention_mask
from proformer_utils.training import train
from proformer_utils.visualization import save_attention_maps


"""
This module implements training, evaluation, and execution of the Proformer model
for process mining and prediction tasks. It includes functionality for hyperparameter
configuration, model training with transformer architecture, and performance evaluation.
"""


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
        save_hidden_states(model.last_hidden_states, f"{MODELS_DIR}/hidden_states.csv")
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

    # best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics = main(opt=opt)
    # print(f"Best epoch: {best_epoch} \t loss: {best_valid_loss} \t best accs: {best_valid_accs}")

    # Carica il modello migliore e salva le mappe di attenzione
    model = torch.load(f"{MODELS_DIR}/proformer-base.bin")
    loader = Dataloader(filename=opt["dataset"], opt=opt)
    loader.get_dataset(num_test_ex=opt["test_split_size"])

    # Salva le mappe di attenzione
    save_attention_maps(model, loader, opt)