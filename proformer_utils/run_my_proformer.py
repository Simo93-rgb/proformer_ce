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


def main(opt: Optional[Dict[str, Any]] = None, load_vocab: bool = False) -> Tuple[
    float, float, Dict[int, float], int, Dict[int, float], Optional[Dict[str, float]]]:
    """
    Funzione principale per l'addestramento e la valutazione del modello Proformer.

    Args:
        opt: Parametri di configurazione (se None, vengono estratti dalla riga di comando)
        load_vocab: Se True, carica il vocabolario da file

    Returns:
        Tuple con: (miglior_loss_train, miglior_loss_valid, migliori_accuratezze_valid,
                   miglior_epoca, accuratezze_test, metriche_classificazione_test)
    """
    # ====== 1. CONFIGURAZIONE INIZIALE ======
    random.seed(datetime.now().timestamp())

    # Carica configurazione se non fornita
    if opt is None:
        print("-- PARSING CMD ARGS --")
        opt = parse_params(bpi_params)
    print(opt)

    # ====== 2. PREPARAZIONE DATI ======
    loader = _initialize_dataloader(opt)

    # ====== 3. CREAZIONE DEL MODELLO ======
    model = _build_model(loader, opt)

    # ====== 4. CONFIGURAZIONE TRAINING ======
    optimizer, scheduler = _setup_optimizer_scheduler(model, opt)

    # Parametri per early stopping e tracciamento del miglior modello
    best_val_acc = -float('inf')
    best_train_loss, best_valid_loss, best_valid_accs, best_epoch = 0.0, 0.0, None, 0
    test_accs, test_cls_metrics = None, None
    patience = opt.get("early_stopping_patience", 15)
    min_delta = opt.get("early_stopping_min_delta", 0.0001)
    counter = 0

    # ====== 5. LOOP DI TRAINING ======
    total_start_time = time.time()

    for epoch in range(1, opt["epochs"] + 1):
        # 5.1 Addestramento per un'epoca
        epoch_start_time = time.time()
        train_loss = train(model, opt, loader, optimizer)

        # 5.2 Valutazione su validation set
        valid_loss, valid_accs, valid_cls_metrics = evaluate(model, loader.valid_data, loader, opt)
        valid_ppl = math.exp(valid_loss)

        # 5.3 Salvataggio degli hidden states
        save_hidden_states(model.last_hidden_states, f"{MODELS_DIR}/hidden_states.csv")

        # 5.4 Stampa dei risultati periodica
        elapsed = time.time() - epoch_start_time
        _print_epoch_results(epoch, elapsed, valid_loss, valid_ppl, valid_accs, valid_cls_metrics)

        # 5.5 Controllo miglioramento e salvataggio miglior modello
        if valid_accs[1] > best_val_acc + min_delta:
            counter = 0
            best_val_acc = valid_accs[1]
            best_train_loss, best_valid_loss, best_epoch, best_valid_accs = train_loss, valid_loss, epoch, valid_accs

            # Valutazione sul test set quando troviamo un nuovo miglior modello
            test_loss, test_accs, test_cls_metrics = evaluate(model, loader.test_data, loader, opt, "test")
            test_ppl = math.exp(test_loss)
            print(f"| Performance on test: Test perplexity: {test_ppl:5.2f} | "
                  f"test acc@1: {test_accs[1]:.4f} | test acc@3: {test_accs[3]:.4f}" + " " * 21 + "|")
            print("-" * 104)

            # Salvataggio del miglior modello
            torch.save(model, f"{MODELS_DIR}/proformer-base.bin")
        else:
            # Gestione early stopping
            counter += 1
            if epoch % 10 == 0:
                print(f"| No improvement for {counter} epochs. Best acc@1: {best_val_acc:.4f} |")

        # 5.6 Early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs without improvement")
            break

        # 5.7 Aggiornamento scheduler
        scheduler.step()

    # ====== 6. VALUTAZIONE FINALE ======
    total_elapsed = time.time() - total_start_time
    print(f"\nTempo totale di training: {total_elapsed:.2f} secondi")

    print("\n" + "=" * 50)
    print("VALUTAZIONE FINALE SUL TEST SET:")
    final_test_loss, final_test_accs, final_test_cls_metrics = evaluate(model, loader.test_data, loader, opt, "test")
    _print_test_results(final_test_loss, final_test_accs, final_test_cls_metrics)

    return best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics


def _initialize_dataloader(opt):
    """Inizializza e configura il dataloader."""
    loader = Dataloader(filename=opt["dataset"], opt=opt)
    loader.get_dataset(num_test_ex=opt["test_split_size"], debugging=True)
    print("Esempio sequenza test:", loader.test_data[0])

    # Salva il vocabolario per usi futuri
    with open(f"{MODELS_DIR}/vocab.pkl", "wb") as f:
        pickle.dump(loader.vocab, f)

    return loader


def _build_model(loader, opt):
    """Costruisce il modello con o senza tassonomia."""
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

    return model


def _setup_optimizer_scheduler(model, opt):
    """Configura l'ottimizzatore e lo scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"], weight_decay=opt.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=opt["gamma_scheduler"])
    return optimizer, scheduler


def _print_epoch_results(epoch, elapsed, valid_loss, valid_ppl, valid_accs, valid_cls_metrics):
    """Stampa i risultati dell'epoca corrente."""
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
                  f"Recall: {valid_cls_metrics.get('recall', 0):.4f} |"
                  f"MCC: {valid_cls_metrics.get('mcc', 0):.4f} |")


def _print_test_results(test_loss, test_accs, test_cls_metrics):
    """Stampa i risultati finali sul test set."""
    test_ppl = math.exp(test_loss)
    print(f"| Test perplexity: {test_ppl:5.2f} | test acc@1: {test_accs[1]:.4f} | test acc@3: {test_accs[3]:.4f} |")
    if test_cls_metrics:
        print(f"| Classification metrics: Accuracy: {test_cls_metrics.get('accuracy', 0):.4f} | "
              f"F1: {test_cls_metrics.get('f1', 0):.4f} | "
              f"Precision: {test_cls_metrics.get('precision', 0):.4f} | "
              f"Recall: {test_cls_metrics.get('recall', 0):.4f} |"
              f"MCC: {test_cls_metrics.get('mcc', 0):.4f} |")
    print("=" * 50)

