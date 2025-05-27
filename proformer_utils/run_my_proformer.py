import time
import math
import random
import argparse
import torch
import torch.nn.functional as F
import pickle
from datetime import datetime
import pandas as pd
import os
from typing import Dict, Any, Tuple, Optional, List, Union
from proformer_utils.dataloader import Dataloader
from proformer_utils.params import bpi_params, parse_params
from proformer_utils.visualization import plot_attention_map, save_attention_maps
from proformer_utils.proformer import TransformerModel
from taxonomy import TaxonomyEmbedding
from config import DATA_DIR, MODELS_DIR
from proformer_utils.evaluation import evaluate, calculate_classification_metrics
from proformer_utils.tensor_utils import filter_padding, save_hidden_states, create_attention_mask
from proformer_utils.training import train


def initialize_dataloader(opt: Dict[str, Any]) -> Dataloader:
    """
    Initialize and configure the dataloader with dataset processing.

    Args:
        opt: Configuration options dictionary

    Returns:
        Configured Dataloader instance with processed datasets
    """
    try:
        loader = Dataloader(filename=opt["dataset"], opt=opt)
        loader.get_dataset(num_test_ex=opt["test_split_size"], debugging=True)

        # Save vocabulary for future use
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            with open(f"{MODELS_DIR}/vocab.pkl", "wb") as f:
                pickle.dump(loader.vocab, f)
        except IOError as e:
            print(f"Warning: Could not save vocabulary: {e}")

        return loader
    except Exception as e:
        print(f"Error initializing dataloader: {e}")
        raise


def build_model(loader: Dataloader, opt: Dict[str, Any]) -> TransformerModel:
    """
    Build the transformer model with optional taxonomy embeddings.

    Args:
        loader: Dataloader instance with vocabulary
        opt: Configuration options dictionary

    Returns:
        Initialized TransformerModel instance
    """
    try:
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
    except Exception as e:
        print(f"Error building model: {e}")
        raise


def setup_optimizer_scheduler(model: TransformerModel, opt: Dict[str, Any]) -> Tuple[
    torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Configure the optimizer and learning rate scheduler.

    Args:
        model: TransformerModel instance
        opt: Configuration options dictionary

    Returns:
        Tuple containing (optimizer, scheduler)
    """
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt["lr"],
            weight_decay=opt.get("weight_decay", 0.0)
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            1,
            gamma=opt["gamma_scheduler"]
        )
        return optimizer, scheduler
    except Exception as e:
        print(f"Error setting up optimizer and scheduler: {e}")
        raise


def print_epoch_results(epoch: int, elapsed: float, valid_loss: float,
                        valid_ppl: float, valid_accs: Dict[int, float],
                        valid_cls_metrics: Optional[Dict[str, float]]) -> None:
    """
    Print the training results for the current epoch.

    Args:
        epoch: Current epoch number
        elapsed: Time elapsed during epoch (seconds)
        valid_loss: Validation loss
        valid_ppl: Validation perplexity
        valid_accs: Dictionary of validation accuracies at different ranks
        valid_cls_metrics: Optional dictionary of classification metrics
    """
    try:
        if (epoch % 10) == 0:
            print('-' * 104)
            print(f'| End of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
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
    except Exception as e:
        print(f"Error printing epoch results: {e}")


def print_test_results(test_loss: float, test_accs: Dict[int, float],
                       test_cls_metrics: Optional[Dict[str, float]]) -> None:
    """
    Print the final test set evaluation results.

    Args:
        test_loss: Test loss
        test_accs: Dictionary of test accuracies at different ranks
        test_cls_metrics: Optional dictionary of classification metrics
    """
    try:
        test_ppl = math.exp(test_loss)
        print(f"| Test perplexity: {test_ppl:5.2f} | test acc@1: {test_accs[1]:.4f} | test acc@3: {test_accs[3]:.4f} |")
        if test_cls_metrics:
            print(f"| Classification metrics: Accuracy: {test_cls_metrics.get('accuracy', 0):.4f} | "
                  f"F1: {test_cls_metrics.get('f1', 0):.4f} | "
                  f"Precision: {test_cls_metrics.get('precision', 0):.4f} | "
                  f"Recall: {test_cls_metrics.get('recall', 0):.4f} |"
                  f"MCC: {test_cls_metrics.get('mcc', 0):.4f} |")
        print("=" * 50)
    except Exception as e:
        print(f"Error printing test results: {e}")


def save_best_model(model: TransformerModel, model_path: str = None) -> bool:
    """
    Save the model to disk.

    Args:
        model: TransformerModel instance to save
        model_path: Optional path where to save the model

    Returns:
        Boolean indicating success or failure
    """
    try:
        path = model_path or f"{MODELS_DIR}/proformer-base.bin"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model, path)
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


def run_training_epoch(model: TransformerModel, epoch: int, loader: Dataloader,
                       optimizer: torch.optim.Optimizer, opt: Dict[str, Any]) -> Tuple[
    float, float, Dict[int, float], Optional[Dict[str, float]]]:
    """
    Run a single training epoch and evaluate on validation set.

    Args:
        model: TransformerModel instance
        epoch: Current epoch number
        loader: Dataloader instance
        optimizer: Optimizer instance
        opt: Configuration options dictionary

    Returns:
        Tuple containing (train_loss, valid_loss, valid_accs, valid_cls_metrics)
    """
    try:
        # Train for one epoch
        epoch_start_time = time.time()
        train_loss = train(model, opt, loader, optimizer)

        # Evaluate on validation set
        valid_loss, valid_accs, valid_cls_metrics = evaluate(model, loader.valid_data, loader, opt)
        valid_ppl = math.exp(valid_loss)

        # Save hidden states for analysis
        if hasattr(model, 'last_hidden_states') and model.last_hidden_states is not None:
            save_hidden_states(model.last_hidden_states, f"{MODELS_DIR}/hidden_states.csv")

        # Print results periodically
        elapsed = time.time() - epoch_start_time
        print_epoch_results(epoch, elapsed, valid_loss, valid_ppl, valid_accs, valid_cls_metrics)

        return train_loss, valid_loss, valid_accs, valid_cls_metrics
    except Exception as e:
        print(f"Error during training epoch {epoch}: {e}")
        return float('inf'), float('inf'), {1: 0.0, 3: 0.0, 5: 0.0}, None


def main(opt: Optional[Dict[str, Any]] = None, load_vocab: bool = False) -> Tuple[
    float, float, Dict[int, float], int, Dict[int, float], Optional[Dict[str, float]]]:
    """
    Main function for training and evaluating the Proformer model.

    Args:
        opt: Configuration parameters (if None, extracted from command line)
        load_vocab: If True, loads vocabulary from file

    Returns:
        Tuple containing (best_train_loss, best_valid_loss, best_valid_accs,
                         best_epoch, test_accs, test_cls_metrics)
    """
    try:
        # ====== 1. INITIAL SETUP ======
        random.seed(datetime.now().timestamp())

        # Load configuration if not provided
        if opt is None:
            print("-- PARSING CMD ARGS --")
            opt = parse_params(bpi_params)
        print(opt)

        # ====== 2. DATA PREPARATION ======
        loader = initialize_dataloader(opt)

        # ====== 3. MODEL CREATION ======
        model = build_model(loader, opt)

        # ====== 4. TRAINING SETUP ======
        optimizer, scheduler = setup_optimizer_scheduler(model, opt)

        # Parameters for early stopping and best model tracking
        best_val_acc = -float('inf')
        best_train_loss, best_valid_loss, best_epoch = 0.0, 0.0, 0
        best_valid_accs, test_accs, test_cls_metrics = None, None, None
        patience = opt.get("early_stopping_patience", 15)
        min_delta = opt.get("early_stopping_min_delta", 0.0001)
        counter = 0

        # ====== 5. TRAINING LOOP ======
        total_start_time = time.time()

        for epoch in range(1, opt["epochs"] + 1):
            # 5.1 Train for one epoch and evaluate
            train_loss, valid_loss, valid_accs, valid_cls_metrics = run_training_epoch(
                model, epoch, loader, optimizer, opt
            )

            # 5.2 Check improvement and save best model
            if valid_accs[1] > best_val_acc + min_delta:
                counter = 0
                best_val_acc = valid_accs[1]
                best_train_loss, best_valid_loss, best_epoch, best_valid_accs = train_loss, valid_loss, epoch, valid_accs

                # Evaluate on test set when we find a new best model
                test_loss, test_accs, test_cls_metrics = evaluate(model, loader.test_data, loader, opt, "test")
                test_ppl = math.exp(test_loss)
                print(f"| Performance on test: Test perplexity: {test_ppl:5.2f} | "
                      f"test acc@1: {test_accs[1]:.4f} | test acc@3: {test_accs[3]:.4f}" + " " * 21 + "|")
                print("-" * 104)

                # Save the best model
                save_best_model(model)
            else:
                # Handle early stopping
                counter += 1
                if epoch % 10 == 0:
                    print(f"| No improvement for {counter} epochs. Best acc@1: {best_val_acc:.4f} |")

            # 5.3 Early stopping
            if counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs without improvement")
                break

            # 5.4 Update scheduler
            scheduler.step()

        # ====== 6. FINAL EVALUATION ======
        total_elapsed = time.time() - total_start_time
        print(f"\nTotal training time: {total_elapsed:.2f} seconds")

        print("\n" + "=" * 50)
        print("FINAL EVALUATION ON TEST SET:")
        final_test_loss, final_test_accs, final_test_cls_metrics = evaluate(model, loader.test_data, loader, opt,
                                                                            "test")
        print_test_results(final_test_loss, final_test_accs, final_test_cls_metrics)

        # Save attention maps from the final model
        try:
            attention_maps = model.get_attention_maps()
            if attention_maps:
                save_attention_maps(attention_maps, f"{MODELS_DIR}/attention_maps.png")
        except Exception as e:
            print(f"Error saving attention maps: {e}")

        return best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics

    except Exception as e:
        print(f"Critical error in main function: {e}")
        return 0.0, 0.0, {1: 0.0, 3: 0.0, 5: 0.0}, 0, {1: 0.0, 3: 0.0, 5: 0.0}, None


if __name__ == "__main__":
    main()