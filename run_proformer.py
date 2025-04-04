import time
import math
import random
import argparse
import torch
import torch.nn.functional as F
from dataloader import Dataloader
from proformer import TransformerModel
from params import bpi_params
from taxonomy import Taxonomy, TaxonomyEmbedding
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

def parse_params():
    """
    Parse command-line arguments for Proformer configuration.

    Returns:
        dict: Dictionary containing all configuration parameters
    """
    parser = argparse.ArgumentParser()

    # Add arguments with defaults from bpi_params
    parser.add_argument("--device", type=str, default=bpi_params.get("device", "cuda:0"))
    parser.add_argument("--test_split_size", type=int, default=bpi_params.get("test_split_size", 1000), help="Number of examples to use for valid and test")
    parser.add_argument("--pad", action="store_true", help="Pads the sequences to bptt", default=bpi_params.get("pad", True))
    parser.add_argument("--bptt", type=int, default=bpi_params.get("bptt", 237), help="Max len of sequences")
    parser.add_argument("--split_actions", action="store_true", default=bpi_params.get("split_actions", True), help="Splits multiple action if in one (uses .split('_se_'))")
    parser.add_argument("--batch_size", type=int, default=bpi_params.get("batch_size", 2), help="Regulates the batch size")
    parser.add_argument("--pos_enc_dropout", type=float, default=bpi_params.get("pos_enc_dropout", 0.1), help="Regulates dropout in pe")
    parser.add_argument("--d_model", type=int, default=bpi_params.get("d_model", 128))
    parser.add_argument("--nhead", type=int, default=bpi_params.get("nhead", 1))
    parser.add_argument("--nlayers", type=int, default=bpi_params.get("nlayers", 3))
    parser.add_argument("--dropout", type=float, default=bpi_params.get("dropout", 0.1))
    parser.add_argument("--d_hid", type=int, default=bpi_params.get("d_hid", 128))
    parser.add_argument("--epochs", type=int, default=bpi_params.get("epochs", 150))
    parser.add_argument("--lr", type=float, default=bpi_params.get("lr", 3.))
    parser.add_argument("--gamma_scheduler", type=float, default=bpi_params.get("gamma_scheduler", 0.97))
    parser.add_argument("--use_l2_data", action="store_true", default=bpi_params.get("use_l2_data", True), help="Uses data from level 2 dataset")
    parser.add_argument("--use_taxonomy", action="store_true", default=bpi_params.get("use_taxonomy", False), help="Introduces weights based on a taxonomy of the tokens")
    parser.add_argument("--use_pe", action="store_true", default=bpi_params.get("use_pe", False))
    parser.add_argument("--taxonomy_emb_type", type=str, default=bpi_params.get("taxonomy_emb_type", "laplacian"))
    parser.add_argument("--taxonomy_emb_size", type=int, default=bpi_params.get("taxonomy_emb_size", 64))
    parser.add_argument("--weight_decay", type=float, default=bpi_params.get("weight_decay", 1e-5), help="L2 regularization weight decay")
    parser.add_argument("--warmup_steps", type=int, default=bpi_params.get("warmup_steps", 4000), help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--gradient_clip", type=float, default=bpi_params.get("gradient_clip", 1.0), help="Gradient clipping value")
    parser.add_argument("--early_stopping_patience", type=int, default=bpi_params.get("early_stopping_patience", 15), help="Patience for early stopping")
    parser.add_argument("--early_stopping_min_delta", type=float, default=bpi_params.get("early_stopping_min_delta", 0.0001), help="Minimum delta for early stopping")

    # Default dataset and taxonomy files
    data_dir = "data"
    dataset_filename = "finale_preprocessed.csv"
    taxonomy_filename = "bpi_taxonomy.csv"
    dataset_file_path = os.path.join(data_dir, dataset_filename)
    taxonomy_file_path = os.path.join(data_dir, taxonomy_filename)
    # check if the path is ok
    if not os.path.exists(dataset_file_path):
        raise FileNotFoundError(f"The file {dataset_file_path} does not exist. Please check the path and try again.")

    if not os.path.exists(taxonomy_file_path):
        raise FileNotFoundError(f"The file {taxonomy_file_path} does not exist. Please check the path and try again.")

    #    parser.add_argument("--dataset", type=str, default="data/BPI_Challenge_2012.csv")
    parser.add_argument("--dataset", type=str, default=dataset_file_path)
    parser.add_argument("--taxonomy", type=str, default=taxonomy_file_path)

    args = parser.parse_args()
    opt = vars(args)

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
    """
    Train the model for one epoch.

    Args:
        model (TransformerModel): The Proformer model to train
        opt (dict): Configuration parameters
        loader (Dataloader): Data loader for training data
        optimizer (torch.optim.Optimizer): Optimizer for parameter updates

    Returns:
        float: Average training loss for the epoch
    """
    batch:int=0
    model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(loader.train_data) // opt["bptt"]

    for batch, i in enumerate(range(0, loader.train_data.size(0) - 1, opt["bptt"])):
        # Get batch data and targets
        data, targets = loader.get_batch(loader.train_data, i)
        # Create attention mask to prevent attending to future tokens
        attn_mask = model.create_masked_attention_matrix(data.size(0)).to(opt["device"])

        output = model(data, attn_mask)
        output_flat = output.view(-1, model.ntokens)

        # Create mask to ignore padding tokens (1 and 8)
        pad_mask = (targets != 1) & (targets != 8)
        targets = targets[pad_mask]
        output_flat = output_flat[pad_mask, :]

        weights = torch.ones(model.ntokens).to(opt["device"])

        # Calculate cross entropy loss
        loss = F.cross_entropy(output_flat, targets, weight=weights)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Gradient clipping for stability
        optimizer.step()

        total_loss += loss.item()

    return total_loss / (batch+1)


def evaluate(model, eval_data, loader, opt):
    """
    Evaluate the model on validation or test data.

    Args:
        model (TransformerModel): The Proformer model to evaluate
        eval_data (torch.Tensor): Evaluation dataset
        loader (Dataloader): Data loader
        opt (dict): Configuration parameters

    Returns:
        tuple: (loss, accuracies_dict) containing perplexity and accuracy metrics
    """
    model.eval()
    total_loss = 0.
    accs = {1: 0., 3: 0., 5: 0.}  # Track accuracy@1, accuracy@3, accuracy@5

    with torch.no_grad():
        for batch,i in enumerate(range(0, eval_data.size(0) - 1, opt["bptt"])):
            # Get batch data
            data, targets = loader.get_batch(eval_data, i)
            attn_mask = model.create_masked_attention_matrix(data.size(0)).to(opt["device"])

            seq_len = data.size(0)
            output = model(data, attn_mask)

            output_flat = output.view(-1, model.ntokens)

            # Filter out padding tokens
            pad_mask = (targets != 1) & (targets != 8)
            targets = targets[pad_mask]
            output_flat = output_flat[pad_mask, :]

            # Calculate loss and accuracy metrics
            total_loss += seq_len * F.cross_entropy(output_flat, targets).item()
            accs = get_ranked_metrics(accs, output_flat, targets)

        # Normalize metrics by number of batches
        for k in accs.keys():
            accs[k] = accs[k] / (batch+1)
        loss = total_loss / (len(eval_data) - 1)

    return loss, accs


def main(opt):
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
        opt = parse_params()
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
        model = TransformerModel(vocab_dim, opt).to(opt["device"])

    # Setup optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=opt["gamma_scheduler"])
    best_val_acc = -float('inf')

    # Initialize early stopping variables
    patience = opt.get("early_stopping_patience", 15)  # Default to 15 if not specified
    min_delta = opt.get("early_stopping_min_delta", 0.0001) # Default to 0.0001 if not specified
    counter = 0
    # Training loop
    for epoch in range(1, opt["epochs"]+1):
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss = train(model, opt, loader, optimizer)
        # Evaluate on validation set
        valid_loss, valid_accs = evaluate(model, loader.valid_data, loader, opt)
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
            test_loss, test_accs = evaluate(model, loader.test_data, loader, opt)
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

    return best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs

if __name__ == "__main__":
    opt = parse_params()
    opt["dataset"] = "data/aggregated_case_details.csv"
    best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs = main(opt=opt)
    print(f"Best epoch: {best_epoch} \t loss: {best_valid_loss} \t best accs: {best_valid_accs}")