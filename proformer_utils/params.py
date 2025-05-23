import argparse
from typing import Dict, Any, Tuple, Optional


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
    parser.add_argument("--train", action="store_false", default=True)
    args = parser.parse_args()
    opt = vars(args)

    return opt

bpi_params = {
    'batch_size': 32,  # Increase for better gradient estimation
    'd_hid': 512,     # Standard ratio: 4x d_model
    'd_model': 128,    # Keep current size
    'dropout': 0.187731533170227,    # Slightly higher to prevent overfitting
    'gamma_scheduler': 0.9898196793787607,  # Slower decay for better convergence
    'lr': 0.00129695816783916,      # Higher learning rate with warmup
    'warmup_steps': 6000,  # Add warmup for stable initial training
    'nhead': 2,        # Current value is appropriate
    'nlayers': 3,      # Current value is good
    'taxonomy_emb_size': 32,  # Increase taxonomic representation
    'taxonomy_emb_type': 'laplacian',
    'use_pe': True,    # Enable proper positional encoding
    'weight_decay': 1e-5,  # Add L2 regularization
    "epochs": 300,     # Train longer with early stopping
    "bptt": 980,
    "split_actions": True,
    "pad": True,
    "test_split_size": 1000,
    "pos_enc_dropout": 0.1,
    "use_taxonomy": False,  # Enable if taxonomy data is available
    "use_l2_data": False,
    "gradient_clip": 1.0,  # Add gradient clipping for stability
        # Early stopping parameters
    'early_stopping_patience': 15,  # Stop after this many epochs without improvement
    'early_stopping_min_delta': 0.0001,  # Minimum change to count as improvement
}