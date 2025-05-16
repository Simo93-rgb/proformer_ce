# This file contains the parameters for the BPI model.
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
    "bptt": 237,
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