# bpi_params = {
#                 'batch_size': 4,
#                 'd_hid': 32,
#                 'd_model': 32,
#                 'dropout': 0.2163753310981849,
#                 'gamma_scheduler': 0.989695663239636,
#                 'lr': 0.0028363294173614,
#                 'nhead': 1,
#                 'nlayers': 1,
#                 'taxonomy_emb_size': 32,
#                 'taxonomy_emb_type': 'laplacian',
#                 'use_pe': False,
#                 # 'use_taxonomy': True,
#                 "epochs": 150,
#                 "bptt": 237,
#                 "split_actions": True,
#                 "pad": True,
#                 "test_split_size": 5000,
#                 "pos_enc_dropout": 0.01,
#                 "use_l2_data": False,
#               }

# bpi_params = {
#     'batch_size': 32,  # Aumenta la dimensione del batch
#     'd_hid': 512,  # Aumenta la dimensione del feed-forward
#     'd_model': 256,  # Aumenta la dimensione del modello
#     'dropout': 0.2,  # Modifica il tasso di dropout
#     'gamma_scheduler': 0.95,  # Modifica il fattore di riduzione del tasso di apprendimento
#     'lr': 0.0001,  # Modifica il tasso di apprendimento
#     'nhead': 8,  # Aumenta il numero di teste di attenzione
#     'nlayers': 6,  # Aumenta il numero di strati
#     'taxonomy_emb_size': 32,
#     'taxonomy_emb_type': 'laplacian',
#     'use_pe': False,
#     # 'use_taxonomy': True,
#     "epochs": 150,
#     "bptt": 237,
#     "split_actions": True,
#     "pad": True,
#     "test_split_size": 5000,
#     "pos_enc_dropout": 0.1,
#     "use_l2_data": False,
# }

bpi_params = {
    'batch_size': 128,  # Increase for better gradient estimation
    'd_hid': 1024,     # Standard ratio: 4x d_model
    'd_model': 256,    # Keep current size
    'dropout': 0.3,    # Slightly higher to prevent overfitting
    'gamma_scheduler': 0.98,  # Slower decay for better convergence
    'lr': 0.0005,      # Higher learning rate with warmup
    'warmup_steps': 6000,  # Add warmup for stable initial training
    'nhead': 16,        # Current value is appropriate
    'nlayers': 8,      # Current value is good
    'taxonomy_emb_size': 256,  # Increase taxonomic representation
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