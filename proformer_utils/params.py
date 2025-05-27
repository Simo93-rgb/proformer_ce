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
    parser.add_argument("--mask_prob", type=float, default=params['mask_prob'])
    args = parser.parse_args()
    opt = vars(args)

    return opt


bpi_params = {
    # Parametri di base del batch e sequenza
    'batch_size': 4,  # Dimensione del batch - Numero di esempi elaborati contemporaneamente
    'bptt': 980,  # Backpropagation Through Time - Lunghezza massima della sequenza di input

    # Parametri dell'architettura del transformer
    'd_model': 1024,  # Dimension Model - Dimensione degli embedding e degli strati del transformer - Divisibile per il numero di teste
    'd_hid': 2048,  # Dimension Hidden - Dimensione dello strato feed-forward interno del transformer
    'nlayers': 6,  # Number of Layers - Numero di layer transformer impilati
    'nhead': 8,  # Number of Heads - Numero di teste nel meccanismo di multi-head attention
    'dropout': 0.1,  # Dropout Rate - Probabilità di dropout per la regolarizzazione

    # Parametri di ottimizzazione
    'lr': 0.0000129695816783916,  # Learning Rate - Velocità di apprendimento iniziale
    'gamma_scheduler': 0.9898196793787607,  # Gamma Scheduler - Fattore di decadimento per lo scheduler LR
    'weight_decay': 1e-5,  # Weight Decay - Regolarizzazione L2 per prevenire overfitting
    'gradient_clip': 1.0,  # Gradient Clipping - Limite massimo della norma del gradiente

    # Parametri di addestramento
    'epochs': 300,  # Epochs - Numero massimo di cicli completi di addestramento
    'warmup_steps': 6000,  # Warmup Steps - Passi iniziali per incrementare gradualmente il learning rate
    'early_stopping_patience': 10,  # Patience - Epoche senza miglioramento prima di fermare l'addestramento
    'early_stopping_min_delta': 0.0001,  # Min Delta - Miglioramento minimo da considerare significativo
    'mask_prob': 0.5,  # Mask Probability - Probabilità di mascherare token per apprendimento MLM

    # Parametri di positional encoding
    'use_pe': True,  # Use Positional Encoding - Se utilizzare l'encoding posizionale
    'pos_enc_dropout': 0.1,  # Positional Encoding Dropout - Dropout specifico per l'encoding posizionale

    # Parametri della tassonomia
    'use_taxonomy': False,  # Use Taxonomy - Se utilizzare embedding basati su tassonomia
    'taxonomy_emb_type': 'laplacian',  # Taxonomy Embedding Type - Tipo di embedding per la tassonomia
    'taxonomy_emb_size': 32,  # Taxonomy Embedding Size - Dimensione degli embedding della tassonomia

    # Parametri di processamento dati
    'split_actions': True,  # Split Actions - Se dividere le azioni in token separati
    'pad': True,  # Padding - Se applicare padding alle sequenze più corte
    'test_split_size': 1000,  # Test Split Size - Numero di esempi da riservare per il test
    'use_l2_data': False,  # Use L2 Data - Se utilizzare dati di livello 2 (maggior dettaglio)
}