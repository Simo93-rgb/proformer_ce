import torch
import pandas as pd
from typing import Tuple
from config import MODELS_DIR
from typing import Dict

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



def save_hidden_states(hidden_states: torch.Tensor, path: str = f"{MODELS_DIR}/hidden_states.csv") -> None:
    """
    Save hidden states as a CSV file.

    Args:
        hidden_states (torch.Tensor): Hidden states tensor.
        path (str): Output CSV path.
    """
    hs_cpu = hidden_states.cpu()
    df = pd.DataFrame(hs_cpu.reshape(-1, hs_cpu.shape[-1]))
    df.to_csv(path, index=False)

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