import matplotlib
matplotlib.use('Agg')  # Usa il backend non interattivo

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import numpy as np
import torch
from proformer_utils.tensor_utils import create_attention_mask
from config import MODELS_DIR

def save_attention_maps(model, data_loader, opt, output_dir=f"{MODELS_DIR}/attention_maps"):
    """
    Salva le mappe di attenzione del modello per un singolo batch di dati.

    Args:
        model (TransformerModel): Il modello proformer
        data_loader (Dataloader): Il data loader
        opt (dict): Le opzioni di configurazione
        output_dir (str): Directory dove salvare le mappe
    """
    try:
        # Crea la directory di output se non esiste
        os.makedirs(output_dir, exist_ok=True)

        model.eval()
        with torch.no_grad():
            # Prendi un singolo batch
            data, targets = data_loader.get_batch(data_loader.test_data, 0)
            attn_mask = create_attention_mask(data.size(0), opt["device"])

            # Esegui il forward pass
            _ = model(data, attn_mask)

            # Ottieni le mappe di attenzione
            attention_maps = model.get_attention_maps()
            if not attention_maps:
                print("Nessuna mappa di attenzione disponibile nel modello")
                return

            # Ricava i token dell'input usando il vocabolario corretto
            input_tokens = [data_loader.vocab.get_itos()[idx] for idx in data[0].tolist()]

            total_maps = len(attention_maps) * attention_maps[0].shape[0]
            print(f"Generando {total_maps} mappe di attenzione...")
            
            # Salva le mappe per ogni livello e testa
            for layer_idx in range(len(attention_maps)):
                num_heads = attention_maps[layer_idx].shape[0]
                for head_idx in range(num_heads):
                    output_file = f"{output_dir}/attn_layer{layer_idx}_head{head_idx}.png"
                    print(f"Elaborazione mappa: layer {layer_idx}, head {head_idx}")
                    plot_attention_maps(
                        attention_maps,
                        output_file,
                        tokens=input_tokens,
                        layer_idx=layer_idx,
                        head_idx=head_idx,
                        title=f"Attention Map"
                    )

            print(f"Mappe di attenzione salvate in {output_dir}")
    except Exception as e:
        print(f"Errore durante la generazione delle mappe di attenzione: {e}")
        
def plot_attention_maps(attention_maps, output_file, tokens=None, layer_idx=0, head_idx=0, title="Mappa di Attenzione"):
    """
    Visualizza e salva le mappe di attenzione per un determinato livello e testa,
    utilizzando indici numerici nella heatmap e una legenda separata.

    Args:
        attention_maps: Lista di tensori di attenzione
        output_file: Percorso del file dove salvare il grafico
        tokens: Lista di token per etichettare gli assi (optional)
        layer_idx: Indice del livello da visualizzare (default: 0)
        head_idx: Indice della testa di attenzione da visualizzare (default: 0)
        title: Titolo del grafico
    """

    if layer_idx >= len(attention_maps):
        print(f"Errore: indice del livello {layer_idx} fuori range. Disponibili {len(attention_maps)} livelli.")
        return

    attn = attention_maps[layer_idx]
    if len(attn.shape) == 2:
        attn = attn.unsqueeze(0)  # Aggiungi la dimensione della testa per coerenza

    if head_idx >= attn.shape[0]:
        print(f"Errore: indice della testa {head_idx} fuori range. Disponibili {attn.shape[0]} teste.")
        return

    attn = attn[head_idx].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn, cmap='viridis')

    # Mostra gli indici numerici sulla heatmap
    ax.set_xticks(np.arange(attn.shape[1]))
    ax.set_yticks(np.arange(attn.shape[0]))
    ax.set_xticklabels(np.arange(1, attn.shape[1] + 1))  # Inizia da 1 per chiarezza
    ax.set_yticklabels(np.arange(1, attn.shape[0] + 1))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_title(f"{title} - Livello {layer_idx}, Testa {head_idx}")
    ax.set_xlabel("Key Index")
    ax.set_ylabel("Query Index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.savefig(output_file.replace(".png", "_heatmap.png"), dpi=300)  # Salva la heatmap

    # Crea e salva la legenda separata
    if tokens is not None:
        plt.figure(figsize=(8, len(tokens) // 2))  # Regola l'altezza in base al numero di token
        plt.axis('off')
        patches = [mpatches.Patch(color='none', label=f"{i + 1}: {token}") for i, token in enumerate(tokens)]  # Inizia l'indice da 1
        plt.legend(handles=patches, loc='center', fontsize='small')
        plt.tight_layout()
        plt.savefig(output_file.replace(".png", "_legend.png"), dpi=300)  # Salva la legenda
    plt.close('all') # Close all figures to prevent memory issues

    
def plot_attention_maps_old(attention_maps, output_file, tokens=None, layer_idx=0, head_idx=0, title="Mappa di Attenzione"):
    """
    Visualizza e salva le mappe di attenzione per un determinato livello e testa

    Args:
        attention_maps: Lista di tensori di attenzione
        output_file: Percorso del file dove salvare il grafico
        tokens: Lista di token per etichettare gli assi (optional)
        layer_idx: Indice del livello da visualizzare (default: 0)
        head_idx: Indice della testa di attenzione da visualizzare (default: 0)
        title: Titolo del grafico
    """
    import matplotlib.pyplot as plt

    if layer_idx >= len(attention_maps):
        print(f"Errore: indice del livello {layer_idx} fuori range. Disponibili {len(attention_maps)} livelli.")
        return

    # Estrazione della mappa di attenzione specifica
    attn = attention_maps[layer_idx]

    # Se la mappa è stata mediata tra le teste, visualizziamo direttamente
    if len(attn.shape) == 2:
        plt.figure(figsize=(8, 6))
        plt.imshow(attn.detach().cpu().numpy(), cmap='viridis')
        plt.title(f"{title} - Livello {layer_idx}")
    else:
        # Altrimenti, estraiamo la testa specifica
        if head_idx >= attn.shape[0]:
            print(f"Errore: indice della testa {head_idx} fuori range. Disponibili {attn.shape[0]} teste.")
            return

        plt.figure(figsize=(8, 6))
        plt.imshow(attn[head_idx].detach().cpu().numpy(), cmap='viridis')
        plt.title(f"{title} - Livello {layer_idx}, Testa {head_idx}")

    # Aggiungi etichette degli assi se forniti i token
    if tokens is not None:
        plt.xticks(range(len(tokens)), tokens, rotation=90)
        plt.yticks(range(len(tokens)), tokens)

    plt.xlabel("Key (indice token input)")
    plt.ylabel("Query (indice token input)")
    plt.colorbar()
    plt.tight_layout()

    # Salva l'immagine su file invece di mostrarla
    plt.savefig(output_file, dpi=300)
    plt.close()



