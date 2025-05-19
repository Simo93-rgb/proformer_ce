
def plot_attention_maps(attention_maps, output_file, tokens=None, layer_idx=0, head_idx=0, title="Mappa di Attenzione"):
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



