import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from proformer import TransformerModel


# Definizione del classificatore MLP
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model_and_hidden_states(model_path, hidden_states_path):
    """
    Carica il modello Proformer e gli hidden states salvati.

    Args:
        model_path (str): Percorso del file del modello salvato.
        hidden_states_path (str): Percorso del file CSV degli hidden states.

    Returns:
        model (TransformerModel): Modello Proformer caricato.
        hidden_states (torch.Tensor): Hidden states caricati come tensore.
    """
    # Carica il modello Proformer salvato
    model = torch.load(model_path)
    model.eval()  # Imposta il modello in modalità di valutazione

    # Carica gli hidden states dal file CSV
    hidden_states_df = pd.read_csv(hidden_states_path, index_col=0)
    hidden_states = torch.tensor(hidden_states_df.values, dtype=torch.float32)

    return model, hidden_states

def classify(hidden_states, classifier, device):
    """
    Classifica gli hidden states utilizzando il classificatore MLP.

    Args:
        hidden_states (torch.Tensor): Hidden states da classificare.
        classifier (MLPClassifier): Modello classificatore.
        device (torch.device): Dispositivo su cui eseguire il calcolo.

    Returns:
        torch.Tensor: Predizioni delle classi.
    """
    hidden_states = hidden_states.to(device)
    classifier.eval()
    with torch.no_grad():
        outputs = classifier(hidden_states)
        _, predicted = torch.max(outputs, 1)
    return predicted

if __name__ == "__main__":
    # Percorsi dei file
    model_path = "../models/proformer-base.bin"
    hidden_states_path = "../models/hidden_states.csv"

    # Carica il modello Proformer e gli hidden states
    print("Caricamento del modello Proformer e degli hidden states...")
    model, hidden_states = load_model_and_hidden_states(model_path, hidden_states_path)

    # Determina automaticamente la dimensione di input
    input_dim = hidden_states.shape[1]  # Dimensione degli hidden states
    hidden_dim = 128  # Dimensione del livello nascosto del classificatore
    output_dim = 2   # Numero di classi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inizializza il classificatore
    classifier = MLPClassifier(input_dim, hidden_dim, output_dim).to(device)

    # Carica i pesi del classificatore (se disponibili)
    classifier_weights_path = "models/classifier_weights.pth"
    try:
        classifier.load_state_dict(torch.load(classifier_weights_path))
        print("Pesi del classificatore caricati con successo.")
    except FileNotFoundError:
        print("Pesi del classificatore non trovati. Assicurati di addestrare il classificatore prima di usarlo.")

    # Classifica gli hidden states
    print("Classificazione degli hidden states...")
    predictions = classify(hidden_states, classifier, device)

    # Stampa le predizioni
    print(f"Predizioni delle classi: {predictions.tolist()}")
    # Salva le predizioni in un file CSV
    predictions_df = pd.DataFrame(predictions.cpu().numpy(), columns=["Predicted Class"])
    predictions_df.to_csv("models/predictions.csv", index=False)
    print("Predizioni salvate in 'models/predictions.csv'.")