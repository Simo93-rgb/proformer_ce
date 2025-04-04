import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from classification.proformer_classifier import ProformerClassifier
from data_utils.data_preparation import prepare_data


def fine_tune_model(model, train_data, val_data, vocab, device, epochs=10, lr=1e-4):
    train_traces, train_labels = train_data
    val_traces, val_labels = val_data

    # Definisci la loss e l'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for trace, label in zip(train_traces, train_labels):
            trace_indices = [vocab[event] if event in vocab else vocab["<unk>"] for event in trace]
            trace_tensor = torch.tensor(trace_indices, dtype=torch.long).unsqueeze(1).to(device)
            attn_mask = model.encoder.create_masked_attention_matrix(trace_tensor.size(0)).to(device)

            label_tensor = torch.tensor([label], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(trace_tensor, attn_mask)
            loss = criterion(logits, label_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_traces)}")

        # Valutazione sul validation set
        model.eval()
        correct = 0
        with torch.no_grad():
            for trace, label in zip(val_traces, val_labels):
                trace_indices = [vocab[event] if event in vocab else vocab["<unk>"] for event in trace]
                trace_tensor = torch.tensor(trace_indices, dtype=torch.long).unsqueeze(1).to(device)
                attn_mask = model.encoder.create_masked_attention_matrix(trace_tensor.size(0)).to(device)

                logits = model(trace_tensor, attn_mask)
                predicted = torch.argmax(logits, dim=1).item()
                if predicted == label:
                    correct += 1

        accuracy = correct / len(val_traces)
        print(f"Validation Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    dataset_path = "data/aggregated_case_detailed_to_classify_final.csv"
    vocab_path = "models/vocab.pkl"
    model_path = "models/proformer-base.bin"

    # Carica il vocabolario
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    print(type(vocab))
    print(vocab)

    # Carica il modello pre-addestrato
    encoder = torch.load(model_path)
    input_dim = encoder.d_model  # Dimensione del modello
    num_classes = 2  # Numero di classi

    # Inizializza il modello con la testa di classificazione
    model = ProformerClassifier(encoder, input_dim, num_classes)

    # Prepara i dati
    train_traces, val_traces, train_labels, val_labels = prepare_data(dataset_path)

    # Esegui il fine-tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fine_tune_model(
        model,
        (train_traces, train_labels),
        (val_traces, val_labels),
        vocab,
        device,
        epochs=10,
        lr=1e-4
    )