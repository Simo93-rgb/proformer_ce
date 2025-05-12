import ast

import torch
import pandas as pd
import pickle
from dataloader import Dataloader
from proformer import TransformerModel

def classify_new_traces(model_path, vocab_path, dataset_path, output_path, opt):
    """
    Classify new traces using the trained Proformer model.

    Args:
        model_path (str): Path to the saved model.
        vocab_path (str): Path to the saved vocabulary.
        dataset_path (str): Path to the CSV file containing new traces.
        output_path (str): Path to save the classification results.
        opt (dict): Configuration parameters.
    """
    # Load the saved model
    model = torch.load(model_path)
    model.eval()

    # Load the saved vocabulary
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    print("Vocab size:", len(vocab))  # Stampa la dimensione del vocabolario

    # Load the new dataset
    df = pd.read_csv(dataset_path)


    # Applica il parsing delle attività
    traces = df.iloc[:, 1].apply(lambda x: [action[0] for action in ast.literal_eval(x) if isinstance(action, tuple)])
    predictions = []
    with torch.no_grad():
        for trace in traces:
            # Preprocess the trace
            trace_indices = [vocab[event] if event in vocab else vocab["<unk>"] for event in trace]
            trace_tensor = torch.tensor(trace_indices, dtype=torch.long).unsqueeze(1).to(opt["device"])  # Add batch dimension

            # Create attention mask
            attn_mask = model.create_masked_attention_matrix(trace_tensor.size(0)).to(opt["device"])

            # Generate predictions
            output = model(trace_tensor, attn_mask)  # Output shape: [seq_len, batch_size, num_classes]
            predicted_class = torch.argmax(output[-1], dim=1).item()  # Use the last token's prediction
            predictions.append(predicted_class)

    # Save predictions to CSV
    df["Predicted Class"] = predictions
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    opt = {
        "device": "cuda:0",
        "bptt": 237,  # Ensure this matches the training configuration
    }
    classify_new_traces(
        model_path="models/proformer-base.bin",
        vocab_path="models/vocab.pkl",
        dataset_path="data/aggregated_case_tuple_to_classify.csv",
        output_path="data/classification_results.csv",
        opt=opt
    )