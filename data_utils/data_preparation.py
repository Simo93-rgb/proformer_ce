import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(dataset_path):
    df = pd.read_csv(dataset_path)
    traces = df.iloc[:, 2].str.split(",")  # Colonna delle sequenze
    labels = df.iloc[:, 3].map(lambda x: int(x.split("_")[1]))  # Colonna delle classi (es. "class_0" -> 0)

    # Dividi in training e validation set
    train_traces, val_traces, train_labels, val_labels = train_test_split(
        traces, labels, test_size=0.2, random_state=42
    )
    return train_traces, val_traces, train_labels, val_labels