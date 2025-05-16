import numpy as np 
import pandas as pd
import torch
import torch.nn.functional as F 
import os
import random
import pickle
from config import DATA_DIR
import torchtext
from torchtext.vocab import build_vocab_from_iterator
torchtext.disable_torchtext_deprecation_warning()


# This code was ispired by the following example:
# https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py


class Dataloader:
    def __init__(self, filename, opt):
        self.labels = None
        self.vocab = None
        self.test_data = None
        self.valid_data = None
        self.train_data = None
        self.opt = opt
        # removed for excluding timestamp parsing, was true in first model
        # self.df = pd.read_csv(filename, parse_dates=["timestamp"])
        self.df = pd.read_csv(filename)
        print (self.df)
        if self.opt["use_l2_data"]:
            with open(f'{DATA_DIR}/level2_dataset_preprocessed.pkl', 'rb') as handle:
                act_seq_l2 = pickle.load(handle)
            self.act_seq_l2 = list(map(self.process_seq, act_seq_l2))
        
    
    def preprocessor(self, filename):
        df = pd.read_csv(filename, parse_dates=["Complete Timestamp"])

        df = df.rename(columns={"Case ID": "case_id", "Activity": "activity", 
                                "Resource": "resource", "Complete Timestamp": "timestamp",
                                "Variant": "variant", "Variant index": "variant_id"})

        df.case_id = df.case_id.str.removeprefix("Case ").astype(int)
        df.resource = df.resource.str.removeprefix("Value ").astype(int)
        df.variant = df.variant.str.removeprefix("Variant ").astype(int)
        df["cat_activity"] = df.activity.astype("category").cat.codes
        df.to_csv(f"{filename}_preprocessed.csv", index=False)

        return df

    def process_seq(self, x):
        lx = x
        if self.opt["split_actions"]:
            lx = [i.split("_se_")[::-1] for i in lx]
            lx = [item for row in lx for item in row]

        # Cerca "class_0" o "class_1" nella sequenza e sostituiscili con <mask>
        processed_lx = []

        for item in lx:
            if item == "class_0":
                # Mascheramento casuale per il training (80% di probabilità)
                if random.random() < 0.8:  # Usare opt.get("mask_prob", 0.8) nella versione finale
                    processed_lx.append("<mask>")
                else:
                    processed_lx.append("<cls0>")
            elif item == "class_1":
                if random.random() < 0.8:
                    processed_lx.append("<mask>")
                else:
                    processed_lx.append("<cls1>")
            else:
                processed_lx.append(item)

        out = ["<sos>"] + processed_lx + ["<eos>"]

        if self.opt["pad"]:
            if (len(out) < self.opt["bptt"]):
                out = out + ["<pad>" for _ in range(self.opt["bptt"] - len(out))]

        return out


    def yield_tokens(self):
        act_seq = self.df.groupby("case_id").apply(lambda x : self.process_seq(x.activity.to_list()))
        for act in act_seq:
            yield act

    def batchify(self, data, bsz=80):
        """
        Divides the data into bsz separate sequences.

        Args:
            data (torch.Tensor): The input data tensor to be divided into batches.
            bsz (int, optional): The batch size, i.e., the number of sequences to divide the data into. Defaults to 80.

        Returns:
            torch.Tensor: A tensor of shape (seq_len, bsz) where seq_len is the length of each batch,
                          transposed and moved to the device specified in self.opt["device"].
        """

        # Calculate the length of each batch by dividing the total number of elements by the batch size
        seq_len = data.size(0) // bsz

        # Trim the data tensor to ensure its length is an exact multiple of the batch size
        data = data[:seq_len * bsz]

        # Reshape the trimmed data tensor into a 2D tensor of shape (bsz, seq_len) and transpose it
        data = data.view(bsz, seq_len).t().contiguous()

        # Move the reshaped and transposed data tensor to the specified device and return it
        return data.to(self.opt["device"])

    def data_process(self, raw_text_iter, vocab):
        """
        Processes raw text data into a concatenated tensor of token indices.

        Args:
            raw_text_iter (iterable): An iterable of raw text data.
            vocab (torchtext.vocab.Vocab): A vocabulary object that maps tokens to indices.

        Returns:
            torch.Tensor: A concatenated tensor of token indices.

        Raises:
            ValueError: If no valid tokens are found for concatenation.
        """
        # Convert each item in raw_text_iter into a tensor of token indices using the provided vocabulary
        data = [torch.tensor(vocab([item]), dtype=torch.long) for item in raw_text_iter]

        # Filter out any empty tensors from the list
        non_empty_tensors = list(filter(lambda t: t.numel() > 0, data))

        # Raise an error if no valid tokens are found for concatenation
        if not non_empty_tensors:
            raise ValueError("No valid tokens found for concatenation. Check your data and vocabulary.")

        # Concatenate the non-empty tensors into a single tensor and return it
        return torch.cat(non_empty_tensors)


    def get_batch(self, source, i):
        seq_len = min(self.opt["bptt"], len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len].reshape(-1)

        return data, target

    def get_batch_labels(self, batch_idx, batch_size=None):
        """
        Ottiene le etichette per un batch specifico.
        """
        if batch_size is None:
            batch_size = self.opt["batch_size"]

        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(self.labels))

        return self.labels[start_idx:end_idx].to(self.opt["device"])

    def get_dataset(self, num_test_ex=1000, vocab = None, debugging=False):
        act_seq = self.df.groupby("case_id").apply(
            lambda x: self.process_seq(x.activity.to_list())
        )
        if self.opt["use_l2_data"]:
            act_seq = act_seq + self.act_seq_l2

        # Estrai le etichette dalle sequenze se necessario
        labels = []
        for seq in act_seq:
            # Cerca "class_0" o "class_1" nella sequenza
            if "class_0" in seq:
                labels.append(0.0)
            elif "class_1" in seq:
                labels.append(1.0)
            else:
                # Gestione del caso in cui l'etichetta non è presente
                labels.append(float('nan'))

        train_act_seq = act_seq[num_test_ex:]
        valid_act_seq = act_seq[:num_test_ex - (num_test_ex // 2)]
        test_act_seq = act_seq[num_test_ex - (num_test_ex // 2):num_test_ex]

        # Se il vocabolario non è fornito, costruiscilo
        if vocab is None:
            vocab = build_vocab_from_iterator(
                self.yield_tokens(),
                specials=['<unk>', '<sos>', '<eos>', '<pad>', '<mask>', '<cls0>', '<cls1>']
            )
            vocab.set_default_index(vocab['<unk>'])

        # Salva le labels per il training
        self.labels = torch.tensor(labels, dtype=torch.float)

        # Salva il vocabolario in un file per debugging
        if debugging:
            with open("../data/vocab.txt", "w") as f:
                for token in vocab.get_itos():
                    f.write(f"{token}\n")

        # Prepara i dati
        train_raw_data = [item for sublist in train_act_seq for item in sublist]
        valid_raw_data = [item for sublist in valid_act_seq for item in sublist]
        test_raw_data = [item for sublist in test_act_seq for item in sublist]

        train_data = self.batchify(self.data_process(train_raw_data, vocab), bsz=len(train_act_seq) // self.opt["batch_size"])
        valid_data = self.batchify(self.data_process(valid_raw_data, vocab), bsz=len(valid_act_seq) // self.opt["batch_size"])
        test_data = self.batchify(self.data_process(test_raw_data, vocab), bsz=len(test_act_seq) // self.opt["batch_size"])

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.vocab = vocab

        return vocab, train_data, valid_data, test_data


    def preprocess_trace(self, trace):
        """
        Preprocess a single trace for classification.

        Args:
            trace (list): List of events in the trace.

        Returns:
            torch.Tensor: Preprocessed trace tensor.
        """
        indices = [self.vocab.get(event, self.vocab["<unk>"]) for event in trace]
        return torch.tensor(indices, dtype=torch.long)
