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

    def batchify_sequences(self, sequences, batch_size):
        # sequences: lista di sequenze (ognuna già paddata a bptt)
        batches = []
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            # Stack per ottenere un tensore (batch_size, seq_len)
            batch_tensor = torch.stack([torch.tensor(self.vocab(seq), dtype=torch.long) for seq in batch])
            batches.append(batch_tensor.to(self.opt["device"]))
        return batches

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

    def get_batch_from_list(self, source, batch_idx):
        # source è una lista di batch, ognuno shape (batch_size, seq_len)
        data = source[batch_idx]  # shape: (batch_size, seq_len)
        # Per language modeling: target è la stessa sequenza shiftata di 1 a sinistra
        target = data[:, 1:].contiguous().view(-1)  # shape: (batch_size * (seq_len-1))
        data = data[:, :-1].contiguous()  # shape: (batch_size, seq_len-1)
        return data, target

    def get_batch_labels(self, batch_idx, batch_size=None):
        """
        Ottiene le etichette per un batch specifico.
        """
        # Ricava la lista dati corretta
        if hasattr(self, "train_data") and self.train_data is not None:
            data_list = self.train_data
        elif hasattr(self, "valid_data") and self.valid_data is not None:
            data_list = self.valid_data
        elif hasattr(self, "test_data") and self.test_data is not None:
            data_list = self.test_data
        else:
            raise ValueError("Nessun dato batch trovato.")

        # Calcola la dimensione reale del batch corrente
        real_batch_size = data_list[batch_idx].shape[0]

        # Calcola start_idx sommando le dimensioni reali dei batch precedenti
        start_idx = sum(data_list[i].shape[0] for i in range(batch_idx))
        end_idx = start_idx + real_batch_size
        return self.labels[start_idx:end_idx].to(self.opt["device"])

    # def get_batch_labels(self, batch_idx, batch_size=None):
    #     """
    #     Ottiene le etichette per un batch specifico.
    #     """
    #     if batch_size is None:
    #         batch_size = self.opt["batch_size"]
    #
    #     start_idx = batch_idx * batch_size
    #     end_idx = min(start_idx + batch_size, len(self.labels))
    #
    #     return self.labels[start_idx:end_idx].to(self.opt["device"])

    # def get_masked_batch_labels(self, batch_idx, mask_positions):
    #     batch_labels = self.get_batch_labels(batch_idx)
    #     # mask_positions: (batch_size, seq_len-1) -> True se c'è almeno un <mask> nella sequenza
    #     has_mask = mask_positions.any(dim=1)  # (batch_size,)
    #     return batch_labels[has_mask], has_mask
    def get_masked_batch_labels(self, batch_idx, mask_positions):
        real_batch_size = mask_positions.shape[0]
        batch_labels = self.get_batch_labels(batch_idx, batch_size=real_batch_size)
        has_mask = mask_positions.any(dim=1)  # (real_batch_size,)
        return batch_labels[has_mask], has_mask

    def get_dataset(self, num_test_ex=100, vocab = None, debugging=False):
        raw_act_seq = self.df.groupby("case_id").apply(lambda x: x.activity.to_list())
        # Estrai le etichette dalle sequenze se necessario
        labels = []
        for seq in raw_act_seq:
            # Cerca "class_0" o "class_1" nella sequenza
            if "class_0" in seq:
                labels.append(0.0)
            elif "class_1" in seq:
                labels.append(1.0)
            else:
                # Gestione del caso in cui l'etichetta non è presente
                labels.append(float('nan'))

        act_seq = self.df.groupby("case_id").apply(
            lambda x: self.process_seq(x.activity.to_list())
        )
        if self.opt["use_l2_data"]:
            act_seq = act_seq + self.act_seq_l2


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
        self.vocab = vocab
        # Salva le labels per il training
        self.labels = torch.tensor(labels, dtype=torch.float)

        # Salva il vocabolario in un file per debugging
        if debugging:
            with open(f"{DATA_DIR}/vocab.txt", "w") as f:
                for token in vocab.get_itos():
                    f.write(f"{token}\n")

        # Prepara i dati
        train_raw_data = [item for sublist in train_act_seq for item in sublist]
        valid_raw_data = [item for sublist in valid_act_seq for item in sublist]
        test_raw_data = [item for sublist in test_act_seq for item in sublist]

        # train_data = self.batchify(self.data_process(train_raw_data, vocab), bsz=len(train_act_seq) // self.opt["batch_size"])
        # valid_data = self.batchify(self.data_process(valid_raw_data, vocab), bsz=len(valid_act_seq) // self.opt["batch_size"])
        # test_data = self.batchify(self.data_process(test_raw_data, vocab), bsz=len(test_act_seq) // self.opt["batch_size"])

        train_data = self.batchify_sequences(train_act_seq, batch_size=self.opt["batch_size"])
        valid_data = self.batchify_sequences(valid_act_seq, batch_size=self.opt["batch_size"])
        test_data = self.batchify_sequences(test_act_seq, batch_size=self.opt["batch_size"])

        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data


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
