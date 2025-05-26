import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import os
import random
import pickle
from config import DATA_DIR  # Assuming DATA_DIR is correctly defined in config.py
import torchtext
from torchtext.vocab import build_vocab_from_iterator

torchtext.disable_torchtext_deprecation_warning()


# This code was ispired by the following example:
# https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py

class Dataloader:
    def __init__(self, filename, opt):
        self.labels = None  # Will be split into train/valid/test_labels
        self.vocab = None
        self.test_data = None
        self.valid_data = None
        self.train_data = None
        self.opt = opt
        self.df = pd.read_csv(filename)
        print("Initial DataFrame sample:")
        print(self.df.head())

        self.num_test_ex = 0  # Will be set in get_dataset
        self.valid_split_idx = 0  # Will be set in get_dataset
        self.train_labels = torch.empty(0)
        self.valid_labels = torch.empty(0)
        self.test_labels = torch.empty(0)

        if self.opt.get("use_l2_data", False):
            try:
                with open(f'{DATA_DIR}/level2_dataset_preprocessed.pkl', 'rb') as handle:
                    act_seq_l2_raw = pickle.load(handle)
                # Process L2 sequences using the same method as primary data
                self.act_seq_l2_processed = list(map(self.process_seq, act_seq_l2_raw))
                print(f"Loaded and processed {len(self.act_seq_l2_processed)} L2 sequences.")
            except FileNotFoundError:
                print(
                    f"Warning: L2 data file not found at {DATA_DIR}/level2_dataset_preprocessed.pkl. 'use_l2_data' is True.")
                self.act_seq_l2_processed = []
            except Exception as e:
                print(f"Warning: Error loading or processing L2 data: {e}")
                self.act_seq_l2_processed = []
        else:
            self.act_seq_l2_processed = []

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
        lx = list(x)  # Ensure x is a list of activities
        if self.opt.get("split_actions", False):
            lx = [i.split("_se_")[::-1] for i in lx]
            lx = [item for row in lx for item in row]

        processed_lx = []
        mask_prob = self.opt.get("mask_prob", 0.8)

        for item in lx:
            if item == "class_0":
                if random.random() < mask_prob:
                    processed_lx.append("<mask>")
                else:
                    processed_lx.append("<cls0>")
            elif item == "class_1":
                if random.random() < mask_prob:
                    processed_lx.append("<mask>")
                else:
                    processed_lx.append("<cls1>")
            else:
                processed_lx.append(item)

        out = ["<sos>"] + processed_lx + ["<eos>"]

        if self.opt.get("pad", False) and "bptt" in self.opt:
            if len(out) < self.opt["bptt"]:
                out.extend(["<pad>"] * (self.opt["bptt"] - len(out)))
            elif len(out) > self.opt["bptt"]:  # Truncate if longer than bptt
                out = out[:self.opt["bptt"] - 1] + ["<eos>"]

        return out

    def batchify_sequences(self, sequences, batch_size):
        """
        Raggruppa sequenze di lunghezza simile per ridurre il padding necessario.

        Args:
            sequences (list): Lista di sequenze da batchificare
            batch_size (int): Dimensione del batch

        Returns:
            list: Lista di batch tensori
        """
        if not sequences:
            return []

        # Ordina le sequenze per lunghezza
        seq_lengths = [len(seq) for seq in sequences]
        sorted_indices = sorted(range(len(sequences)), key=lambda i: seq_lengths[i])
        sorted_sequences = [sequences[idx] for idx in sorted_indices]

        batches = []
        for i in range(0, len(sorted_sequences), batch_size):
            batch = sorted_sequences[i:i + batch_size]

            # Usa il padding minimo necessario per questo batch specifico
            max_len_in_batch = max(len(seq) for seq in batch)

            # Crea batch con padding minimo
            padded_batch_tensors = []
            for seq in batch:
                # Converti in indici del vocabolario
                numerical_seq = self.vocab(seq)

                # Applica padding se necessario
                padding_needed = max_len_in_batch - len(numerical_seq)
                if padding_needed > 0:
                    numerical_seq.extend([self.vocab["<pad>"]] * padding_needed)

                padded_batch_tensors.append(torch.tensor(numerical_seq, dtype=torch.long))

            try:
                batch_tensor = torch.stack(padded_batch_tensors)
                batches.append(batch_tensor.to(self.opt.get("device", "cpu")))
            except Exception as e:
                print(f"Errore durante lo stacking del batch: {e}")
                continue  # Salta il batch problematico

        return batches
    # def batchify_sequences(self, sequences, batch_size):
    #     if not sequences:  # Handle empty list of sequences
    #         return []
    #     batches = []
    #     for i in range(0, len(sequences), batch_size):
    #         batch = sequences[i:i + batch_size]
    #         try:
    #             # Stack per ottenere un tensore (batch_size, seq_len)
    #             batch_tensor = torch.stack([torch.tensor(self.vocab(seq), dtype=torch.long) for seq in batch])
    #             batches.append(batch_tensor.to(self.opt.get("device", "cpu")))
    #         except RuntimeError as e:
    #             print(
    #                 f"Error stacking batch starting at index {i}. Sequences might have varying lengths after processing, despite padding attempts.")
    #             print(f"Details: {e}")
    #             # Optionally, inspect sequences in `batch` here
    #             # for s_idx, s_val in enumerate(batch):
    #             #     print(f"Sequence {s_idx} in problematic batch, length: {len(s_val)}")
    #             #     print(s_val)
    #
    #             # Fallback: pad to max length in current batch if padding was meant to make them equal
    #             if self.opt.get("pad", False) and "bptt" in self.opt:  # If global padding was intended
    #                 max_len_in_batch = self.opt["bptt"]
    #             else:  # if no global padding, pad to max_len in this specific batch
    #                 max_len_in_batch = max(len(s) for s in batch)
    #
    #             padded_batch_numerical = []
    #             for seq in batch:
    #                 numerical_seq = self.vocab(seq)
    #                 padding_needed = max_len_in_batch - len(numerical_seq)
    #                 if padding_needed > 0:
    #                     numerical_seq.extend([self.vocab["<pad>"]] * padding_needed)
    #                 elif padding_needed < 0:  # Should not happen if global truncation is applied
    #                     numerical_seq = numerical_seq[:max_len_in_batch]
    #                 padded_batch_numerical.append(torch.tensor(numerical_seq, dtype=torch.long))
    #
    #             try:
    #                 batch_tensor = torch.stack(padded_batch_numerical)
    #                 batches.append(batch_tensor.to(self.opt.get("device", "cpu")))
    #                 print(
    #                     f"Successfully processed batch starting at index {i} with local dynamic padding to length {max_len_in_batch}.")
    #             except Exception as final_e:
    #                 print(
    #                     f"Failed to process batch starting at index {i} even after local dynamic padding attempts. Error: {final_e}")
    #                 # Skip this batch or raise error
    #                 continue  # Skipping problematic batch
    #     return batches

    def get_dataset(self, num_test_ex=100, vocab=None, debugging=False):
        # 1. Extract labels from raw sequences in the primary DataFrame
        primary_labels_list = []
        # Group by case_id and then extract activities to ensure correct order
        grouped_df = self.df.groupby("case_id")

        processed_primary_sequences = []

        for _, group in grouped_df:
            raw_activity_sequence = group.activity.to_list()

            # Extract label
            if "class_0" in raw_activity_sequence:
                primary_labels_list.append(0.0)
            elif "class_1" in raw_activity_sequence:
                primary_labels_list.append(1.0)
            else:
                primary_labels_list.append(float('nan'))

            # Process sequence
            processed_primary_sequences.append(self.process_seq(raw_activity_sequence))

        # 2. Combine with L2 data (if any)
        # self.act_seq_l2_processed are already processed sequences from __init__
        all_processed_sequences = list(processed_primary_sequences)  # Make a copy
        all_labels_list = list(primary_labels_list)  # Make a copy

        if self.act_seq_l2_processed:
            all_processed_sequences.extend(self.act_seq_l2_processed)
            # Add NaN labels for L2 sequences, as their classification labels are not explicitly loaded here.
            # This assumes L2 data is primarily for other tasks or NaN labels are handled downstream.
            num_l2_sequences = len(self.act_seq_l2_processed)
            all_labels_list.extend([float('nan')] * num_l2_sequences)
            print(
                f"Added {num_l2_sequences} L2 sequences. Total sequences: {len(all_processed_sequences)}. Total labels: {len(all_labels_list)}.")

        # 3. Convert labels to tensor
        all_labels_tensor = torch.tensor(all_labels_list, dtype=torch.float)

        # 4. Split sequences and labels
        self.num_test_ex = num_test_ex
        self.valid_split_idx = max(0, num_test_ex - (num_test_ex // 2))
        # Ensure valid_split_idx is not problematic if num_test_ex is 0 or 1
        if num_test_ex > 0 and self.valid_split_idx >= num_test_ex:
            self.valid_split_idx = num_test_ex - 1

        total_sequences = len(all_processed_sequences)

        # Adjust split points if num_test_ex is larger than available data
        actual_num_test_ex = min(self.num_test_ex, total_sequences)
        actual_valid_split_idx = min(self.valid_split_idx, actual_num_test_ex)

        train_act_seq = all_processed_sequences[actual_num_test_ex:]
        valid_act_seq = all_processed_sequences[:actual_valid_split_idx]
        test_act_seq = all_processed_sequences[actual_valid_split_idx:actual_num_test_ex]

        self.train_labels = all_labels_tensor[actual_num_test_ex:]
        self.valid_labels = all_labels_tensor[:actual_valid_split_idx]
        self.test_labels = all_labels_tensor[actual_valid_split_idx:actual_num_test_ex]

        print(f"Data split: Train ({len(train_act_seq)} sequences, {len(self.train_labels)} labels), "
              f"Valid ({len(valid_act_seq)} sequences, {len(self.valid_labels)} labels), "
              f"Test ({len(test_act_seq)} sequences, {len(self.test_labels)} labels)")

        # 5. Build or use vocabulary
        if vocab is None:
            # Vocab should be built from all sequences that will be processed
            def combined_token_iterator():
                for seq in all_processed_sequences:  # Iterate over all sequences (train, valid, test, incl. L2 if added)
                    yield seq  # seq is already tokenized list from process_seq

            self.vocab = build_vocab_from_iterator(
                combined_token_iterator(),
                specials=['<unk>', '<sos>', '<eos>', '<pad>', '<mask>', '<cls0>', '<cls1>'],
                min_freq=self.opt.get("vocab_min_freq", 1)  # Allow configuring min_freq
            )
            self.vocab.set_default_index(self.vocab['<unk>'])
            print(f"Built vocabulary with {len(self.vocab)} tokens.")
        else:
            self.vocab = vocab
            print(f"Using provided vocabulary with {len(self.vocab)} tokens.")

        if debugging:
            try:
                # Ensure DATA_DIR exists or is handled if not defined
                vocab_dir = DATA_DIR if 'DATA_DIR' in globals() and DATA_DIR else "."
                os.makedirs(vocab_dir, exist_ok=True)
                with open(os.path.join(vocab_dir, "vocab.txt"), "w", encoding="utf-8") as f:
                    for token_idx, token in enumerate(self.vocab.get_itos()):
                        f.write(f"{token}\t{self.vocab.get_stoi()[token]}\n")  # Also write index
                print(f"Vocabulary saved to {os.path.join(vocab_dir, 'vocab.txt')}")
            except Exception as e:
                print(f"Warning: Could not write vocab.txt for debugging. Error: {e}")

        # 6. Batchify data
        batch_size = self.opt.get("batch_size", 32)  # Default batch size if not in opt
        if batch_size <= 0:
            print(f"Warning: batch_size is {batch_size}. Setting to default 32.")
            batch_size = 32

        self.train_data = self.batchify_sequences(train_act_seq, batch_size=batch_size)
        self.valid_data = self.batchify_sequences(valid_act_seq, batch_size=batch_size)
        self.test_data = self.batchify_sequences(test_act_seq, batch_size=batch_size)

        print(f"Batchification complete: Train batches ({len(self.train_data)}), "
              f"Valid batches ({len(self.valid_data)}), Test batches ({len(self.test_data)})")

        return self.vocab, self.train_data, self.valid_data, self.test_data


    def data_process(self, raw_text_iter, vocab):  # This seems like an older way to process data for batchify
        """
        Processes raw text data into a concatenated tensor of token indices.
        Not directly used by get_dataset if batchify_sequences is the primary method.
        """
        data = [torch.tensor(vocab([item]), dtype=torch.long) for item in raw_text_iter]
        non_empty_tensors = list(filter(lambda t: t.numel() > 0, data))
        if not non_empty_tensors:
            # Allow returning empty tensor if no valid tokens, or handle as error based on requirements
            return torch.empty(0, dtype=torch.long)
            # raise ValueError("No valid tokens found for concatenation. Check your data and vocabulary.")
        return torch.cat(non_empty_tensors)

    def batchify(self, data: torch.Tensor, bsz: int):  # Older batchify, for flattened data
        """
        Divides the data into bsz separate sequences. (Traditional Language Modeling Style)
        Not directly used by get_dataset if batchify_sequences is the primary method.
        """
        if data.numel() == 0:  # Handle empty data tensor
            return torch.empty((0, bsz), device=self.opt.get("device", "cpu"))

        seq_len = data.size(0) // bsz
        if seq_len == 0 and data.size(0) > 0:  # Not enough data for even one item per batch sequence
            print(
                f"Warning: Not enough data ({data.size(0)} tokens) to form batches of length > 0 with bsz={bsz}. Returning empty tensor.")
            return torch.empty((0, bsz), device=self.opt.get("device", "cpu"))
        elif seq_len == 0 and data.size(0) == 0:  # No data at all
            return torch.empty((0, bsz), device=self.opt.get("device", "cpu"))

        data = data[:seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(self.opt.get("device", "cpu"))

    def get_batch_from_list(self, source_data_list, batch_idx):
        # source_data_list is a list of batch tensors, e.g., self.train_data
        if not source_data_list or batch_idx >= len(source_data_list):
            raise IndexError(
                f"batch_idx {batch_idx} is out of range for source_data_list with {len(source_data_list) if source_data_list else 0} batches.")

        data_batch = source_data_list[batch_idx]  # shape: (batch_size, seq_len)

        # For language modeling: target is the same sequence shifted by 1
        # Ensure seq_len is at least 2 to create a target.
        if data_batch.size(1) < 2:
            # Return empty tensors or handle as an error if sequences are too short for LM target
            # This can happen if bptt is very small or sequences are short
            # For now, let's return the data as is for data, and an empty tensor for target to avoid crash
            # The model consuming this needs to handle it.
            print(
                f"Warning: Sequence length ({data_batch.size(1)}) in batch {batch_idx} is less than 2. Cannot create shifted target.")
            target = torch.empty((data_batch.size(0) * 0), dtype=torch.long, device=data_batch.device)  # empty target
            return data_batch, target  # return original data, empty target

        target = data_batch[:, 1:].contiguous().view(-1)  # shape: (batch_size * (seq_len-1))
        data = data_batch[:, :-1].contiguous()  # shape: (batch_size, seq_len-1)
        return data, target

    def get_labels_for_batch_type(self, data_type: str, batch_idx: int):
        # Selects the correct list of batch tensors and the corresponding full labels tensor
        if data_type == "train":
            data_batches_list = self.train_data
            labels_tensor_for_split = self.train_labels
        elif data_type == "valid":
            data_batches_list = self.valid_data
            labels_tensor_for_split = self.valid_labels
        elif data_type == "test":
            data_batches_list = self.test_data
            labels_tensor_for_split = self.test_labels
        else:
            raise ValueError(f"Invalid data_type specified: {data_type}")

        if not data_batches_list or batch_idx >= len(data_batches_list):
            raise IndexError(
                f"batch_idx {batch_idx} out of range for {data_type} data which has {len(data_batches_list) if data_batches_list else 0} batches.")

        start_idx_in_split = sum(data_batches_list[i].shape[0] for i in range(batch_idx))
        current_batch_actual_size = data_batches_list[batch_idx].shape[0]
        end_idx_in_split = start_idx_in_split + current_batch_actual_size

        if end_idx_in_split > len(labels_tensor_for_split):
            raise IndexError(
                f"Calculated end_idx {end_idx_in_split} is out of bounds for {data_type}_labels tensor with length {len(labels_tensor_for_split)}."
                f"Batch index: {batch_idx}, start_idx_in_split: {start_idx_in_split}, current_batch_size: {current_batch_actual_size}")

        return labels_tensor_for_split[start_idx_in_split:end_idx_in_split].to(self.opt.get("device", "cpu"))

    def get_masked_batch_labels(self, data_type: str, batch_idx: int, mask_positions: torch.Tensor):
        labels_for_current_batch = self.get_labels_for_batch_type(data_type, batch_idx)

        if mask_positions.shape[0] != labels_for_current_batch.shape[0]:
            raise ValueError(
                f"Mismatch in batch size between mask_positions ({mask_positions.shape[0]}) "
                f"and labels_for_current_batch ({labels_for_current_batch.shape[0]}) "
                f"for data_type '{data_type}', batch_idx {batch_idx}."
            )

        has_mask_in_sequence = mask_positions.any(dim=1)
        masked_labels = labels_for_current_batch[has_mask_in_sequence]
        return masked_labels, has_mask_in_sequence

    def preprocess_trace(self, trace: list):  # Added type hint for trace
        """
        Preprocess a single trace for classification/inference.
        Args:
            trace (list): List of event strings in the trace.
        Returns:
            torch.Tensor: Preprocessed trace tensor of indices.
        """
        if not self.vocab:
            raise RuntimeError("Vocabulary not built or loaded. Call get_dataset() first.")

        # Apply the same processing as in process_seq (sos, eos, masking if applicable for inference context)
        # For simplicity, this example assumes trace is already somewhat pre-tokenized.
        # A more robust version would apply parts of process_seq logic here.
        # For now, just add <sos>, <eos> and convert to indices.
        # This does not apply padding or masking used during training by default.
        processed_trace_tokens = ["<sos>"] + trace + ["<eos>"]

        # Optional: Apply padding/truncation if model expects fixed length input during inference
        if self.opt.get("pad", False) and "bptt" in self.opt:
            bptt_len = self.opt["bptt"]
            if len(processed_trace_tokens) < bptt_len:
                processed_trace_tokens.extend([self.vocab["<pad>"]] * (bptt_len - len(processed_trace_tokens)))
            elif len(processed_trace_tokens) > bptt_len:
                processed_trace_tokens = processed_trace_tokens[:bptt_len - 1] + [self.vocab["<eos>"]]

        indices = [self.vocab.get(event, self.vocab["<unk>"]) for event in processed_trace_tokens]
        return torch.tensor(indices, dtype=torch.long).to(self.opt.get("device", "cpu"))