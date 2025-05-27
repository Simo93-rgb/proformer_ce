from typing import List, Tuple, Dict, Any, Optional, Iterator, Union
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


class Dataloader:
    """
    Dataloader class for processing and managing sequences data for transformer models.
    Handles data preprocessing, vocabulary building, and batch creation.
    """

    def __init__(self, filename: str, opt: Dict[str, Any]):
        """
        Initialize the dataloader with configuration options.

        Args:
            filename: Path to the CSV file containing the data
            opt: Dictionary of configuration options
        """
        self.labels = None
        self.vocab = None
        self.test_data = None
        self.valid_data = None
        self.train_data = None
        self.opt = opt
        self.df = pd.read_csv(filename)
        print("Initial DataFrame sample:")
        print(self.df.head())

        self.num_test_ex = 0
        self.valid_split_idx = 0
        self.train_labels = torch.empty(0)
        self.valid_labels = torch.empty(0)
        self.test_labels = torch.empty(0)

        if self.opt.get("use_l2_data", False):
            try:
                with open(f'{DATA_DIR}/level2_dataset_preprocessed.pkl', 'rb') as handle:
                    act_seq_l2_raw = pickle.load(handle)
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

    def preprocessor(self, filename: str) -> pd.DataFrame:
        """
        Preprocess the raw CSV file by standardizing column names and data types.

        Args:
            filename: Path to the CSV file

        Returns:
            Preprocessed DataFrame
        """
        try:
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
        except Exception as e:
            print(f"Error preprocessing file {filename}: {e}")
            raise

    def process_seq(self, x: List[str]) -> List[str]:
        """
        Preprocesses a sequence of activities by applying optional splitting, masking,
        and padding/truncation based on the provided options.

        Args:
            x: A list of activity strings to process

        Returns:
            Processed sequence with special tokens added, optionally padded or truncated
        """
        if not isinstance(x, list):
            raise ValueError("Input x must be a non-empty list of activities.")

        lx = list(x)

        # Optionally split actions based on configuration
        if self.opt.get("split_actions", False):
            try:
                lx = [i.split("_se_")[::-1] for i in lx]
                lx = [item for row in lx for item in row]
            except Exception as e:
                print(f"Error splitting actions: {e}")
                # Continue with original list if splitting fails

        processed_lx = []
        mask_prob = self.opt.get("mask_prob", 0.8)

        # Process each token for masking
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

        # Add special tokens
        out = ["<sos>"] + processed_lx + ["<eos>"]

        return self._apply_bptt(out)

    def batchify_sequences(self, sequences: List[List[str]], batch_size: int) -> List[torch.Tensor]:
        """
        Groups sequences of similar length to minimize required padding.

        Args:
            sequences: List of sequences to process into batches
            batch_size: Size of each batch

        Returns:
            List of tensor batches ready for model processing
        """
        if not sequences:
            return []

        try:
            # Sort sequences by length for more efficient padding
            seq_lengths = [len(seq) for seq in sequences]
            sorted_indices = sorted(range(len(sequences)), key=lambda i: seq_lengths[i])
            sorted_sequences = [sequences[idx] for idx in sorted_indices]

            batches = []
            for i in range(0, len(sorted_sequences), batch_size):
                batch = sorted_sequences[i:i + batch_size]

                # Use minimum necessary padding for this specific batch
                max_len_in_batch = max(len(seq) for seq in batch)

                # Create batch with minimum padding
                padded_batch_tensors = []
                for seq in batch:
                    # Convert to vocabulary indices
                    numerical_seq = self.vocab(seq)

                    # Apply padding if necessary
                    padding_needed = max_len_in_batch - len(numerical_seq)
                    if padding_needed > 0:
                        numerical_seq.extend([self.vocab["<pad>"]] * padding_needed)

                    padded_batch_tensors.append(torch.tensor(numerical_seq, dtype=torch.long))

                batch_tensor = torch.stack(padded_batch_tensors)
                batches.append(batch_tensor.to(self.opt.get("device", "cpu")))

            return batches
        except Exception as e:
            print(f"Error creating batches: {e}")
            return []

    def get_dataset(self, num_test_ex: int = 100, vocab: Optional[torchtext.vocab.Vocab] = None,
                    debugging: bool = False) -> Tuple[torchtext.vocab.Vocab, List[torch.Tensor],
    List[torch.Tensor], List[torch.Tensor]]:
        """
        Processes raw data into train, validation and test sets with appropriate labels.

        Args:
            num_test_ex: Number of examples to use for testing
            vocab: Optional pre-built vocabulary to use
            debugging: Whether to save vocabulary to file for debugging

        Returns:
            Tuple containing (vocabulary, train_data, valid_data, test_data)
        """
        try:
            # 1. Extract labels from raw sequences in the primary DataFrame
            primary_labels_list = []
            # Group by case_id and extract activities to ensure correct order
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
            all_processed_sequences = list(processed_primary_sequences)
            all_labels_list = list(primary_labels_list)

            if self.act_seq_l2_processed:
                all_processed_sequences.extend(self.act_seq_l2_processed)
                # Add NaN labels for L2 sequences
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
                # Build vocabulary from all sequences
                def combined_token_iterator() -> Iterator[List[str]]:
                    for seq in all_processed_sequences:
                        yield seq

                self.vocab = build_vocab_from_iterator(
                    combined_token_iterator(),
                    specials=['<unk>', '<sos>', '<eos>', '<pad>', '<mask>', '<cls0>', '<cls1>'],
                    min_freq=self.opt.get("vocab_min_freq", 1)
                )
                self.vocab.set_default_index(self.vocab['<unk>'])
                print(f"Built vocabulary with {len(self.vocab)} tokens.")
            else:
                self.vocab = vocab
                print(f"Using provided vocabulary with {len(self.vocab)} tokens.")

            if debugging:
                self._save_vocabulary_for_debugging()

            # 6. Batchify data
            batch_size = self.opt.get("batch_size", 32)
            if batch_size <= 0:
                print(f"Warning: batch_size is {batch_size}. Setting to default 32.")
                batch_size = 32

            self.train_data = self.batchify_sequences(train_act_seq, batch_size=batch_size)
            self.valid_data = self.batchify_sequences(valid_act_seq, batch_size=batch_size)
            self.test_data = self.batchify_sequences(test_act_seq, batch_size=batch_size)

            print(f"Batchification complete: Train batches ({len(self.train_data)}), "
                  f"Valid batches ({len(self.valid_data)}), Test batches ({len(self.test_data)})")

            return self.vocab, self.train_data, self.valid_data, self.test_data

        except Exception as e:
            print(f"Error in get_dataset: {e}")
            # Return empty data structures as fallback
            if self.vocab is None and vocab is not None:
                self.vocab = vocab
            elif self.vocab is None:
                # Create minimal vocabulary with special tokens
                self.vocab = build_vocab_from_iterator(
                    [['<unk>', '<sos>', '<eos>', '<pad>', '<mask>', '<cls0>', '<cls1>']],
                    specials=['<unk>', '<sos>', '<eos>', '<pad>', '<mask>', '<cls0>', '<cls1>']
                )
                self.vocab.set_default_index(self.vocab['<unk>'])

            return self.vocab, [], [], []

    def _save_vocabulary_for_debugging(self) -> None:
        """
        Save the vocabulary to a text file for debugging purposes.
        """
        try:
            vocab_dir = DATA_DIR if 'DATA_DIR' in globals() and DATA_DIR else "."
            os.makedirs(vocab_dir, exist_ok=True)
            with open(os.path.join(vocab_dir, "vocab.txt"), "w", encoding="utf-8") as f:
                for token_idx, token in enumerate(self.vocab.get_itos()):
                    f.write(f"{token}\t{self.vocab.get_stoi()[token]}\n")
            print(f"Vocabulary saved to {os.path.join(vocab_dir, 'vocab.txt')}")
        except Exception as e:
            print(f"Warning: Could not write vocab.txt for debugging. Error: {e}")

    def get_batch_from_list(self, source_data_list: List[torch.Tensor],
                            batch_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a specific batch from the data list and prepares it for language modeling.

        Args:
            source_data_list: List of batch tensors
            batch_idx: Index of the batch to retrieve

        Returns:
            Tuple containing (input data, target data)
        """
        if not source_data_list or batch_idx >= len(source_data_list):
            raise IndexError(
                f"batch_idx {batch_idx} is out of range for source_data_list with {len(source_data_list) if source_data_list else 0} batches.")

        data_batch = source_data_list[batch_idx]  # shape: (batch_size, seq_len)

        # For language modeling: target is the same sequence shifted by 1
        if data_batch.size(1) < 2:
            print(
                f"Warning: Sequence length ({data_batch.size(1)}) in batch {batch_idx} is less than 2. Cannot create shifted target.")
            target = torch.empty(0, dtype=torch.long, device=data_batch.device)
            return data_batch, target

        # Create input/target pairs for language modeling
        target = data_batch[:, 1:].contiguous().view(-1)  # shape: (batch_size * (seq_len-1))
        data = data_batch[:, :-1].contiguous()  # shape: (batch_size, seq_len-1)
        return data, target

    def get_labels_for_batch_type(self, data_type: str, batch_idx: int) -> torch.Tensor:
        """
        Retrieves labels for a specific batch based on data type.

        Args:
            data_type: Type of data ("train", "valid", or "test")
            batch_idx: Index of the batch

        Returns:
            Tensor of labels for the specified batch
        """
        try:
            # Select appropriate data source
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

            # Calculate indices in the labels tensor
            start_idx_in_split = sum(data_batches_list[i].shape[0] for i in range(batch_idx))
            current_batch_actual_size = data_batches_list[batch_idx].shape[0]
            end_idx_in_split = start_idx_in_split + current_batch_actual_size

            if end_idx_in_split > len(labels_tensor_for_split):
                raise IndexError(
                    f"Calculated end_idx {end_idx_in_split} is out of bounds for {data_type}_labels tensor with length {len(labels_tensor_for_split)}.")

            return labels_tensor_for_split[start_idx_in_split:end_idx_in_split].to(self.opt.get("device", "cpu"))

        except Exception as e:
            print(f"Error retrieving labels for batch {batch_idx} of {data_type} data: {e}")
            return torch.tensor([], device=self.opt.get("device", "cpu"))

    def get_masked_batch_labels(self, data_type: str, batch_idx: int,
                                mask_positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves labels for sequences that contain mask tokens.

        Args:
            data_type: Type of data ("train", "valid", or "test")
            batch_idx: Index of the batch
            mask_positions: Boolean tensor indicating positions of mask tokens

        Returns:
            Tuple containing (labels for masked positions, boolean mask of sequences with masks)
        """
        try:
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

        except Exception as e:
            print(f"Error retrieving masked batch labels: {e}")
            device = self.opt.get("device", "cpu")
            return torch.tensor([], device=device), torch.tensor([], dtype=torch.bool, device=device)

    def preprocess_trace(self, trace: List[str]) -> torch.Tensor:
        """
        Preprocess a single trace for classification/inference.

        Args:
            trace: List of event strings in the trace

        Returns:
            Preprocessed trace tensor of indices
        """
        if not self.vocab:
            raise RuntimeError("Vocabulary not built or loaded. Call get_dataset() first.")

        try:
            # Add special tokens
            processed_trace_tokens = ["<sos>"] + trace + ["<eos>"]

            # Apply padding/truncation if configured
            processed_trace_tokens = self._apply_bptt(processed_trace_tokens)

            # Convert to indices
            indices = [self.vocab.get(event, self.vocab["<unk>"]) for event in processed_trace_tokens]
            return torch.tensor(indices, dtype=torch.long).to(self.opt.get("device", "cpu"))

        except Exception as e:
            print(f"Error preprocessing trace: {e}")
            # Return a minimal valid tensor
            return torch.tensor([self.vocab["<sos>"], self.vocab["<eos>"]],
                                dtype=torch.long).to(self.opt.get("device", "cpu"))

    def _apply_bptt(self, tokens: List[str]) -> List[str]:
        """
        Pad or truncate a token sequence to opt['bptt'] length.

        Args:
            tokens (List[str]): Input sequence of tokens, including <sos> and <eos>.

        Returns:
            List[str]: If sequence length < opt['bptt'], pads with '<pad>'
                       up to opt['bptt']. If length > opt['bptt'],
                       truncates to opt['bptt'] - 1 and appends '<eos>'.
        """
        if not (self.opt.get("pad", False) and "bptt" in self.opt):
            return tokens

        bptt = self.opt["bptt"]
        if len(tokens) < bptt:
            return tokens + ["<pad>"] * (bptt - len(tokens))
        return tokens[: bptt - 1] + ["<eos>"]