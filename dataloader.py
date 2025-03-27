import numpy as np 
import pandas as pd
import torch
import torch.nn.functional as F 
import os
import random
import pickle

import torchtext
from torchtext.vocab import build_vocab_from_iterator
torchtext.disable_torchtext_deprecation_warning()


# This code was ispired by the following example:
# https://github.com/pytorch/tutorials/blob/main/beginner_source/transformer_tutorial.py


class Dataloader():
    def __init__(self, filename, opt):
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
            with open('data/level2_dataset_preprocessed.pkl', 'rb') as handle:
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
    
        #lx = x.activity.to_list()
        lx = x
        if self.opt["split_actions"]:
            lx = [i.split("_se_")[::-1] for i in lx]
            lx = [item for row in lx for item in row]
        out = ["<sos>"]+lx+["<eos>"]

        if self.opt["pad"]:
            if(len(out) < self.opt["bptt"]):
                out = out+["<pad>" for _ in range(self.opt["bptt"] - len(out))]

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


    def get_dataset(self, num_test_ex=1000):

        act_seq = self.df.groupby("case_id").apply(lambda x : self.process_seq(x.activity.to_list()))
        act_seq = act_seq.to_list()
        random.shuffle(act_seq)

        if self.opt["use_l2_data"]:
            act_seq = act_seq+self.act_seq_l2

        train_act_seq = act_seq[num_test_ex:]
        #valid_act_seq = act_seq[:num_test_ex-500]
        valid_act_seq = act_seq[:num_test_ex-(num_test_ex//2)]

        # -- testset is currently NOT separate -- #
        # test_act_seq = act_seq[num_test_ex-500:num_test_ex]
        test_act_seq = act_seq[num_test_ex-(num_test_ex//2):num_test_ex]
        
        # print(len(act_seq), num_test_ex)
        # print(len(train_act_seq),len(valid_act_seq), len(test_act_seq))

        vocab = build_vocab_from_iterator(self.yield_tokens(), specials=['<unk>'])
        vocab.set_default_index(vocab['<unk>'])

        train_raw_data = [item for sublist in train_act_seq for item in sublist]
        valid_raw_data = [item for sublist in valid_act_seq for item in sublist]
        test_raw_data = [item for sublist in test_act_seq for item in sublist]

        # batch size param is not batchsize but is num_exmp / batch_size (inverse of batch size) - should be fixed
        train_data = self.batchify(self.data_process(train_raw_data, vocab), bsz=len(train_act_seq) // self.opt["batch_size"])
        valid_data = self.batchify(self.data_process(valid_raw_data, vocab), bsz=len(valid_act_seq) // self.opt["batch_size"])
        test_data = self.batchify(self.data_process(test_raw_data, vocab), bsz=len(test_act_seq) // self.opt["batch_size"])
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.vocab = vocab

        return vocab, train_data, valid_data, test_data
    
    def get_clf_dataset(self, vocab=None, num_test_ex=1000):

        act_seq = self.df.groupby("case_id").apply(lambda x : self.process_seq(x.activity.to_list()))
        act_seq = act_seq.to_list()
        
        # assumes that the class of the sequence is in a "class" column
        lab_seq = self.df["class"]
        all_data = list(zip(act_seq, lab_seq))
        random.shuffle(all_data)
        act_seq, lab_seq = zip(*all_data)


        if self.opt["use_l2_data"]:
            act_seq = act_seq+self.act_seq_l2

        train_act_seq = act_seq[num_test_ex:]
        valid_act_seq = act_seq[:num_test_ex-8]

        # -- testset is currently NOT separate -- #
        test_act_seq = act_seq[num_test_ex-8:num_test_ex]

        # -- Splitting classification labels -- #
        train_lab_seq = lab_seq[num_test_ex:]
        valid_lab_seq = lab_seq[:num_test_ex-8]
        test_lab_seq = lab_seq[num_test_ex-8:num_test_ex]
        
        # print(len(act_seq), num_test_ex)
        # print(len(train_act_seq),len(valid_act_seq), len(test_act_seq))

        if vocab is None:
            vocab = build_vocab_from_iterator(self.yield_tokens(), specials=['<unk>'])
            vocab.set_default_index(vocab['<unk>'])

        train_raw_data = [item for sublist in train_act_seq for item in sublist]
        valid_raw_data = [item for sublist in valid_act_seq for item in sublist]
        test_raw_data = [item for sublist in test_act_seq for item in sublist]

        # batch size param is not batchsize but is num_exmp / batch_size (inverse of batch size) - should be fixed
        train_data = self.batchify(self.data_process(train_raw_data, vocab), bsz=len(train_act_seq) // self.opt["batch_size"])
        valid_data = self.batchify(self.data_process(valid_raw_data, vocab), bsz=len(valid_act_seq) // self.opt["batch_size"])
        test_data = self.batchify(self.data_process(test_raw_data, vocab), bsz=len(test_act_seq) // self.opt["batch_size"])
        
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.vocab = vocab

        return vocab, train_data, valid_data, test_data,\
               train_lab_seq, valid_lab_seq, test_lab_seq
