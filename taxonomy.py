import os
import pandas as pd
import torch.nn.functional as F
import torch
import pickle
import math
import networkx as nx
from torch_geometric.utils import from_networkx
from torch_geometric.transforms import AddRandomWalkPE, AddLaplacianEigenvectorPE


class Taxonomy():
    """
    Represents a taxonomy based on a pre-computed dictionary of paths between tokens.

    This class is responsible for loading taxonomy information from a file,
    merging it with a vocabulary, and computing weights for each token based on its
    relationships within the taxonomy.

    Attributes:
        toks (list): List of tokens in the vocabulary.
        paths (dict): Dictionary of paths between tokens, loaded from the input file.
        tax_weights (torch.Tensor): Tensor containing the computed taxonomy weights for each token.

    Args:
        filename (str): The path to the file containing the pre-computed paths (pickle file).
        vocab (torchtext.vocab.Vocab): The vocabulary object containing the tokens.
        device (torch.device): The device to store the taxonomy weights on (e.g., 'cpu' or 'cuda').
    """
    def __init__(self, filename, vocab, device):
        """
        Initializes the Taxonomy object.
        """
        with open(filename, 'rb') as handle:
                paths = pickle.load(handle)
        self.toks = vocab.get_itos()
        self.paths = self.merge_dicts(paths, self.toks)
        self.tax_weights = self.get_all_weights(self.paths, self.toks).to(device)


    def merge_dicts(self, paths, toks):
        """
        Merges the loaded paths dictionary with the vocabulary tokens.

        This function ensures that every token in the vocabulary has an entry in the paths dictionary.
        If a token is not found in the original paths, a default path with a weight of 20 for all tokens is assigned.

        Args:
            paths (dict): The dictionary of paths loaded from the input file.
            toks (list): The list of tokens in the vocabulary.

        Returns:
            dict: A new dictionary containing paths for all tokens in the vocabulary.
        """
        new_dict = {}
        sub_dict = {k: 20 for k in toks}
        for t in toks:
            if (t in paths.keys()):
                nd = {}
                for k in set(toks):
                    if k in paths.keys():
                        nd[k] = paths[t][k]
                    else:
                        nd[k] = 20
                new_dict[t] = nd
            else:
                new_dict[t] = sub_dict
        return new_dict


    def get_norm_path(self, p):
        """
        Normalizes a path (dictionary of weights) using the softmax function.

        Args:
            p (dict): A dictionary representing a path, where keys are tokens and values are weights.

        Returns:
            torch.Tensor: A tensor containing the normalized weights for the path.
        """
        #np = F.softmin(torch.tensor(list(p.values())).float(), dim=-1)
        np = torch.tensor(list(p.values())).long()
        np = np / math.sqrt(np.size(0))
        np = F.softmin(np, dim=-1)

        return np


    def get_all_weights(self, paths, toks):
        """
        Computes the taxonomy weights for all tokens in the vocabulary.

        Args:
            paths (dict): A dictionary containing paths for all tokens in the vocabulary.
            toks (list): The list of tokens in the vocabulary.

        Returns:
            torch.Tensor: A tensor containing the computed taxonomy weights for each token.
        """
        tax_weights = []
        for i, k in enumerate(toks):
            try:
                tax_weights.append(self.get_norm_path(paths[k]))
            except Exception as e:
                print(e)

        tax_weights = torch.vstack(tax_weights)
        return tax_weights


    def get_weights(self, src):
        """
        Retrieves the taxonomy weights for a given source token.

        Args:
            src (int): The index of the source token in the vocabulary.

        Returns:
            torch.Tensor: The taxonomy weights for the source token.
        """
        return self.tax_weights[src]


class TaxonomyEmbedding():
    """
    Generates taxonomy embeddings based on a taxonomy graph.

    This class loads a taxonomy from a CSV file, constructs a graph representation,
    and then computes embeddings for each node in the graph using either Laplacian Eigenvector PE
    or DeepWalk PE.

    Attributes:
        vocab (torchtext.vocab.Vocab): The vocabulary object containing the tokens.
        opt (dict): A dictionary containing various options, including the embedding type and size.
        taxonomy_df (pandas.DataFrame): DataFrame containing the taxonomy information loaded from the input file.
        G (torch_geometric.data.Data): The graph representation of the taxonomy.
        embs (torch.Tensor): The computed taxonomy embeddings.

    Args:
        vocab (torchtext.vocab.Vocab): The vocabulary object containing the tokens.
        filename (str): The path to the CSV file containing the taxonomy data.
        opt (dict): A dictionary containing various options, including the embedding type and size, and device.
    """
    def __init__(self, vocab, filename, opt):
        """
        Initializes the TaxonomyEmbedding object.
        """
        self.vocab = vocab
        self.opt = opt
        self.taxonomy_df = self.load_taxonomy(filename)
        self.G = self.get_graph(self.taxonomy_df, vocab)

        if opt["taxonomy_emb_type"] == "laplacian":
            self.G = AddLaplacianEigenvectorPE(opt["taxonomy_emb_size"], attr_name="pe")(self.G)
        elif opt["taxonomy_emb_type"] == "deepwalk":
            self.G = AddRandomWalkPE(opt["taxonomy_emb_size"], attr_name="pe")(self.G)

        self.embs = self.G.pe.to(opt["device"])


    def load_taxonomy(self, filename):
        """
        Loads the taxonomy data from a CSV file into a pandas DataFrame.

        Args:
            filename (str): The path to the CSV file containing the taxonomy data.

        Returns:
            pandas.DataFrame: A DataFrame containing the taxonomy data.
        """
        df = pd.read_csv(filename, names=["dst", "src"])
        # df = df.drop("m", axis=1)
        # df["dst"] = df.dst.apply(lambda x: x.split("_se_")[0])
        # df["src"] = df.src.apply(lambda x: x.split("_se_")[0])
        return df

    def get_graph(self, df, vocab):
        """
        Constructs a graph representation of the taxonomy from a DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame containing the taxonomy data.
            vocab (torchtext.vocab.Vocab): The vocabulary object containing the tokens.

        Returns:
            torch_geometric.data.Data: A graph representation of the taxonomy.
        """
        toks = self.vocab.get_itos()

        g = nx.Graph()
        for i, t in enumerate(toks):
            g.add_node(t, ind=i, name=t)

        for t in df.src:
            if t not in g.nodes:
                g.add_node(t, ind=-1, name=t)

        for t in df.dst:
            if t not in g.nodes:
                g.add_node(t, ind=-1, name=t)

        for i,r in df.iterrows():
            g.add_edge(r.src, r.dst)

        return from_networkx(g).to(self.opt["device"])