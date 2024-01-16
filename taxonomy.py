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
    def __init__(self, filename, vocab, device):
        with open(filename, 'rb') as handle:
                paths = pickle.load(handle)
        self.toks = vocab.get_itos()
        self.paths = self.merge_dicts(paths, self.toks)
        self.tax_weights = self.get_all_weights(self.paths, self.toks).to(device)
        

    def merge_dicts(self, paths, toks):
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
        #np = F.softmin(torch.tensor(list(p.values())).float(), dim=-1)
        np = torch.tensor(list(p.values())).long()
        np = np / math.sqrt(np.size(0))
        np = F.softmin(np, dim=-1)

        return np
    

    def get_all_weights(self, paths, toks):
        tax_weights = []
        for i, k in enumerate(toks):
            try:
                tax_weights.append(self.get_norm_path(paths[k]))
            except Exception as e:
                print(e)

        tax_weights = torch.vstack(tax_weights)
        return tax_weights
    

    def get_weights(self, src):
        return self.tax_weights[src]


class TaxonomyEmbedding():
    def __init__(self, vocab, filename, opt):
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
        df = pd.read_csv(filename, names=["dst", "src"])
        # df = df.drop("m", axis=1)
        # df["dst"] = df.dst.apply(lambda x: x.split("_se_")[0])
        # df["src"] = df.src.apply(lambda x: x.split("_se_")[0])
        return df
    
    def get_graph(self, df, vocab):
        toks = self.vocab.get_itos()

        g = nx.Graph()
        for i, t in enumerate(toks):
            g.add_node(t, ind=i, name=t)

        for t in df.src:
            if(t not in g.nodes):
                g.add_node(t, ind=-1, name=t)

        for t in df.dst:
            if(t not in g.nodes):
                g.add_node(t, ind=-1, name=t)

        for i,r in df.iterrows():
            g.add_edge(r.src, r.dst)
        
        return from_networkx(g).to(self.opt["device"])