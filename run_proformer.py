import time
import math
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from dataloader import Dataloader
from proformer import TransformerModel
from params import bpi_params
from taxonomy import Taxonomy, TaxonomyEmbedding
import pickle

def parse_params():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--test_split_size", type=int, default=500, help="Number of examples to use for valid and test")
    parser.add_argument("--pad", action="store_true", help="Pads the sequences to bptt", default=True)
    parser.add_argument("--bptt", type=int, default=34, help="Max len of sequences")
    parser.add_argument("--split_actions", action="store_true", default=True, help="Splits multiple action if in one (uses .split('_se_'))")
    parser.add_argument("--batch_size", type=int, default=2, help="Regulates the batch size")
    parser.add_argument("--pos_enc_dropout", type=float, default=0.1, help="Regulates dropout in pe")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--nlayers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--d_hid", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3.)
    parser.add_argument("--gamma_scheduler", type=float, default=0.97)
    parser.add_argument("--use_l2_data", action="store_true", default=True, help="Uses data from level 2 dataset")
    parser.add_argument("--use_taxonomy", action="store_true", default=False, help="Introduces weights based on a taxonomy of the tokens")
    parser.add_argument("--use_pe", action="store_true", default=False)
    parser.add_argument("--taxonomy_emb_type", type=str, default="laplacian")
    parser.add_argument("--taxonomy_emb_size", type=int, default=16)


    args = parser.parse_args()
    opt = vars(args)

    return opt


# -- SLOW IMPLEMENTATION --
# def get_ranked_metrics(accs, out, t):
#     ks = list(accs.keys())
#     out = torch.softmax(out, dim=1).topk(max(ks), dim=1).indices
#     #out = out.topk(max(ks), dim=1).indices
#     print(out)
#     print(t)
#     for k in ks:
#         all = 0
#         for i, el in enumerate(out[:,:k]):
#             all+=(torch.isin(el, t[i]).max().int())
#         accs[k] += all / t.size(0)

#     return accs

def get_ranked_metrics(accs, out, t):
    ks = list(accs.keys())
    out = torch.softmax(out, dim=1).topk(max(ks), dim=1).indices
    # out = out.topk(max(ks), dim=1).indices
    all = []
    for i, el in enumerate(out[:,:max(ks)]):
        all.append(torch.isin(el, t[i]))
    
    all = torch.vstack(all)
    for k in ks:
        accs[k] += all[:,:k].int().sum() / t.size(0)

    return accs


def train(model, opt, loader, optimizer):
    
    model.train()
    total_loss = 0.
    log_interval = 200
    start_time = time.time()

    num_batches = len(loader.train_data) // opt["bptt"]

    for batch, i in enumerate(range(0, loader.train_data.size(0) - 1, opt["bptt"])):
        
        data, targets = loader.get_batch(loader.train_data, i)
        attn_mask = model.create_masked_attention_matrix(data.size(0)).to(opt["device"])

        output = model(data, attn_mask)
        output_flat = output.view(-1, model.ntokens)

        pad_mask = (targets != 1) & (targets != 8)
        targets = targets[pad_mask]
        output_flat = output_flat[pad_mask, :]
        
        weights = torch.ones(model.ntokens).to(opt["device"])
        
        loss = F.cross_entropy(output_flat, targets, weight=weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    
        total_loss += loss.item()

    return total_loss / (batch+1)


def evaluate(model, eval_data, loader, opt):
    model.eval() 
    total_loss = 0.
    accs = {1: 0., 3: 0., 5: 0.}

    with torch.no_grad():
        for batch,i in enumerate(range(0, eval_data.size(0) - 1, opt["bptt"])):

            data, targets = loader.get_batch(eval_data, i)
            attn_mask = model.create_masked_attention_matrix(data.size(0)).to(opt["device"])
            
            seq_len = data.size(0)
            output = model(data, attn_mask)
            
            output_flat = output.view(-1, model.ntokens)

            pad_mask = (targets != 1) & (targets != 8)
            targets = targets[pad_mask]
            output_flat = output_flat[pad_mask, :]
            
            total_loss += seq_len * F.cross_entropy(output_flat, targets).item()
            accs = get_ranked_metrics(accs, output_flat, targets)

        for k in accs.keys():
            accs[k] = accs[k] / (batch+1)
        loss = total_loss / (len(eval_data) - 1)
    
    return loss, accs


def main(opt):
    random.seed(123)

    if(opt == None):
        print("-- PARSING CMD ARGS --")
        opt = parse_params()
        for k in bpi_params.keys():
            opt[k] = bpi_params[k]
    print(opt)

    # -- Add optional params here --
    #
    # opt["d_model"] = 32
    # opt["d_hid"] = 32
    # opt["nlayers"] = 1
    # opt["nhead"] = 1
    # opt["use_l2_data"] = False
    # opt["test_split_size"] = 7000
    # opt["epochs"] = 100
    # opt["use_taxonomy"] = False
    # opt["use_pe"] = False
    # opt["bptt"] = 175
    # opt["lr"] = 3e-3
    # opt["taxonomy_emb_type"] = "laplacian"
    # opt["taxonomy_emb_size"] = 8
    # 
    # ------------------------------
    
    loader = Dataloader("data/BPI_Challenge_2012.csv", opt)
    loader.get_dataset(opt["test_split_size"])

    tax = TaxonomyEmbedding(loader.vocab, "data/bpi_taxonomy.csv", opt)
    
    model = TransformerModel(len(loader.vocab), opt, taxonomy=tax.embs).to(opt["device"])
    # model = TransformerModel(len(loader.vocab), opt).to(opt["device"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1., gamma=opt["gamma_scheduler"])
    best_val_acc = -float('inf')

    for epoch in range(1, opt["epochs"]+1):
        
        epoch_start_time = time.time()  
        
        train_loss = train(model, opt, loader, optimizer)
        valid_loss, valid_accs = evaluate(model, loader.valid_data, loader, opt)
        valid_ppl = math.exp(valid_loss)
        
        elapsed = time.time() - epoch_start_time

        if((epoch % 10) == 0):
            print('-' * 104)
            print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'valid loss {valid_loss:5.2f} | valid ppl {valid_ppl:7.2f} | '
                  f'acc@1 {valid_accs[1]:.4f} | '
                  f'acc@3 {valid_accs[3]:.4f} |')
            print('-' * 104)

        if (valid_accs[1] > best_val_acc):
            best_train_loss = train_loss
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_valid_accs = valid_accs
            best_val_acc = valid_accs[1]

            # -- execute eval on testset -- 
            test_loss, test_accs = evaluate(model, loader.test_data, loader, opt)
            test_ppl = math.exp(test_loss)
            print(f"| Performance on test: Test ppl: {test_ppl:5.2f} | "
                  f"test acc@1: {test_accs[1]:.4f} | test acc@3: {test_accs[3]:.4f}"+(" ")*23+"|")
            print("-"*104)
            torch.save(model, "models/proformer-base.bin")

        scheduler.step()
    
    return best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs

if __name__ == "__main__":
    best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs = main(opt=None)
    print(f"Best epoch: {best_epoch} \t loss: {best_valid_loss} \t best accs: {best_valid_accs}")