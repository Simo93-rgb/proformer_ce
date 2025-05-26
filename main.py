from proformer_utils.params import bpi_params
from proformer_utils.run_my_proformer import parse_params, main, save_attention_maps
from proformer_utils.dataloader import Dataloader
from config import DATA_DIR, MODELS_DIR
import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()  # Disattiva i warning di deprecazione
import matplotlib
matplotlib.use('Agg')  # Imposta un backend non interattivo prima di qualsiasi import di matplotlib



if __name__ == "__main__":
    opt = parse_params(bpi_params)
    # opt["dataset"] = "data/aggregated_case_tuple.csv"
    # opt["dataset"] = "data/aggregated_case_detailed.csv"
    opt["dataset"] = f"{DATA_DIR}/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv"



    print("Inizializzazione configurazione...")
    opt = parse_params(bpi_params)
    opt["dataset"] = f"{DATA_DIR}/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited_balanced.csv"

    if opt["train"]:
        print("Addestramento modello...")
        best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics = main(opt=opt)
        print(f"Best epoch: {best_epoch} \t loss: {best_valid_loss} \t best accs: {best_valid_accs}")
    
    print("Caricamento modello...")
    model = torch.load(f"{MODELS_DIR}/proformer-base.bin")
    
    print("Caricamento dataset...")
    loader = Dataloader(filename=opt["dataset"], opt=opt)
    loader.get_dataset(num_test_ex=opt["test_split_size"])

    print("Generazione mappe di attenzione...")
    save_attention_maps(model, loader, opt)
    print("Completato!")