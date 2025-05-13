import optuna
import argparse
import os

from proformer import run_my_proformer


def objective(trial):
    # Definizione iniziale dei parametri (senza 'nhead' per ora)
    params = {
        "device": "cuda:0",
        "epochs": 100,
        "bptt": 34,
        "split_actions": True,
        "pad": True,
        "use_l2_data": False,
        "test_split_size": 1000,
        "pos_enc_dropout": 0.01,
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128]),
        "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128, 256]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, log=True),
        "d_hid": trial.suggest_categorical("d_hid", [64, 128, 256, 512, 1024]),
        "lr": trial.suggest_float("lr", 1e-3, 3e-3, log=True),
        "gamma_scheduler": trial.suggest_float("gamma_scheduler", 0.85, 0.99, log=True),
        "use_taxonomy": False,
        "use_pe": True,
        "taxonomy_emb_type": trial.suggest_categorical("taxonomy_emb_type", ["laplacian", "deepwalk"]),
        "taxonomy_emb_size": trial.suggest_categorical("taxonomy_emb_size", [8, 16, 32]),
        "weight_decay": 0.0,
        "gradient_clip": 0.5,
        "early_stopping_patience": 15,
        "early_stopping_min_delta": 0.0001,
        "dataset": "../data/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv"
    }

    # Estrai d_model per controllare i valori validi di nhead
    d_model = params["d_model"]

    # Lista di nhead validi (devono dividere d_model)
    valid_nheads = [h for h in [2, 4, 8, 16, 32] if d_model % h == 0]

    # Se non ci sono valori validi, salta questa trial
    if not valid_nheads:
        raise optuna.TrialPruned()

    # Usa un nome unico per evitare conflitti tra combinazioni diverse
    nhead_name = f"nhead__{d_model}"
    params["nhead"] = trial.suggest_categorical(nhead_name, valid_nheads)

    # Aggiungi nlayers dopo aver definito nhead
    params["nlayers"] = trial.suggest_int("nlayers", 1, 5, log=True)

    # Esegui il training del modello
    best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics = run_my_proformer.main(
        opt=params)

    # Memorizza ulteriori informazioni utili nella trial
    trial.set_user_attr("acc-3", float(best_valid_accs[3]))
    trial.set_user_attr("train_loss", float(best_train_loss))
    trial.set_user_attr("valid_loss", float(best_valid_loss))
    trial.set_user_attr("best_epoch", int(best_epoch))

    # Restituisce la metrica da ottimizzare
    return best_valid_accs[1]

def save_best_params_as_py(best_params, filename):
    """
    Saves the best parameters as a Python dictionary in a .py file.
    """
    with open(filename, "w") as f:
        f.write("# Optuna best parameters\n")
        f.write("optuna_best_params = ")
        f.write(repr(best_params))
        f.write("\n")

def main(opt):
    """
    Main function to run Optuna study, save results and best parameters.
    """
    study_name = opt["study_name"]
    n_trials = opt["ntrials"]
    studies_dir = "studies"

    # Ensure the studies directory exists
    os.makedirs(studies_dir, exist_ok=True)

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=f"sqlite:///{studies_dir}/{study_name}.sqlite3",
        load_if_exists=True
    )

    study.optimize(objective, n_trials=n_trials)

    print(study.best_trial)
    df = study.trials_dataframe()
    df.to_csv(f"{studies_dir}/{study_name}.csv", index=False)

    # Save best parameters as a Python dictionary
    save_best_params_as_py(study.best_trial.params, f"{studies_dir}/optuna_params.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for Proformer model.")
    parser.add_argument("--study_name", type=str, default="proformer", help="Name of the Optuna study.")
    parser.add_argument("--ntrials", type=int, default=100, help="Number of Optuna trials.")

    args = parser.parse_args()
    opt = vars(args)

    main(opt)