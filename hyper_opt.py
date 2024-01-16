import run_proformer
import pandas as pd
import optuna
import argparse

def objective(trial):

    params = {  "device": "cuda:0",
                "epochs": 100,
                "bptt": 34, 
                "split_actions": True, 
                "pad": True,
                "use_l2_data": False,
                "test_split_size": 1000,
                "pos_enc_dropout": 0.01,
                "batch_size": trial.suggest_categorical("batch_size", [2, 4, 8, 16]),
                "d_model": trial.suggest_categorical("d_model", [16, 32, 64, 128, 256]),
                "nhead": trial.suggest_categorical("nhead", [1, 2, 4, 8]),
                "nlayers": trial.suggest_int("nlayers", 1, 5, log=True),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5, log=True),
                "d_hid": trial.suggest_categorical("d_hid", [16, 32, 64, 128, 256]),
                "lr": trial.suggest_float("lr", 1e-3, 3e-3, log=True),
                "gamma_scheduler": trial.suggest_float("gamma_scheduler", 0.85, 0.99, log=True),
                "use_taxonomy": trial.suggest_categorical("use_taxonomy", [True, False]),
                "use_pe": trial.suggest_categorical("use_pe", [True, False]),
                "taxonomy_emb_type": trial.suggest_categorical("taxonomy_emb_type", ["laplacian", "deepwalk"]),
                "taxonomy_emb_size": trial.suggest_categorical("taxonomy_emb_size", [8, 16, 32])
            }
    
    train_loss, valid_loss, valid_accs, epoch, test_accs = run_proformer.main(params)

    trial.set_user_attr("acc-3",  valid_accs[3].item())
    trial.set_user_attr("train_loss",  train_loss)
    trial.set_user_attr("valid_loss",  valid_loss)
    trial.set_user_attr("best_epoch",  epoch)
    
    return valid_accs[1]

def main(opt):
    study_name = opt["study_name"]
    n_trials = opt["ntrials"]
    study = optuna.create_study(direction="maximize", 
                                study_name=study_name, 
                                storage=f"sqlite:///./studies/{study_name}.sqlite3", 
                                load_if_exists=1)
    
    study.optimize(objective, n_trials=n_trials)

    print(study.best_trial)
    df = study.trials_dataframe()
    df.to_csv(f"studies/{study_name}.csv", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--study_name", type=str, default="proformer")
    parser.add_argument("--ntrials", type=int, default=100)
    
    args = parser.parse_args()
    opt = vars(args)

    main(opt)
