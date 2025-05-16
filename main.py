from proformer_utils.params import bpi_params
from proformer_utils.run_my_proformer import parse_params, main
from config import DATA_DIR

if __name__ == "__main__":
    opt = parse_params(bpi_params)
    opt["dataset"] = f"{DATA_DIR}/ALL_20DRG_2022_2023_CLASS_Duration_ricovero_dimissioni_LAST_17Jan2025_padded_edited.csv"

    best_train_loss, best_valid_loss, best_valid_accs, best_epoch, test_accs, test_cls_metrics = main(opt=opt)
    print(f"Best epoch: {best_epoch} \t loss: {best_valid_loss} \t best accs: {best_valid_accs}")
    if test_cls_metrics:
        print(f"Test classification metrics: Accuracy: {test_cls_metrics['accuracy']:.4f}, "
              f"F1: {test_cls_metrics['f1']:.4f}, "
              f"Precision: {test_cls_metrics['precision']:.4f}, "
              f"Recall: {test_cls_metrics['recall']:.4f}")