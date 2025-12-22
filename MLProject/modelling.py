import argparse
import os

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

import mlflow
import mlflow.sklearn

from pathlib import Path
import mlflow

with mlflow.start_run(run_name=args.run_name) as run:
    out_dir = Path("ci_outputs")  # atau args.out_dir kalau ada
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_id.txt").write_text(run.info.run_id)

def load_split(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))

    # aman kalau CSV y punya kolom "target" atau cuma 1 kolom
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_test_df  = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    y_train = y_train_df["target"] if "target" in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test  = y_test_df["target"] if "target" in y_test_df.columns else y_test_df.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="namadataset_preprocessing",
                        help="Folder berisi X_train/X_test/y_train/y_test")
    parser.add_argument("--experiment-name", default="HeartDisease_Basic", help="Nama experiment di MLflow")
    parser.add_argument("--run-name", default="logreg_autolog", help="Nama run")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    mlflow.set_experiment(args.experiment_name)

    mlflow.sklearn.autolog(log_models=True)
    model = LogisticRegression(max_iter=2000)

    with mlflow.start_run(run_name=args.run_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        else:
            auc = None

        mlflow.log_metric("test_accuracy_manual", acc)
        mlflow.log_metric("test_f1_manual", f1)
        if auc is not None:
            mlflow.log_metric("test_auc_manual", auc)

        print("✅ Training selesai")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}" + (f" | AUC: {auc:.4f}" if auc is not None else ""))


if __name__ == "__main__":
    main()
