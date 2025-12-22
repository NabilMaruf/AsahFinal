"""
modelling.py (Kriteria 3 - Advanced)
- Training + MLflow logging
- Simpan run_id ke file run_id.txt (dipakai build docker image)
"""

import argparse
import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn


def load_split(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze("columns")
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze("columns")
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="heart_preprocessed")
    parser.add_argument("--experiment-name", default="HeartDisease_CI")
    parser.add_argument("--run-name", default="ci_train")
    parser.add_argument("--out-dir", default="ci_outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    mlflow.set_experiment(args.experiment_name)
    mlflow.sklearn.autolog(log_models=True)

    model = LogisticRegression(max_iter=2000)

    with mlflow.start_run(run_name=args.run_name) as run:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)

        mlflow.log_metric("test_accuracy_manual", acc)
        mlflow.log_metric("test_f1_manual", f1)
        if auc is not None:
            mlflow.log_metric("test_auc_manual", auc)

        run_id = run.info.run_id
        with open(os.path.join(args.out_dir, "run_id.txt"), "w", encoding="utf-8") as f:
            f.write(run_id)

        with open(os.path.join(args.out_dir, "metrics_summary.txt"), "w", encoding="utf-8") as f:
            f.write(f"acc={acc}\nf1={f1}\nauc={auc}\n")

        mlflow.log_artifacts(args.out_dir, artifact_path="ci_outputs")

        print("✅ CI training selesai")
        print("run_id:", run_id)
        print(f"Accuracy={acc:.4f} | F1={f1:.4f}" + (f" | AUC={auc:.4f}" if auc is not None else ""))


if __name__ == "__main__":
    main()
