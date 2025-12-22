import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn


def load_split(data_dir: str):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))

    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    y_test_df  = pd.read_csv(os.path.join(data_dir, "y_test.csv"))

    y_train = y_train_df["target"] if "target" in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test  = y_test_df["target"] if "target" in y_test_df.columns else y_test_df.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="heart_preprocessed",
        help="Folder berisi X_train/X_test/y_train/y_test",
    )
    parser.add_argument("--experiment-name", default="HeartDisease_CI", help="Nama experiment di MLflow")
    parser.add_argument("--run-name", default="ci_train", help="Nama run")
    parser.add_argument(
        "--out-dir",
        default="ci_outputs",
        help="Folder output untuk menyimpan run_id.txt dan ringkasan metrik",
    )
    args = parser.parse_args()

    # Pastikan out_dir ada (MLflow path param butuh existing path di beberapa skenario)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    # Set experiment
    mlflow.set_experiment(args.experiment_name)

    # (opsional) autolog untuk sklearn
    mlflow.sklearn.autolog(log_models=False)  # kita log model manual ke path "model"

    model = LogisticRegression(max_iter=2000)

    with mlflow.start_run(run_name=args.run_name) as run:
        # Train
        model.fit(X_train, y_train)

        # Predict + metrics
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = None

        # Log metrics (manual)
        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1", float(f1))
        if auc is not None:
            mlflow.log_metric("test_auc", float(auc))

        # Log model ke artifact_path="model" (PENTING untuk docker build: runs:/<run_id>/model)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Simpan run_id untuk CI
        (out_dir / "run_id.txt").write_text(run.info.run_id)

        # Simpan ringkasan metrik juga (opsional)
        summary_lines = [
            f"run_id={run.info.run_id}",
            f"test_accuracy={acc:.6f}",
            f"test_f1={f1:.6f}",
        ]
        if auc is not None:
            summary_lines.append(f"test_auc={auc:.6f}")
        (out_dir / "metrics_summary.txt").write_text("\n".join(summary_lines))

        print("✅ Training selesai")
        print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
