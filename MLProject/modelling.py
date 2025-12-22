import argparse
import os
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn


def load_split(data_dir: str):
    base = Path(data_dir)
    if not (base / "X_train.csv").exists():
        alt = Path(__file__).parent / data_dir
        if (alt / "X_train.csv").exists():
            base = alt

    X_train = pd.read_csv(base / "X_train.csv")
    X_test  = pd.read_csv(base / "X_test.csv")
    y_train_df = pd.read_csv(base / "y_train.csv")
    y_test_df  = pd.read_csv(base / "y_test.csv")

    y_train = y_train_df["target"] if "target" in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test  = y_test_df["target"]  if "target" in y_test_df.columns  else y_test_df.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="heart_preprocessed")
    parser.add_argument("--experiment-name", default="HeartDisease_CI")
    parser.add_argument("--run-name", default="ci-retrain")
    args = parser.parse_args()

    if os.getenv("MLFLOW_TRACKING_URI") is None:
        mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    mlflow.set_experiment(args.experiment_name)

    # 🔹 Autolog BOLEH, tapi jangan log model otomatis
    mlflow.sklearn.autolog(log_models=False)

    model = LogisticRegression(max_iter=2000)

    with mlflow.start_run(run_name=args.run_name):

        # 1️⃣ TRAIN
        model.fit(X_train, y_train)

        # 2️⃣ EVALUASI
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy_manual", float(acc))
        mlflow.log_metric("test_f1_manual", float(f1))

        # 3️⃣ CONDA ENV (DI SINI)
        conda_env = {
            "name": "mlflow-py310-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {
                    "pip": [
                        "mlflow==2.14.1",
                        "scikit-learn",
                        "pandas",
                        "numpy"
                    ]
                }
            ]
        }

        # 4️⃣ LOG MODEL (DI SINI)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_py310",
            conda_env=conda_env
        )

        print("✅ Training selesai")
        print(f"Accuracy={acc:.4f} | F1={f1:.4f}")


if __name__ == "__main__":
    main()
