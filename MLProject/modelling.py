import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn


def load_split(data_dir: str):
    """
    Baca X_train, X_test, y_train, y_test dari folder data_dir.
    y boleh punya kolom 'target' atau hanya 1 kolom.
    """
    base = Path(data_dir)

    # Jika user passing relatif (mis. "heart_preprocessed") tapi lokasi sebenarnya di samping file modelling.py
    if not (base / "X_train.csv").exists():
        alt = Path(__file__).parent / data_dir
        if (alt / "X_train.csv").exists():
            base = alt

    X_train = pd.read_csv(base / "X_train.csv")
    X_test = pd.read_csv(base / "X_test.csv")

    y_train_df = pd.read_csv(base / "y_train.csv")
    y_test_df = pd.read_csv(base / "y_test.csv")

    y_train = y_train_df["target"] if "target" in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test = y_test_df["target"] if "target" in y_test_df.columns else y_test_df.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="heart_preprocessed",
        help="Folder berisi X_train/X_test/y_train/y_test"
    )
    parser.add_argument("--experiment-name", default="HeartDisease_CI", help="Nama experiment di MLflow")
    parser.add_argument("--run-name", default="ci-retrain", help="Nama run di MLflow")
    args = parser.parse_args()

    # Pastikan MLflow tracking URI pakai folder lokal mlruns (di CI akan di-set lewat env juga)
    # Jika env sudah diset, ini tidak mengganggu.
    if os.getenv("MLFLOW_TRACKING_URI") is None:
        mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    mlflow.set_experiment(args.experiment_name)

    # Autolog untuk memenuhi bukti logging lengkap
    mlflow.sklearn.autolog(log_models=True)

    model = LogisticRegression(max_iter=2000)

    with mlflow.start_run(run_name=args.run_name):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        auc = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)

        # Manual metrics (tambahan bukti)
        mlflow.log_metric("test_accuracy_manual", float(acc))
        mlflow.log_metric("test_f1_manual", float(f1))
        if auc is not None:
            mlflow.log_metric("test_auc_manual", float(auc))

        # ✅ PENTING: Log model manual dengan pip_requirements
        # Ini supaya saat mlflow models build-docker tidak memakai python 3.8 lagi.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_py310",
            pip_requirements=[
                "mlflow==2.14.1",
                "scikit-learn",
                "pandas",
                "numpy"
            ]
        )

        print("✅ Training selesai")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}" + (f" | AUC: {auc:.4f}" if auc is not None else ""))


if __name__ == "__main__":
    main()
