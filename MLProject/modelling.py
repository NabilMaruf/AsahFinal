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
    Load preprocessed split data (X_train/X_test/y_train/y_test) from a folder.
    Works whether data_dir is absolute/relative, and also tries relative to this file.
    """
    base = Path(data_dir)

    # If passed path doesn't exist, try relative to this script directory.
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
    parser.add_argument("--data-dir", default="heart_preprocessed", help="Folder split preprocessed CSV")
    parser.add_argument("--experiment-name", default="HeartDisease_CI", help="MLflow experiment name")
    parser.add_argument("--run-name", default="ci-retrain", help="MLflow run name")
    args = parser.parse_args()

    # ✅ Use tracking URI from environment if provided by CI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # local fallback
        mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")

    mlflow.set_experiment(args.experiment_name)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    # Model
    model = LogisticRegression(max_iter=2000)

    with mlflow.start_run(run_name=args.run_name) as run:
        # ✅ save run id for CI step (build docker)
        out_dir = Path("ci_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        # --- train ---
        model.fit(X_train, y_train)

        # --- evaluate (manual) ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 2000)

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1", float(f1))

        # AUC (if available)
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                mlflow.log_metric("test_roc_auc", float(auc))
            except Exception:
                pass

        # ✅ conda_env for build-docker reproducibility (Python 3.10)
        conda_env = {
            "name": "mlflow-py310-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10",
                "pip",
                {"pip": [
                    "mlflow==2.16.2",
                    "scikit-learn",
                    "pandas",
                    "numpy",
                    "pyarrow==14.0.2",
                ]},
            ],
        }

        # ✅ log model to a stable artifact path used by CI build-docker
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_py310",
            conda_env=conda_env,
        )

        print(f"✅ DONE | run_id={run.info.run_id} | acc={acc:.4f} | f1={f1:.4f}")


if __name__ == "__main__":
    main()
