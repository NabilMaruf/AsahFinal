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
    X_test = pd.read_csv(base / "X_test.csv")

    y_train_df = pd.read_csv(base / "y_train.csv")
    y_test_df = pd.read_csv(base / "y_test.csv")

    y_train = y_train_df["target"] if "target" in y_train_df.columns else y_train_df.iloc[:, 0]
    y_test = y_test_df["target"] if "target" in y_test_df.columns else y_test_df.iloc[:, 0]

    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="heart_preprocessed")
    parser.add_argument("--experiment-name", default="HeartDisease_CI")
    parser.add_argument("--run-name", default="ci-retrain")
    args = parser.parse_args()

    # tracking uri dari env CI (kalau ada)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")

    # set experiment (boleh, aman)
    mlflow.set_experiment(args.experiment_name)

    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    model = LogisticRegression(max_iter=2000)

    # ✅ IMPORTANT:
    # Kalau dijalankan lewat `mlflow run`, MLflow Projects sudah bikin active run.
    # Jadi kita pakai active run kalau ada, kalau tidak ada baru start_run.
    active = mlflow.active_run()
    started_here = False
    if active is None:
        run = mlflow.start_run(run_name=args.run_name)
        started_here = True
    else:
        run = active

    try:
        # simpan run_id buat CI (build docker)
        out_dir = Path("ci_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_id.txt").write_text(run.info.run_id, encoding="utf-8")

        # train
        model.fit(X_train, y_train)

        # eval
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 2000)

        mlflow.log_metric("test_accuracy", float(acc))
        mlflow.log_metric("test_f1", float(f1))

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                mlflow.log_metric("test_roc_auc", float(auc))
            except Exception:
                pass

        # log model untuk build-docker (path konsisten)
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
                    "numpy==1.26.4",
                    "pyarrow==14.0.2",
                ]},
            ],
        }

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_py310",
            conda_env=conda_env,
        )

        print(f"✅ DONE | run_id={run.info.run_id} | acc={acc:.4f} | f1={f1:.4f}")

    finally:
        if started_here:
            mlflow.end_run()


if __name__ == "__main__":
    main()
