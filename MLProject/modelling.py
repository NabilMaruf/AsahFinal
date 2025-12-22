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
    y bisa punya kolom 'target' atau hanya 1 kolom.
    """
    base = Path(data_dir)

    # kalau path relatif yang dipassing ternyata ada di sebelah file ini
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
    parser.add_argument("--data-dir", default="heart_preprocessed", help="Folder berisi X_train/X_test/y_train/y_test")
    parser.add_argument("--experiment-name", default="HeartDisease_CI", help="Nama experiment MLflow")
    parser.add_argument("--run-name", default="ci-retrain", help="Nama run MLflow")
    args = parser.parse_args()

    # tracking lokal ke folder 'mlruns' di workspace (CI juga set env ini)
    if os.getenv("MLFLOW_TRACKING_URI") is None:
        mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")

    X_train, X_test, y_train, y_test = load_split(args.data_dir)

    mlflow.set_experiment(args.experiment_name)

    # Autolog boleh untuk param/metric otomatis, tapi jangan log model otomatis (biar model cuma 1: model_py310)
    mlflow.sklearn.autolog(log_models=False)

    model = LogisticRegression(max_iter=2000)

    with mlflow.start_run(run_name=args.run_name) as run:
        # simpan run_id agar workflow nggak perlu scan mlruns
        out_dir = Path("ci_outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_id.txt").write_text(run.info.run_id)

        # TRAIN
        model.fit(X_train, y_train)

        # PRED
        y_pred = model.predict(X_test)

        # METRICS
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy_manual", float(acc))
        mlflow.log_metric("test_f1_manual", float(f1))

        # AUC (kalau binary dan ada predict_proba)
        auc = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_proba)
                mlflow.log_metric("test_auc_manual", float(auc))
            except Exception:
                # kalau bentuk label/kelas tidak cocok, skip AUC
                pass

        # ✅ Log model dengan pip_requirements (lebih stabil untuk build-docker via virtualenv)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_py310",
            pip_requirements=[
                "mlflow==2.16.2",
                "scikit-learn",
                "pandas",
                "numpy",
                "pyarrow==14.0.2",
            ],
        )

        print("✅ Training selesai")
        msg = f"Accuracy={acc:.4f} | F1={f1:.4f}"
        if auc is not None:
            msg += f" | AUC={auc:.4f}"
        print(msg)


if __name__ == "__main__":
    main()
