import argparse
import os
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


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

    # tracking uri dari CI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"file:{Path.cwd() / 'mlruns'}")

    # MLflow Projects menyediakan RUN ID ini
    env_run_id = os.getenv("MLFLOW_RUN_ID")
    if not env_run_id:
        raise RuntimeError("MLFLOW_RUN_ID tidak ditemukan. Jalankan via `mlflow run .` di CI.")

    # simpan run_id buat step build docker
    out_dir = Path("ci_outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_id.txt").write_text(env_run_id, encoding="utf-8")

    # pastikan experiment ada (tidak bikin run baru)
    mlflow.set_experiment(args.experiment_name)

    # training
    X_train, X_test, y_train, y_test = load_split(args.data_dir)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            pass

    # ✅ log param/metric lewat client (tidak start_run)
    client = MlflowClient()

    client.log_param(env_run_id, "model_type", "LogisticRegression")
    client.log_param(env_run_id, "max_iter", "2000")
    client.log_param(env_run_id, "run_name", args.run_name)

    client.log_metric(env_run_id, "test_accuracy", float(acc))
    client.log_metric(env_run_id, "test_f1", float(f1))
    if auc is not None:
        client.log_metric(env_run_id, "test_roc_auc", float(auc))

    # ✅ log model sebagai artifact (ini tidak membuat run baru)
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

    # ini akan menyimpan artifact ke run_id yang aktif di env (Projects)
    # tapi agar benar-benar aman, kita set "run_id" lewat start_run? tidak perlu.
    # mlflow.sklearn.log_model akan pakai run aktif jika ada; kalau tidak ada, ia bisa coba start run.
    # Jadi kita pakai log_model + specify registered? Tidak ada.
    # Cara aman: gunakan mlflow.start_run(run_id=env_run_id) TANPA konflik dengan env (karena sama),
    # tapi MLflow tadi konflik. Maka kita log model sebagai artifact lokal lalu upload via client.

    import tempfile
    import shutil

    tmpdir = Path(tempfile.mkdtemp())
    local_model_dir = tmpdir / "model_py310"

    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(local_model_dir),
        conda_env=conda_env,
    )

    # upload seluruh folder model ke artifacts/model_py310
    client.log_artifacts(env_run_id, str(local_model_dir), artifact_path="model_py310")

    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"✅ DONE | run_id={env_run_id} | acc={acc:.4f} | f1={f1:.4f}")


if __name__ == "__main__":
    main()
