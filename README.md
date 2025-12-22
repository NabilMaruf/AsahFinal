# Workflow-CI (Kriteria 3 - Advanced)

## Checklist yang dipenuhi
- ✅ Ada folder `MLProject`
- ✅ Ada workflow CI (GitHub Actions) untuk retraining otomatis
- ✅ Artefak CI disimpan (Actions Artifacts)
- ✅ Docker image dibangun pakai `mlflow models build-docker` dan di-push ke Docker Hub

## Secrets yang wajib ditambahkan
Repo GitHub → Settings → Secrets and variables → Actions:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

## Docker Hub
Image yang dipush:
`<DOCKERHUB_USERNAME>/heartdisease-mlflow:latest`

Tautan Docker Hub yang bisa kamu tulis di submission:
`https://hub.docker.com/r/<DOCKERHUB_USERNAME>/heartdisease-mlflow`

## Cara jalanin
Push ke branch `main` atau jalankan manual dari tab Actions.
