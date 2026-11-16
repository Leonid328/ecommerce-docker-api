# Homework: FastAPI layer for ecommerce-start

## API

- `GET /health`
- `POST /predict`

## Docker

- `Dockerfile.api` is a multi-stage image based on `python:3.11-slim`.
- Compose service `api` maps `8000:8000`, bind-mounts `./models:/app/models`, and health-checks every 30 seconds.

## Run

```bash
python scripts/train_api_model.py
docker compose up -d --build
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"recency":12,"frequency":15,"monetary":240,"tenure_days":180}'
```
