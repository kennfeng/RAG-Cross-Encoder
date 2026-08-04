#!/bin/sh
set -e
python scripts/pre_pull.py
python -m scripts.seed_db
exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-8000}"
