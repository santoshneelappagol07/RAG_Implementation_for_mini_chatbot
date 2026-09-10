#!/bin/bash
# startup.sh — Gunicorn with Uvicorn workers for Azure App Service
# Azure sets the PORT environment variable (usually 8000 or 80)

PORT="${PORT:-8000}"
WORKERS="${WEB_CONCURRENCY:-2}"

echo "Starting PDF Chatbot on port $PORT with $WORKERS workers..."

exec gunicorn backend.main:app \
    --bind "0.0.0.0:$PORT" \
    --workers "$WORKERS" \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile - \
    --error-logfile - \
    --log-level info
