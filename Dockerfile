# ============================================================
# Multi-Stage Dockerfile for PDF Chatbot (FastAPI + React)
# Stage 1: Build React frontend → static files
# Stage 2: Python runtime → FastAPI serves API + static files
# ============================================================

# ── Stage 1: Build React Frontend ─────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files first for Docker layer caching
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --production=false

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python Runtime ───────────────────────────────────
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies required by faiss-cpu and other packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./backend/

# Copy built React frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create storage directories (ephemeral in container — use Azure Blob for persistence)
RUN mkdir -p storage/pdfs storage/vectors

# Copy startup script
COPY startup.sh .
RUN chmod +x startup.sh

# Expose port (Azure App Service sets PORT env var)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Start with Gunicorn + Uvicorn workers
CMD ["./startup.sh"]
