# ── Builder stage ────────────────────────────────────────
FROM python:3.10-slim AS base

# System deps for Pillow / torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cache-friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create folders for uploads/outputs
RUN mkdir -p static/uploads static/outputs models

# Expose Flask port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
