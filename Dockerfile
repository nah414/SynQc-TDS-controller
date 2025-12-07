FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (kept minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY synqc_tds ./synqc_tds
COPY synqc_tds_super_backend.py .
COPY static ./static

EXPOSE 8000

CMD ["uvicorn", "synqc_tds_super_backend:app", "--host", "0.0.0.0", "--port", "8000"]
