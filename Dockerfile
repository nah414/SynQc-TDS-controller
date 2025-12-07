# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Copy only the necessary Python files
COPY synqc_tds_super_backend.py .
COPY synqc_agent.py .
COPY synqc_control_panel.html .
COPY ui_static/ ./ui_static/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      fastapi \
      uvicorn[standard] \
      pydantic \
      numpy \
      python-dotenv \
      orjson \
      openai

# Create state directory
RUN mkdir -p /app/synqc_state

# Health check
HEALTHCHECK --interval=5s --timeout=3s --start-period=2s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/synqc/health').read()" || exit 1

# Expose port
EXPOSE 8000

# Run backend with uvicorn
CMD ["uvicorn", "synqc_tds_super_backend:app", "--host", "0.0.0.0", "--port", "8000"]
