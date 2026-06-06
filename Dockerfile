FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

RUN useradd --create-home --uid 10001 appworker
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --default-timeout=600 --resume-retries=5 \
        -r requirements.txt

COPY . /app
RUN mkdir -p /app/data/audit /app/data/logs /app/data/models /app/data/reports \
    && chown -R appworker:appworker /app
USER appworker

EXPOSE 8501

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -m src.main --mode health

CMD ["python", "-m", "src.main", "--mode", "trade"]
