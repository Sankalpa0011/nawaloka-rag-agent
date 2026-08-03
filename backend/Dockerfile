# Backend API image.
#
# Playwright/Chromium is deliberately NOT installed here. It is only needed to
# re-crawl nawaloka.com, and the rendered HTML is committed to data/raw, so the
# index rebuilds from cache without a browser. That keeps this image ~1 GB
# smaller. Run `python -m app.ingest.scrape` locally when you need a fresh crawl.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libpq for psycopg, then drop the toolchain again.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq5 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY data ./data
COPY evals ./evals
COPY pytest.ini .

# fastembed pulls its ONNX models on first use. Baking them in keeps the first
# request fast and lets the container run without outbound network access.
RUN python -c "from fastembed import TextEmbedding, SparseTextEmbedding; \
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding('Qdrant/bm25'); \
TextCrossEncoder('Xenova/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8001

# start-period is generous: the first boot builds the vector index from the
# cached HTML (~60 s) in the startup lifespan, before uvicorn binds the port.
HEALTHCHECK --interval=15s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -fsS http://localhost:8001/health || exit 1

# app.main's __main__ calls uvicorn.run() with the configured port, so there is
# no shell needed to expand BACKEND_PORT.
CMD ["python", "-m", "app.main"]
