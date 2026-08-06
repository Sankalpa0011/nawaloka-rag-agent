# Nawaloka Hospitals - Hybrid Agentic Assistant

A healthcare chatbot that answers patient questions by routing between two sources:

- **A relational database** - doctors, consultation fees, channeling schedules, lab test prices and health check packages (from the supplied `data.sql`)
- **A vector store** - pages scraped from [nawaloka.com](https://www.nawaloka.com), covering clinical centres, services, facilities and policies

Backend is FastAPI + LangGraph; frontend is a Next.js chat UI.

## Requirements

- Docker and Docker Compose
- One free-tier LLM API key: [Groq](https://console.groq.com/keys), [Google AI Studio](https://aistudio.google.com/apikey), or [OpenRouter](https://openrouter.ai/keys)

For running locally without Docker you also need Python 3.12 and Node.js 20+.

## Quick start (Docker Compose)

```bash
git clone <repo-url> && cd nawaloka-agent
cp .env.example .env          # add any one of GROQ_/GEMINI_/OPENROUTER_API_KEY
docker compose up -d --build
```

Open **http://localhost:3000**.

| Service | URL |
|---|---|
| Chat UI | http://localhost:3000 |
| Backend API docs | http://localhost:8001/docs |
| Backend health | http://localhost:8001/health |
| Qdrant dashboard (vector store) | http://localhost:6333/dashboard |
| Postgres (relational store) | `localhost:5433` - see [Inspecting the data stores](#inspecting-the-data-stores) |

Boot is fully automated - no manual scraping or seeding command required, regardless of what's already on disk:

1. Postgres initialises from `db/init/` on first start.
2. A one-shot `scraper` service checks `backend/data/raw/manifest.json`. Normally that cache is committed to git, so the service finds it and exits immediately. If it's missing - a fresh crawl, or you deleted `data/raw/` yourself - the service crawls `nawaloka.com` with Playwright and writes the cache before anything else starts.
3. Once the scraper finishes (or skips), `backend` starts and seeds the vector store from that cache. This step never touches the network either way.

Later starts skip both the crawl and the seed - a restart costs one existence check and one count query.

```bash
docker compose logs -f scraper backend   # watch the crawl (if any) and the seed
curl http://localhost:8001/health        # service status
```

API docs: **http://localhost:8001/docs**

> `backend/data/ontology/symptoms.yaml` is hand-authored config (symptom → specialty mappings) - nothing regenerates it, so keep it under version control. Everything else under `backend/data/` (`raw/`, `extracted/`, `chunks/`) is either the committed crawl cache or fully disposable output, rebuilt automatically by the scraper and the ingest pipeline.

## Running the backend locally

```bash
# Data services only
docker compose up -d postgres qdrant

# Backend
cd backend
python -m venv .venv
.venv\Scripts\activate                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --port 8001

# Frontend, in a second terminal
cd ../frontend && npm install && npm run dev
```

> **Windows note:** run `npm` from PowerShell or CMD, not Git Bash - npm's path resolution breaks there. Python works in either.

## Environment setup

Copy `.env.example` to `.env`. Every setting has a working default; only an API key is required.

```bash
# LLM providers - at least one key required
GROQ_API_KEY=
GEMINI_API_KEY=
OPENROUTER_API_KEY=
LLM_PROVIDER_ORDER=groq,gemini,openrouter

# Data stores
DATABASE_URL=postgresql://nawaloka:nawaloka_dev_pw@localhost:5433/nawaloka
DATABASE_RO_URL=postgresql://chatbot_ro:chatbot_ro_dev_pw@localhost:5433/nawaloka
QDRANT_URL=http://localhost:6333

# App
BACKEND_PORT=8001
CORS_ORIGINS=http://localhost:3000
```

See `.env.example` for the full list, including retrieval, routing and feature-flag settings.

> `gemini-2.x` is unavailable on new AI Studio accounts, and OpenRouter's free model list changes often. Run `python -m app.llm.check_models` to derive working model IDs for your key.

## Inspecting the data stores

With `docker compose up -d`, both stores are reachable on the host - useful for checking what actually got seeded.

**Qdrant (vector store).** Open **http://localhost:6333/dashboard** for the built-in web UI: browse collections, inspect a point's payload/vector, and run search queries by hand. The collections created by the ingest pipeline are:

| Collection | Content |
|---|---|
| `nawaloka_web` (`QDRANT_COLLECTION`) | Chunked pages scraped from nawaloka.com |
| `faq_curated` (`QDRANT_FAQ_COLLECTION`) | Hand-written FAQ answers |
| `faq_cache` (`QDRANT_CACHE_COLLECTION`) | Semantic cache of generated answers |
| `symptom_ontology` (`QDRANT_ONTOLOGY_COLLECTION`) | Symptom -> specialty lookup |

The REST API behind the dashboard is also queryable directly, e.g. `curl http://localhost:6333/collections/nawaloka_web` for point counts and index status.

**Postgres (relational store).** Exposed on the host at `localhost:5433` (mapped from the container's `5432` - see `POSTGRES_PORT` in `.env`). Connect with `psql`, or point a GUI client (DBeaver, pgAdmin, TablePlus, the VS Code PostgreSQL extension) at it:

```
Host:     localhost
Port:     5433
Database: nawaloka
User:     nawaloka            # owner - or chatbot_ro for the read-only role
Password: from POSTGRES_PASSWORD / CHATBOT_RO_PASSWORD in .env
```

```bash
# Via a local psql install
psql postgresql://nawaloka:nawaloka_dev_pw@localhost:5433/nawaloka

# Or without installing anything, through the running container
docker exec -it nawaloka-postgres psql -U nawaloka -d nawaloka
```

Both containers, along with `backend` and `frontend`, also show up in Docker Desktop's container list (or `docker compose ps`) with their port mappings, so you can confirm what's running and jump into a shell (`docker exec`) or logs from there.
