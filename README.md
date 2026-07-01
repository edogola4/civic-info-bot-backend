# Kenya Civic Info Bot — Backend

FastAPI service powering the Kenya Civic Info Bot. Handles PDF ingestion, vector embeddings, semantic search, and Claude-powered responses.

## Stack

- **FastAPI** — API framework
- **PyMuPDF** — Kenya Constitution PDF extraction
- **OpenAI** `text-embedding-3-small` — chunk embeddings
- **pgvector on Neon PostgreSQL** — vector similarity search
- **Anthropic Claude** `claude-3-5-haiku-20241022` — answer generation

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Server status + chunk count |
| `POST` | `/ingest` | Seed the vector DB (requires `x-api-key` header) |
| `POST` | `/chat` | Answer a civic question with sources |

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# fill in .env
uvicorn main:app --reload
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key for embeddings |
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude |
| `DATABASE_URL` | Neon PostgreSQL connection string with pgvector |
| `INGEST_API_KEY` | Secret key to protect the `/ingest` endpoint |

## Seeding the Database

Run once after deployment:

```bash
curl -X POST https://YOUR_RAILWAY_URL/ingest \
  -H "x-api-key: YOUR_INGEST_API_KEY"
```

Ingests the Kenya Constitution 2010 PDF (~450 chunks) + IEBC voter registration info + county government explainer.

## Deployment

Deployed on Railway. The `railway.toml` at the root configures the start command automatically.

Set all four environment variables in the Railway dashboard before deploying.
