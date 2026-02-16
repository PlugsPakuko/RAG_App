# RAG_App (ChromaDB + Ollama + FastAPI)

A minimal Retrieval-Augmented Generation (RAG) app that:

- Embeds `aboutme.txt` into a **ChromaDB** persistent vector store (`./chroma_db`)
- Exposes a **FastAPI** server with a `/query` endpoint
- Uses **Ollama** to generate answers grounded on retrieved documents

## Requirements

- Python (recommended: use the existing virtualenv in `rag_app/`)
- Ollama running locally (default: `http://localhost:11434`)

## 1) Embed documents into ChromaDB

This reads `aboutme.txt`, splits it by non-empty lines, and stores them in the ChromaDB collection `personal_info`.

```bash
python embed.py
```

It will create/update the local vector DB at `./chroma_db/`.

## 2) Start Ollama

Make sure Ollama is running and the model configured in `app.py` is available.

Then set the model in `app.py`:

- `OLLAMA_API_URL` (default): `http://localhost:11434/api/generate`
- `OLLAMA_MODEL`: change to your installed model (e.g. `llama3`)

## 3) Run the FastAPI server

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open:

- API root: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`

## API

### POST `/query`

Retrieves relevant docs from ChromaDB and asks Ollama to answer using that context.

Example:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is my name?","n_results":3}'
```

Response includes:

- `answer`
- `relevant_documents` (the retrieved context lines)

### GET `/health`

Quick check that the API is up and shows how many documents exist in the ChromaDB collection.

```bash
curl http://127.0.0.1:8000/health
```

## Project files

- `aboutme.txt`: source text to embed
- `embed.py`: ingests `aboutme.txt` into ChromaDB
- `app.py`: FastAPI app + Chroma retrieval + Ollama generation
- `chroma_db/`: persisted ChromaDB storage (created after embedding)

