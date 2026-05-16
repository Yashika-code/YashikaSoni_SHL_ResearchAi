# SHL Recommendation Agent

FastAPI service that scrapes the SHL Individual Test Solutions catalog, builds a FAISS index, and serves conversational assessment recommendations through `/chat`.

## Local setup

1. Create and activate a Python 3.11 virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Build the catalog artifacts:

```powershell
python app/catalog.py
```

4. Set your Anthropic key:

```powershell
$env:ANTHROPIC_API_KEY="your_key_here"
```

5. Start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

6. Check the service:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/health
```

## Render deployment

This repo includes [render.yaml](render.yaml) for a native Render web service.

### Deploy steps

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the repo.
3. Let Render use [render.yaml](render.yaml).
4. Add the environment variable `ANTHROPIC_API_KEY` in Render.
5. Deploy.

### Render behavior

- Build step: `pip install -r requirements.txt && python app/catalog.py`
- Start step: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health check: `/health`

## API

### `GET /health`

Returns:

```json
{"status":"ok"}
```

### `POST /chat`

Request:

```json
{
  "messages": [
    {"role": "user", "content": "hiring a java developer"}
  ]
}
```

Response:

```json
{
  "reply": "string",
  "recommendations": [
    {"name": "string", "url": "string", "test_type": "string"}
  ],
  "end_of_conversation": false
}
```
