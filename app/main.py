from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.agent import create_agent
from app.catalog import CatalogStore, build_catalog
from app.models import ChatRequest, ChatResponse


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

app = FastAPI(title="SHL Recommendation Agent", version="1.0.0")


@app.on_event("startup")
async def startup_event() -> None:
    LOGGER.info("Loading catalog artifacts and embedding model")
    try:
        app.state.catalog = await asyncio.to_thread(CatalogStore.load)
    except FileNotFoundError:
        LOGGER.info("Catalog artifacts were missing; building them now")
        await asyncio.to_thread(build_catalog)
        app.state.catalog = await asyncio.to_thread(CatalogStore.load)


def _get_agent() -> object:
    agent = getattr(app.state, "agent", None)
    if agent is None:
        try:
            agent = create_agent(app.state.catalog)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        app.state.agent = agent
    return agent


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "message": "SHL Recommendation Agent is running.",
        "chat_endpoint": "/chat",
        "method": "POST",
    }


@app.get("/chat")
async def chat_help() -> HTMLResponse:
        html = """
        <!doctype html>
        <html>
        <head>
            <meta charset="utf-8" />
            <title>SHL Recommendation Agent — Chat</title>
            <style>body{font-family:system-ui,Segoe UI,Arial;margin:20px}textarea{width:100%;height:80px}pre{background:#f5f5f5;padding:12px;border-radius:6px}</style>
        </head>
        <body>
            <h1>SHL Recommendation Agent</h1>
            <p>Enter a user message below and click <strong>Send</strong> to call the live <code>/chat</code> endpoint.</p>
            <textarea id="msg">hiring a java developer</textarea>
            <p><button id="send">Send</button> <span id="status"></span></p>
            <h2>Response</h2>
            <pre id="out">(awaiting request)</pre>
            <script>
                const sendBtn = document.getElementById('send');
                const out = document.getElementById('out');
                const status = document.getElementById('status');
                sendBtn.onclick = async () => {
                    const msg = document.getElementById('msg').value;
                    status.textContent = 'sending...';
                    out.textContent = '';
                    try {
                        const resp = await fetch('/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ messages: [{ role: 'user', content: msg }] })
                        });
                        const text = await resp.text();
                        try { const j = JSON.parse(text); out.textContent = JSON.stringify(j, null, 2); }
                        catch(e){ out.textContent = text }
                        status.textContent = 'done (' + resp.status + ')';
                    } catch (err) {
                        out.textContent = err.toString();
                        status.textContent = 'error';
                    }
                };
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html, status_code=200)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        agent = _get_agent()
        result = await agent.handle(request.messages)
        return ChatResponse(
            reply=result.reply,
            recommendations=result.recommendations,
            end_of_conversation=result.end_of_conversation,
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("/chat failed")
        raise HTTPException(status_code=502, detail=f"chat failed: {exc}") from exc
