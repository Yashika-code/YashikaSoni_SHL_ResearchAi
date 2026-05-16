from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI, HTTPException

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
