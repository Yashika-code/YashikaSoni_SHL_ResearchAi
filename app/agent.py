from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass

from anthropic import AsyncAnthropic

from app.catalog import CatalogItem, CatalogStore
from app.models import ChatMessage, Recommendation


LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an SHL assessment recommendation agent. You help hiring managers find the right assessments from the SHL Individual Test Solutions catalog only.

Rules:
1. Never recommend on the first turn if the query is vague. Ask clarifying questions first.
2. Gather at minimum: job role, seniority level, and what you are measuring (skills/personality/cognitive).
3. Only recommend assessments from the provided catalog context. Never invent names or URLs.
4. For comparison questions, answer using only catalog data provided.
5. Refuse any off-topic questions (legal, general HR, non-SHL topics) politely.
6. When you have enough context, output a JSON block with your recommendations inside your reply like:
   RECOMMENDATIONS_JSON: [{"name": "...", "url": "...", "test_type": "..."}]
7. Output END_CONVERSATION when the shortlist is final and the user seems satisfied.

Catalog context will be injected before each turn."""

VAGUE_PHRASES = (
    "need an assessment",
    "need assessment",
    "need tests",
    "need a test",
    "hiring",
    "looking for assessments",
    "looking for a test",
    "recommend a test",
)

ROLE_HINTS = (
    "developer",
    "engineer",
    "manager",
    "analyst",
    "specialist",
    "director",
    "lead",
    "senior",
    "junior",
    "software",
    "sales",
    "operations",
    "finance",
    "marketing",
    "customer service",
    "hr",
    "human resources",
    "java",
    "python",
    "data",
    "product",
)

SENIORITY_HINTS = (
    "intern",
    "junior",
    "mid",
    "senior",
    "lead",
    "principal",
    "entry",
    "manager",
    "director",
    "vp",
    "executive",
    "experienced",
)

MEASUREMENT_HINTS = (
    "cognitive",
    "personality",
    "behavior",
    "behaviour",
    "ability",
    "skills",
    "skill",
    "knowledge",
    "aptitude",
    "leadership",
    "fit",
    "both",
)

OFF_TOPIC_PATTERNS = (
    r"\blegal advice\b",
    r"\blawyer\b",
    r"\bcontract\b",
    r"\bgeneral hr\b",
    r"\bhr policy\b",
    r"\bprompt injection\b",
    r"\bsystem prompt\b",
    r"\bignore (?:the )?instructions\b",
    r"\bmedical\b",
    r"\btherapy\b",
)


@dataclass(frozen=True)
class AgentResult:
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _extract_latest_user(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def _conversation_text(messages: list[ChatMessage]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def _has_any(text: str, hints: tuple[str, ...]) -> bool:
    normalized = _normalize(text)
    return any(hint in normalized for hint in hints)


def _is_off_topic(text: str) -> bool:
    normalized = _normalize(text)
    return any(re.search(pattern, normalized) for pattern in OFF_TOPIC_PATTERNS)


def _is_comparison_question(text: str) -> bool:
    normalized = _normalize(text)
    return any(
        keyword in normalized
        for keyword in ("difference between", "compare", "comparison", "vs", "versus", "better than")
    )


def _needs_clarification(messages: list[ChatMessage]) -> bool:
    user_text = " ".join(message.content for message in messages if message.role == "user")
    if not user_text.strip():
        return True

    latest_user = _extract_latest_user(messages)
    if len(messages) <= 1 and any(phrase in _normalize(latest_user) for phrase in VAGUE_PHRASES):
        return True

    has_role = _has_any(user_text, ROLE_HINTS)
    has_seniority = _has_any(user_text, SENIORITY_HINTS)
    has_measurement = _has_any(user_text, MEASUREMENT_HINTS)

    if not has_role:
        return True
    if not (has_seniority or has_measurement):
        return True
    return False


def _build_clarifying_reply() -> AgentResult:
    reply = (
        "I need a little more context before I can recommend an SHL assessment. "
        "What is the job role or title, what seniority level are you hiring for, and are you looking to measure skills/ability, personality/behavior, or both?"
    )
    return AgentResult(reply=reply, recommendations=[], end_of_conversation=False)


def _build_off_topic_reply() -> AgentResult:
    reply = (
        "I can only help with SHL Individual Test Solutions assessment recommendations and catalog comparisons. "
        "Please ask about a hiring role, assessment type, or a comparison between SHL tests."
    )
    return AgentResult(reply=reply, recommendations=[], end_of_conversation=True)


def _build_max_turn_reply() -> AgentResult:
    reply = (
        "This conversation has reached the 8-turn limit. If you need a new shortlist, start a fresh conversation with the role, seniority, and assessment focus."
    )
    return AgentResult(reply=reply, recommendations=[], end_of_conversation=True)


def _build_comparison_reply() -> AgentResult:
    return AgentResult(reply="", recommendations=[], end_of_conversation=False)


def _format_catalog_context(items: list[tuple[CatalogItem, float]]) -> str:
    if not items:
        return "No relevant catalog items were retrieved."
    return "\n".join(
        f"- name: {item.name}\n  test_type: {item.test_type}\n  url: {item.url}\n  description: {item.description[:350]}\n  score: {score:.3f}"
        for item, score in items
    )


def _extract_recommendations_from_text(text: str, catalog: CatalogStore) -> list[Recommendation]:
    match = re.search(r"RECOMMENDATIONS_JSON:\s*(\[[\s\S]*?\])", text)
    if not match:
        return []

    try:
        raw_items = json.loads(match.group(1))
    except json.JSONDecodeError:
        return []

    recommendations: list[Recommendation] = []
    seen_urls: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_item.get("name", "")).strip()
        url = str(raw_item.get("url", "")).strip()
        test_type = str(raw_item.get("test_type", "")).strip()

        catalog_item = catalog.find_by_url(url) if url else None
        if not catalog_item and name:
            catalog_item = catalog.find_by_name(name)

        if not catalog_item:
            continue

        resolved_url = catalog_item.url
        if resolved_url in seen_urls:
            continue

        recommendations.append(
            Recommendation(
                name=catalog_item.name,
                url=resolved_url,
                test_type=catalog_item.test_type or test_type or "A",
            )
        )
        seen_urls.add(resolved_url)

    return recommendations[:10]


def _strip_control_markers(text: str) -> str:
    cleaned = re.sub(r"\n?RECOMMENDATIONS_JSON:\s*\[[\s\S]*?\]", "", text)
    cleaned = cleaned.replace("END_CONVERSATION", "").strip()
    return cleaned


class SHLAgent:
    def __init__(self, catalog: CatalogStore):
        self.catalog = catalog
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is required to use /chat.")
        self.client = AsyncAnthropic(api_key=api_key)

    async def handle(self, messages: list[ChatMessage]) -> AgentResult:
        if len(messages) >= 8:
            return _build_max_turn_reply()

        latest_user = _extract_latest_user(messages)
        if _is_off_topic(latest_user):
            return _build_off_topic_reply()

        if _is_comparison_question(latest_user):
            search_query = _conversation_text(messages)
            retrieved = await asyncio.to_thread(self.catalog.search, search_query, 10)
            context_block = _format_catalog_context(retrieved)
            model_reply = await self._call_model(messages, context_block, comparison=True)
            reply = _strip_control_markers(model_reply)
            if not reply:
                reply = "I found catalog entries relevant to the comparison, but I need the exact test names to compare them accurately."
            return AgentResult(reply=reply, recommendations=[], end_of_conversation=False)

        if _needs_clarification(messages):
            return _build_clarifying_reply()

        search_query = _conversation_text(messages)
        retrieved = await asyncio.to_thread(self.catalog.search, search_query, 10)
        context_block = _format_catalog_context(retrieved)

        model_reply = await self._call_model(messages, context_block)
        recommendations = _extract_recommendations_from_text(model_reply, self.catalog)

        if not recommendations and retrieved:
            recommendations = [
                Recommendation(name=item.name, url=item.url, test_type=item.test_type)
                for item, _ in retrieved[:10]
            ]

        reply = _strip_control_markers(model_reply)
        end_of_conversation = "END_CONVERSATION" in model_reply
        return AgentResult(reply=reply, recommendations=recommendations, end_of_conversation=end_of_conversation)

    async def _call_model(self, messages: list[ChatMessage], context_block: str, comparison: bool = False) -> str:
        conversation = _conversation_text(messages)
        comparison_instructions = (
            "- This is a comparison question. Explain the differences using only the catalog context.\n"
            "- Do not add recommendations for comparison questions.\n"
        )
        user_prompt = (
            "Conversation so far:\n"
            f"{conversation}\n\n"
            "Catalog context:\n"
            f"{context_block}\n\n"
            "Instructions:\n"
            f"{comparison_instructions if comparison else ''}"
            "- Use only the catalog context and the conversation context.\n"
            "- If the conversation is still missing role, seniority, or assessment focus, ask concise clarifying questions.\n"
            "- If recommending, include RECOMMENDATIONS_JSON with 1 to 10 items.\n"
            "- If the shortlist is final and the user seems satisfied, include END_CONVERSATION.\n"
            "- Never invent names or URLs.\n"
        )

        response = await asyncio.wait_for(
            self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=700,
                temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            ),
            timeout=25,
        )

        content_blocks = []
        for block in response.content:
            if getattr(block, "type", "") == "text":
                content_blocks.append(block.text)
        return "\n".join(content_blocks).strip()


def create_agent(catalog: CatalogStore) -> SHLAgent:
    return SHLAgent(catalog)
