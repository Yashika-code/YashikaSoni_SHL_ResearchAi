from __future__ import annotations

import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


LOGGER = logging.getLogger(__name__)

CATALOG_URL = "https://www.shl.com/products/product-catalog/"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"

REQUEST_TIMEOUT = 20
MAX_WORKERS = 8

TEST_TYPE_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("A", ("ability", "aptitude", "cognitive", "reasoning", "numerical", "verbal", "inductive", "deductive", "diagrammatic", "logical")),
    ("P", ("personality", "opq", "motivation", "values", "style", "preferences")),
    ("B", ("behavior", "behaviour", "behavioral", "behavioural", "situational", "sjt", "judgment", "judgement", "leadership", "management style")),
    ("K", ("knowledge", "technical", "job knowledge", "software", "programming", "compliance", "skill", "skills")),
    ("S", ("simulation", "work sample", "situational", "case study")),
]


@dataclass(frozen=True)
class CatalogItem:
    name: str
    url: str
    description: str
    test_type: str

    @property
    def search_text(self) -> str:
        return f"{self.name}\n{self.description}\n{self.test_type}"


def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }
    )
    return session


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _is_individual_section(text: str) -> bool:
    return bool(re.search(r"individual test solutions", text, re.IGNORECASE))


def _is_prepackaged_section(text: str) -> bool:
    return bool(re.search(r"pre-?packaged job solutions", text, re.IGNORECASE))


def _infer_test_type(text: str) -> str:
    haystack = text.lower()
    for test_type, keywords in TEST_TYPE_KEYWORDS:
        if any(keyword in haystack for keyword in keywords):
            return test_type
    return "A"


def _looks_like_product_url(href: str) -> bool:
    href = href.lower()
    return "/solutions/products/" in href or "/productcatalog/" in href or "/assessments/" in href


def _extract_meta_description(soup: BeautifulSoup) -> str:
    meta_selectors = [
        ('meta[name="description"]', "content"),
        ('meta[property="og:description"]', "content"),
        ('meta[name="twitter:description"]', "content"),
    ]
    for selector, attr in meta_selectors:
        tag = soup.select_one(selector)
        if tag and tag.get(attr):
            description = _normalize_space(tag.get(attr, ""))
            if description:
                return description
    return ""


def _extract_title(soup: BeautifulSoup) -> str:
    for selector in ["h1", 'meta[property="og:title"]', "title"]:
        tag = soup.select_one(selector)
        if not tag:
            continue
        if selector == "title":
            text = tag.get_text(" ", strip=True)
        elif selector.startswith("meta"):
            text = tag.get("content", "")
        else:
            text = tag.get_text(" ", strip=True)
        text = _normalize_space(text)
        if text:
            return text
    return ""


def _fetch_html(session: requests.Session, url: str) -> str:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def _candidate_from_anchor(anchor, base_url: str) -> tuple[str, str] | None:
    href = anchor.get("href") or ""
    text = _normalize_space(anchor.get_text(" ", strip=True))
    if not href or not text:
        return None

    absolute_url = urljoin(base_url, href)
    if not _looks_like_product_url(absolute_url):
        return None

    return text, absolute_url


def _collect_candidates(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str, str]]:
    candidates: list[tuple[str, str, str]] = []
    anchors = soup.find_all("a", href=True)
    for anchor in anchors:
        candidate = _candidate_from_anchor(anchor, base_url)
        if not candidate:
            continue

        name, url = candidate
        context_node = anchor.parent
        context_text = _normalize_space(context_node.get_text(" ", strip=True) if context_node else name)
        if _is_prepackaged_section(context_text):
            continue

        if not _is_individual_section(context_text):
            nearby_text = _normalize_space(" ".join(part for part in [name, context_text] if part))
            if not _is_individual_section(nearby_text):
                continue

        candidates.append((name, url, context_text))

    unique: dict[str, tuple[str, str, str]] = {}
    for name, url, context_text in candidates:
        unique.setdefault(url, (name, url, context_text))
    return list(unique.values())


def _extract_listing_fallback(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str, str]]:
    anchors = soup.find_all("a", href=True)
    fallback: list[tuple[str, str, str]] = []
    for anchor in anchors:
        candidate = _candidate_from_anchor(anchor, base_url)
        if not candidate:
            continue
        name, url = candidate
        context_node = anchor.parent
        context_text = _normalize_space(context_node.get_text(" ", strip=True) if context_node else name)
        if _is_prepackaged_section(context_text):
            continue
        fallback.append((name, url, context_text))

    unique: dict[str, tuple[str, str, str]] = {}
    for name, url, context_text in fallback:
        unique.setdefault(url, (name, url, context_text))
    return list(unique.values())


def _scrape_detail(session: requests.Session, candidate: tuple[str, str, str]) -> CatalogItem | None:
    fallback_name, url, context_text = candidate
    try:
        html = _fetch_html(session, url)
        soup = BeautifulSoup(html, "html.parser")
        title = _extract_title(soup) or fallback_name
        description = _extract_meta_description(soup)
        if not description:
            paragraphs = [
                _normalize_space(p.get_text(" ", strip=True))
                for p in soup.find_all("p")
            ]
            paragraphs = [paragraph for paragraph in paragraphs if paragraph]
            description = max(paragraphs, key=len, default="")
        description = description or context_text or fallback_name
        test_type = _infer_test_type(" ".join([title, description, context_text]))
        return CatalogItem(name=title, url=url, description=description, test_type=test_type)
    except Exception as exc:  # pragma: no cover - network dependent
        LOGGER.warning("Failed to scrape product detail for %s: %s", url, exc)
        description = context_text or fallback_name
        test_type = _infer_test_type(" ".join([fallback_name, description]))
        if not fallback_name or not url:
            return None
        return CatalogItem(name=fallback_name, url=url, description=description, test_type=test_type)


def scrape_catalog(url: str = CATALOG_URL) -> list[CatalogItem]:
    session = _create_session()
    html = _fetch_html(session, url)
    soup = BeautifulSoup(html, "html.parser")

    candidates = _collect_candidates(soup, url)
    if not candidates:
        candidates = _extract_listing_fallback(soup, url)

    if not candidates:
        raise RuntimeError("No SHL Individual Test Solutions were found on the catalog page.")

    items: list[CatalogItem] = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(_scrape_detail, session, candidate) for candidate in candidates]
        for future in as_completed(futures):
            item = future.result()
            if item:
                items.append(item)

    deduped: dict[str, CatalogItem] = {}
    for item in items:
        deduped.setdefault(item.url, item)

    return sorted(deduped.values(), key=lambda item: item.name.lower())


def _vectorize_texts(model: SentenceTransformer, texts: Iterable[str]) -> np.ndarray:
    vectors = model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    return vectors


def build_catalog(output_dir: Path = DEFAULT_DATA_DIR) -> list[CatalogItem]:
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Scraping SHL catalog from %s", CATALOG_URL)
    items = scrape_catalog(CATALOG_URL)
    LOGGER.info("Scraped %d catalog items", len(items))

    model = SentenceTransformer(MODEL_NAME)
    vectors = _vectorize_texts(model, (item.search_text for item in items))

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    catalog_json_path = output_dir / "catalog.json"
    faiss_index_path = output_dir / "catalog.faiss"

    with catalog_json_path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(item) for item in items], handle, ensure_ascii=False, indent=2)

    faiss.write_index(index, str(faiss_index_path))
    LOGGER.info("Wrote catalog JSON to %s", catalog_json_path)
    LOGGER.info("Wrote FAISS index to %s", faiss_index_path)
    return items


class CatalogStore:
    def __init__(self, items: list[CatalogItem], index: faiss.Index, model: SentenceTransformer):
        self.items = items
        self.index = index
        self.model = model
        self._items_by_url = {item.url: item for item in items}
        self._items_by_name = {self._normalize(item.name): item for item in items}

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().lower()

    @classmethod
    def load(cls, data_dir: Path = DEFAULT_DATA_DIR, model_name: str = MODEL_NAME) -> "CatalogStore":
        json_path = data_dir / "catalog.json"
        index_path = data_dir / "catalog.faiss"

        if not json_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                "Catalog artifacts are missing. Run `python app/catalog.py` to build catalog.json and catalog.faiss first."
            )

        with json_path.open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)

        items = [CatalogItem(**item) for item in raw_items]
        index = faiss.read_index(str(index_path))
        model = SentenceTransformer(model_name)
        return cls(items=items, index=index, model=model)

    def search(self, query: str, limit: int = 10) -> list[tuple[CatalogItem, float]]:
        limit = max(1, min(limit, len(self.items)))
        if not query.strip() or not self.items:
            return []

        query_vector = _vectorize_texts(self.model, [query])
        scores, indices = self.index.search(query_vector, limit)
        matches: list[tuple[CatalogItem, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.items):
                continue
            matches.append((self.items[index], float(score)))
        return matches

    def find_by_name(self, name: str) -> CatalogItem | None:
        return self._items_by_name.get(self._normalize(name))

    def find_by_url(self, url: str) -> CatalogItem | None:
        return self._items_by_url.get(url)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape SHL catalog and build the FAISS index.")
    parser.add_argument("--output-dir", default=str(DEFAULT_DATA_DIR), help="Directory for catalog.json and catalog.faiss")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_catalog(Path(args.output_dir))


if __name__ == "__main__":
    main()
