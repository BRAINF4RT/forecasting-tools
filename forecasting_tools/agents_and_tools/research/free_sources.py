"""Free, keyless evidence sources for forecasting research.

Every function here is a plain blocking call (``requests``) that returns a list
of :class:`SourceItem`; ``research/free_searcher.py`` runs them in threads.

Sources
-------
* Google News RSS      - recent headlines
* GDELT DOC 2.0        - global news index
* DDGS news            - DuckDuckGo news vertical
* Wikipedia            - background / status-quo intro extracts
* Polymarket + Manifold - prediction-market prices on related questions

None of them needs an API key. Free endpoints change and rate-limit without
notice, so callers must treat every function as fallible.
"""
from __future__ import annotations

import html
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "BRAINF4RT-forecastingbot/1.0 "
    "(+https://github.com/BRAINF4RT/forecastingbot)"
)
HTTP_TIMEOUT = 15
RETRY_SLEEP_SECONDS = 1.5
# GDELT asks clients to send at most one request every ~5 seconds.
GDELT_MIN_INTERVAL = 5.5

_GDELT_LOCK = threading.Lock()
_gdelt_last_call = float("-inf")


class SourceError(RuntimeError):
    """Raised when a free source returns something unusable."""


@dataclass
class SourceItem:
    kind: str  # "news" | "market" | "wiki"
    source: str
    title: str
    url: str = ""
    date: datetime | None = None
    snippet: str = ""
    probability: float | None = None  # markets only: P(Yes) in [0, 1]
    volume: float | None = None  # markets only
    close_time: datetime | None = None  # markets only


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "about", "after", "against", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "before", "being", "between", "by", "can", "could", "did",
    "do", "does", "for", "from", "has", "have", "how", "if", "in", "into", "is",
    "it", "its", "may", "might", "more", "most", "of", "on", "or", "our",
    "over", "should", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "through", "to", "under", "until", "up",
    "was", "we", "were", "what", "when", "where", "which", "who", "will",
    "with", "would", "you", "your", "yes", "no", "least", "than", "end",
}


def keyword_terms(text: str, max_terms: int = 8) -> list[str]:
    """Distinct meaningful lowercase tokens, in order of first appearance."""
    seen: set[str] = set()
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-]*", text or ""):
        low = token.lower().strip("-'")
        if len(low) < 3 or low in _STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(low)
        if len(terms) >= max_terms:
            break
    return terms


def keyword_query(text: str, max_terms: int = 8) -> str:
    """Short keyword query (search engines and market APIs dislike sentences)."""
    return " ".join(keyword_terms(text, max_terms))


def _stem(token: str) -> str:
    """Crude plural folding so 'cuts' matches 'cut' (no external NLP deps)."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def term_overlap(terms: list[str], text: str) -> tuple[int, float]:
    """Return (hits, hit ratio) of ``terms`` found in ``text``."""
    if not terms:
        return 0, 0.0
    haystack = {_stem(t) for t in re.findall(r"[a-z0-9]+", (text or "").lower())}
    hits = sum(1 for term in terms if _stem(term) in haystack)
    return hits, hits / len(terms)


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html.unescape(text or ""))
    return re.sub(r"\s+", " ", text).strip()


def _parse_date(value: Any) -> datetime | None:
    """Parse RFC-822, ISO-8601, GDELT compact and epoch dates to aware UTC."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        seconds = value / 1000 if value > 1e11 else value
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip()
    try:
        return datetime.strptime(text, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        pass
    parsed: datetime | None
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _json_list(value: Any) -> list[Any]:
    """Polymarket returns some list fields as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except ValueError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _get(
    url: str,
    params: dict[str, Any] | None = None,
    retries: int = 2,
) -> requests.Response:
    """GET with a descriptive User-Agent and bounded retries on 429/5xx."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT, "Accept": "*/*"},
                timeout=HTTP_TIMEOUT,
            )
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code == 429 or response.status_code >= 500:
                last_error = SourceError(f"HTTP {response.status_code} from {url}")
            elif response.status_code >= 400:
                raise SourceError(f"HTTP {response.status_code} from {url}")
            else:
                return response
        if attempt < retries:
            time.sleep(RETRY_SLEEP_SECONDS * attempt)
    raise SourceError(f"{url} failed after {retries} attempts: {last_error}")


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------


def google_news_rss(
    query: str, max_results: int = 8, days: int = 14
) -> list[SourceItem]:
    """Recent headlines from the public Google News RSS search feed."""
    query = (query or "").strip()
    if not query:
        return []
    response = _get(
        "https://news.google.com/rss/search",
        params={
            "q": f"{query} when:{days}d",
            "hl": "en-US",
            "gl": "US",
            "ceid": "US:en",
        },
    )
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise SourceError(f"Google News RSS was not valid XML: {exc}") from exc

    items: list[SourceItem] = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        link = (node.findtext("link") or "").strip()
        if not title or not link:
            continue
        source_node = node.find("source")
        source_name = (
            (source_node.text or "").strip()
            if source_node is not None and source_node.text
            else "Google News"
        )
        suffix = f" - {source_name}"
        if source_name != "Google News" and title.endswith(suffix):
            title = title[: -len(suffix)].strip()
        snippet = _strip_html(node.findtext("description") or "")
        if snippet.startswith(title):
            snippet = snippet[len(title):].strip(" -\u2013\u2014")
        if snippet == source_name:
            snippet = ""
        items.append(
            SourceItem(
                kind="news",
                source=source_name,
                title=title,
                url=link,
                date=_parse_date(node.findtext("pubDate")),
                snippet=snippet[:400],
            )
        )
        if len(items) >= max_results:
            break
    return items


def gdelt_articles(
    query: str, max_results: int = 8, timespan: str = "14d"
) -> list[SourceItem]:
    """English-language articles from the GDELT DOC 2.0 API (no key)."""
    global _gdelt_last_call
    query = (query or "").strip()
    if not query:
        return []
    with _GDELT_LOCK:
        wait = GDELT_MIN_INTERVAL - (time.monotonic() - _gdelt_last_call)
        if wait > 0:
            time.sleep(wait)
        try:
            response = _get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": f"{query} sourcelang:english",
                    "mode": "artlist",
                    "maxrecords": max_results,
                    "format": "json",
                    "sort": "datedesc",
                    "timespan": timespan,
                },
                retries=1,
            )
        finally:
            _gdelt_last_call = time.monotonic()
    try:
        data = response.json()
    except ValueError as exc:
        # GDELT answers with plain text for rate limits and bad queries.
        raise SourceError(
            f"GDELT returned non-JSON: {response.text[:120]!r}"
        ) from exc

    items: list[SourceItem] = []
    for article in (data.get("articles") or [])[:max_results]:
        url = str(article.get("url") or "").strip()
        title = str(article.get("title") or "").strip()
        if not url or not title:
            continue
        items.append(
            SourceItem(
                kind="news",
                source=str(article.get("domain") or "GDELT"),
                title=title,
                url=url,
                date=_parse_date(article.get("seendate")),
            )
        )
    return items


def ddgs_news(
    query: str, max_results: int = 8, timelimit: str = "m"
) -> list[SourceItem]:
    """DuckDuckGo news vertical via the ``ddgs`` package (imported lazily)."""
    query = (query or "").strip()
    if not query:
        return []
    from forecasting_tools.util.optional_imports import (  # noqa: PLC0415
        require_optional_package,
    )

    require_optional_package("ddgs", "ddgs", "free-search")
    from ddgs import DDGS  # noqa: PLC0415 - optional at import time

    with DDGS() as ddgs:
        rows = list(
            ddgs.news(
                query,
                region="us-en",
                safesearch="moderate",
                timelimit=timelimit,
                max_results=max_results,
            )
        )
    items: list[SourceItem] = []
    for row in rows:
        url = str(row.get("url") or row.get("href") or "").strip()
        title = str(row.get("title") or "").strip()
        if not url or not title:
            continue
        items.append(
            SourceItem(
                kind="news",
                source=str(row.get("source") or "DDGS news"),
                title=title,
                url=url,
                date=_parse_date(row.get("date")),
                snippet=str(row.get("body") or "").strip()[:400],
            )
        )
    return items


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


def wikipedia_extracts(
    query: str, max_results: int = 3, chars: int = 1200
) -> list[SourceItem]:
    """Intro extracts of the best-matching Wikipedia articles."""
    query = (query or "").strip()
    if not query:
        return []
    response = _get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrlimit": max_results,
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "exlimit": "max",
            "redirects": 1,
        },
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise SourceError("Wikipedia returned non-JSON") from exc
    pages = (data.get("query") or {}).get("pages") or {}
    items: list[SourceItem] = []
    for page in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        title = str(page.get("title") or "").strip()
        extract = str(page.get("extract") or "").strip()
        if not title or not extract:
            continue
        items.append(
            SourceItem(
                kind="wiki",
                source="Wikipedia",
                title=title,
                url="https://en.wikipedia.org/wiki/" + title.replace(" ", "_"),
                snippet=extract[:chars],
            )
        )
    return items[:max_results]


# ---------------------------------------------------------------------------
# Prediction markets
# ---------------------------------------------------------------------------


def polymarket_markets(query: str, max_results: int = 5) -> list[SourceItem]:
    """Open Polymarket markets matching ``query`` (public Gamma search)."""
    query = (query or "").strip()
    if not query:
        return []
    response = _get(
        "https://gamma-api.polymarket.com/public-search",
        params={"q": query, "limit_per_type": max_results},
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise SourceError("Polymarket returned non-JSON") from exc

    items: list[SourceItem] = []
    for event in data.get("events") or []:
        slug = str(event.get("slug") or "").strip()
        for market in event.get("markets") or []:
            if market.get("closed") or market.get("archived"):
                continue
            title = str(market.get("question") or event.get("title") or "").strip()
            if not title:
                continue
            outcomes = [str(o) for o in _json_list(market.get("outcomes"))]
            prices = [_to_float(p) for p in _json_list(market.get("outcomePrices"))]
            probability: float | None = None
            snippet = ""
            if (
                len(outcomes) == 2
                and len(prices) == 2
                and outcomes[0].lower() == "yes"
                and prices[0] is not None
            ):
                probability = prices[0]
            elif outcomes and len(outcomes) == len(prices):
                snippet = "Outcomes: " + ", ".join(
                    f"{name} {price:.0%}"
                    for name, price in zip(outcomes, prices)
                    if price is not None
                )
            items.append(
                SourceItem(
                    kind="market",
                    source="Polymarket",
                    title=title,
                    url=f"https://polymarket.com/event/{slug}" if slug else "",
                    snippet=snippet,
                    probability=probability,
                    volume=_to_float(
                        market.get("volumeNum")
                        if market.get("volumeNum") is not None
                        else market.get("volume")
                    ),
                    close_time=_parse_date(market.get("endDate")),
                )
            )
    return items[:max_results]


def manifold_markets(query: str, max_results: int = 5) -> list[SourceItem]:
    """Open binary Manifold markets matching ``query``."""
    query = (query or "").strip()
    if not query:
        return []
    response = _get(
        "https://api.manifold.markets/v0/search-markets",
        params={"term": query, "limit": max_results, "sort": "relevance"},
    )
    try:
        rows = response.json()
    except ValueError as exc:
        raise SourceError("Manifold returned non-JSON") from exc
    if not isinstance(rows, list):
        raise SourceError(f"Unexpected Manifold response: {str(rows)[:120]!r}")

    items: list[SourceItem] = []
    for row in rows:
        if row.get("isResolved") or row.get("outcomeType") != "BINARY":
            continue
        probability = _to_float(row.get("probability"))
        title = str(row.get("question") or "").strip()
        if not title or probability is None:
            continue
        items.append(
            SourceItem(
                kind="market",
                source="Manifold",
                title=title,
                url=str(row.get("url") or "").strip(),
                probability=probability,
                volume=_to_float(row.get("volume")),
                close_time=_parse_date(row.get("closeTime")),
            )
        )
    return items[:max_results]
