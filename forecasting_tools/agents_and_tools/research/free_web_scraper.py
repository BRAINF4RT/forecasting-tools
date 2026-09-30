"""DDGS search, source ranking, dead-page filtering and resilient scraping."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlparse, urlunparse

import requests

from forecasting_tools.util.optional_imports import require_optional_package

# Optional "free-search" extra: keeps the base install small.
require_optional_package("ddgs", "ddgs", "free-search")
require_optional_package("trafilatura", "trafilatura", "free-search")
require_optional_package("bs4", "beautifulsoup4", "free-search")

import trafilatura  # noqa: E402
from bs4 import BeautifulSoup  # noqa: E402
from ddgs import DDGS  # noqa: E402

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "Chrome/131.0 Safari/537.36"
)
MAX_CHARS_PER_SOURCE = 100000
SEARCH_BACKENDS = ("brave", "google", "bing", "duckduckgo", "yahoo", "wikipedia")
SEARCH_DELAY_SECONDS = 0.25
SCRAPE_TIMEOUT = 12
SEARCH_CANDIDATE_MULTIPLIER = 3
MIN_METADATA_RELEVANCE_SCORE = 2.0
MIN_CONTENT_RELEVANCE_SCORE = 2.0

DEAD_TITLE_MARKERS = (
    "404",
    "page not found",
    "file not found",
    "content not found",
    "page doesn't exist",
    "page does not exist",
    "access denied",
    "forbidden",
)
DEAD_PAGE_MARKERS = (
    "404 not found",
    "page not found",
    "file not found",
    "content not found",
    "this page doesn't exist",
    "this page does not exist",
    "requested page could not be found",
)

STOPWORDS = {
    "a", "about", "after", "against", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "before", "being", "between", "by", "can", "could", "does",
    "for", "from", "has", "have", "how", "if", "in", "into", "is", "it", "its",
    "may", "might", "more", "most", "of", "on", "or", "our", "over", "should",
    "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "under", "until", "up", "was", "we", "were", "what",
    "when", "where", "which", "who", "will", "with", "would", "you", "your",
}


@dataclass
class ScrapedSource:
    query: str
    title: str
    url: str
    snippet: str
    content: str = ""
    method: str = ""
    backend: str = ""
    relevance_score: float = 0.0


def _normalise_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url.rstrip("/")
        clean_query: list[str] = []
        for item in parsed.query.split("&") if parsed.query else []:
            key = item.split("=", 1)[0].lower()
            if key.startswith("utm_") or key in {"fbclid", "gclid", "ref", "ref_src"}:
                continue
            clean_query.append(item)
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                parsed.params,
                "&".join(clean_query),
                "",
            )
        )
    except Exception:
        return url.rstrip("/")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in _tokenize(text)
        if token not in STOPWORDS and len(token) >= 3
    }


def _relevance_score(question_text: str, title: str, text: str) -> float:
    question_terms = _meaningful_tokens(question_text)
    if not question_terms:
        return 0.0
    title_terms = _meaningful_tokens(title)
    text_terms = _meaningful_tokens(text)
    title_hits = question_terms & title_terms
    text_hits = question_terms & text_terms
    score = len(title_hits) * 3.0 + len(text_hits)

    question_words = _tokenize(question_text)
    combined_words = _tokenize(f"{title} {text}")
    question_bigrams = set(zip(question_words, question_words[1:]))
    combined_bigrams = set(zip(combined_words, combined_words[1:]))
    score += len(question_bigrams & combined_bigrams) * 0.75
    return score


def _looks_like_dead_page(text: str, title: str = "") -> bool:
    clean_text = (text or "").strip()
    clean_title = (title or "").strip().lower()
    if any(marker in clean_title for marker in DEAD_TITLE_MARKERS):
        return True
    if len(clean_text) <= 3000:
        haystack = f"{clean_title}\n{clean_text.lower()}"
        return any(marker in haystack for marker in DEAD_PAGE_MARKERS)
    return False


def _valid_result(result: dict) -> bool:
    url = result.get("href") or result.get("url") or ""
    title = result.get("title") or ""
    body = result.get("body") or ""
    return isinstance(url, str) and bool(url.strip()) and bool(title.strip() or body.strip())


def ddgs_search(
    query: str,
    max_results: int = 5,
    relevance_text: str | None = None,
) -> list[dict]:
    """Search multiple DDGS backends and rank candidates deterministically."""
    query = (query or "").strip()
    if not query:
        return []

    target_count = max(1, max_results)
    candidate_limit = max(target_count * SEARCH_CANDIDATE_MULTIPLIER, target_count + 4)
    backend_result_limit = max(target_count, 4)
    results_out: list[dict] = []
    seen_urls: set[str] = set()

    for backend in SEARCH_BACKENDS:
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        max_results=backend_result_limit,
                        backend=backend,
                    )
                )
            for result in results:
                if not _valid_result(result):
                    continue
                url = _normalise_url(result.get("href") or result.get("url") or "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                candidate = dict(result)
                title = str(candidate.get("title") or "").strip()
                snippet = str(candidate.get("body") or "").strip()
                candidate["_backend"] = backend
                candidate["_normalised_url"] = url
                candidate["_relevance_score"] = _relevance_score(
                    relevance_text or query,
                    title,
                    snippet,
                )
                results_out.append(candidate)
                if len(results_out) >= candidate_limit:
                    break

            if len(results_out) >= candidate_limit and backend != SEARCH_BACKENDS[0]:
                break
        except Exception as exc:
            logger.warning("[SEARCH] backend=%s failed for %r: %s", backend, query, exc)
        time.sleep(SEARCH_DELAY_SECONDS)

    results_out.sort(
        key=lambda result: result.get("_relevance_score", 0.0),
        reverse=True,
    )

    if relevance_text:
        filtered = [
            result
            for result in results_out
            if result.get("_relevance_score", 0.0) >= MIN_METADATA_RELEVANCE_SCORE
        ]
        # Do not throw away all low-scoring candidates. If the threshold leaves
        # fewer than the requested count, keep the best remaining candidates so
        # the LLM gets a chance to perform the final relevance judgment.
        if len(filtered) >= target_count:
            results_out = filtered
        else:
            filtered_urls = {
                _normalise_url(
                    result.get("href") or result.get("url") or ""
                )
                for result in filtered
            }
            tail = [
                result
                for result in results_out
                if _normalise_url(result.get("href") or result.get("url") or "")
                not in filtered_urls
            ]
            results_out = filtered + tail
        logger.info(
            "[SEARCH] relevance kept %d/%d candidates for %r",
            min(len(results_out), target_count),
            len(seen_urls),
            query,
        )

    return results_out[:target_count]


def _fetch_html(url: str) -> tuple[str | None, str, str]:
    """Fetch HTML once and classify confirmed dead vs transient failures."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=SCRAPE_TIMEOUT,
            allow_redirects=True,
        )
        final_url = response.url or url
        if response.status_code in {404, 410}:
            return None, final_url, "dead_http"
        if response.status_code >= 400:
            return None, final_url, f"http_{response.status_code}"
        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return None, final_url, "non_html"
        return response.text, final_url, "ok"
    except requests.RequestException as exc:
        logger.debug("[SCRAPE] request failed for %s: %s", url, exc)
        return None, url, "request_failed"


def scrape_with_trafilatura(downloaded: str, title: str = "") -> str | None:
    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
        )
        if not text:
            return None
        text = text.strip()
        if len(text) < 100 or _looks_like_dead_page(text, title):
            return None
        return text
    except Exception as exc:
        logger.debug("[SCRAPE] trafilatura failed: %s", exc)
        return None


def scrape_with_bs4(downloaded: str, title: str = "") -> str | None:
    try:
        soup = BeautifulSoup(downloaded, "html.parser")
        for tag in soup(
            ["script", "style", "nav", "footer", "header", "noscript", "svg", "form"]
        ):
            tag.decompose()
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(" ", strip=True)) >= 40
        ]
        text = "\n".join(paragraphs).strip()
        if len(text) < 100 or _looks_like_dead_page(text, title):
            return None
        return text
    except Exception as exc:
        logger.debug("[SCRAPE] BeautifulSoup failed: %s", exc)
        return None


def _title_similarity(a: str, b: str) -> float:
    left = _meaningful_tokens(a)
    right = _meaningful_tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def scrape_with_ddgs_snippet(
    url: str,
    title: str = "",
    original_snippet: str = "",
) -> str | None:
    """Last-resort indexed-snippet retrieval for pages direct HTTP cannot fetch."""
    target = _normalise_url(url)
    if not target:
        return None

    searches = [f'"{url}"']
    if title.strip():
        searches.append(f'"{title.strip()}"')

    for query in searches:
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.text(
                        query,
                        region="us-en",
                        safesearch="moderate",
                        max_results=8,
                        backend="auto",
                    )
                )
            for result in results:
                result_url = _normalise_url(result.get("href") or result.get("url") or "")
                result_title = str(result.get("title") or "").strip()
                body = str(result.get("body") or "").strip()
                if len(body) < 40 or _looks_like_dead_page(body, result_title):
                    continue
                if result_url == target:
                    return body
                if title and _title_similarity(title, result_title) >= 0.85:
                    return body
        except Exception as exc:
            logger.debug("[SCRAPE] DDGS snippet lookup failed for %s: %s", url, exc)

    # The original search-engine snippet is the final evidence layer. It is
    # explicitly labelled in the source method and is only allowed when HTTP
    # did not establish a definitive 404/410.
    snippet = (original_snippet or "").strip()
    return snippet if len(snippet) >= 40 and not _looks_like_dead_page(snippet, title) else None


def scrape_url(
    url: str,
    title: str = "",
    snippet: str = "",
    relevance_text: str | None = None,
) -> tuple[str, str, float]:
    """Fetch/extract one source, then fall back to DDGS snippets."""
    if not url.strip():
        return "", "invalid_url", 0.0

    downloaded, final_url, fetch_status = _fetch_html(url)
    if fetch_status == "dead_http":
        logger.info("[SOURCE] confirmed dead HTTP page: %s", url)
        return "", "dead_http", 0.0

    if downloaded:
        try:
            soup = BeautifulSoup(downloaded, "html.parser")
            page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
            visible_text = soup.get_text(" ", strip=True)[:3000]
            if _looks_like_dead_page(visible_text, f"{title} {page_title}"):
                logger.info("[SOURCE] rejected soft-dead page: %s", final_url)
                return "", "dead_content", 0.0
        except Exception:
            pass

        text = scrape_with_trafilatura(downloaded, title=title)
        method = "trafilatura" if text else ""
        if not text:
            text = scrape_with_bs4(downloaded, title=title)
            method = "bs4" if text else ""

        if text:
            score = _relevance_score(relevance_text or title, title, text[:30000])
            if relevance_text and score < MIN_CONTENT_RELEVANCE_SCORE:
                return "", "low_relevance", score
            return text, method, score

    # Transient failures, bot checks and empty extraction can still recover from
    # the search engine's index. Confirmed 404/410 pages never reach this path.
    if fetch_status != "dead_http":
        text = scrape_with_ddgs_snippet(
            final_url or url,
            title=title,
            original_snippet=snippet,
        )
        if text:
            score = _relevance_score(relevance_text or title, title, text)
            if relevance_text and score < MIN_CONTENT_RELEVANCE_SCORE:
                return "", "low_relevance", score
            method = "ddgs_snippet" if text != snippet else "ddgs_original_snippet"
            return text, method, score

    return "", "failed", 0.0


def gather_sources(
    queries: Iterable[str],
    results_per_query: int = 3,
    relevance_text: str | None = None,
) -> list[ScrapedSource]:
    """Search, rank, scrape and deduplicate sources."""
    sources: list[ScrapedSource] = []
    seen_urls: set[str] = set()
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    for query_index, query in enumerate(queries, start=1):
        logger.info(
            "[RESEARCH] Query %d/%d: %s",
            query_index,
            len(queries),
            query,
        )
        results = ddgs_search(
            query,
            max_results=results_per_query,
            relevance_text=relevance_text,
        )

        for result in results:
            url = _normalise_url(result.get("href") or result.get("url") or "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            title = str(result.get("title") or url).strip()
            snippet = str(result.get("body") or "").strip()
            backend = str(result.get("_backend") or "unknown")
            content, method, score = scrape_url(
                url,
                title=title,
                snippet=snippet,
                relevance_text=relevance_text,
            )
            if not content:
                continue

            sources.append(
                ScrapedSource(
                    query=query,
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=content[:MAX_CHARS_PER_SOURCE],
                    method=method,
                    backend=backend,
                    relevance_score=score,
                )
            )
            logger.info(
                "[SOURCE] accepted score=%.2f backend=%s method=%s url=%s",
                score,
                backend,
                method,
                url,
            )

    sources.sort(key=lambda source: source.relevance_score, reverse=True)
    logger.info("[RESEARCH] Source gathering complete: %d usable sources", len(sources))
    return sources


def format_sources_as_markdown(sources: list[ScrapedSource]) -> str:
    if not sources:
        return "NO_RESEARCH_AVAILABLE"
    return "\n---\n".join(
        f"### {source.title or source.url}\n"
        f"Source: {source.url}\n"
        f"(query: {source.query!r}; backend: {source.backend}; extraction: "
        f"{source.method}; relevance: {source.relevance_score:.2f})\n\n"
        f"{source.content}\n"
        for source in sources
    )


def web_search(
    query: str,
    max_results: int = 4,
    scrape: bool = True,
    relevance_text: str | None = None,
) -> str:
    """Run one blocking DDGS search and return validated source text."""
    query = (query or "").strip()
    if not query:
        return ""

    if scrape:
        sources = gather_sources(
            [query],
            results_per_query=max_results,
            relevance_text=relevance_text,
        )
        return format_sources_as_markdown(sources)

    results = ddgs_search(
        query,
        max_results=max_results,
        relevance_text=relevance_text,
    )
    sources = [
        ScrapedSource(
            query=query,
            title=str(result.get("title") or result.get("href") or "").strip(),
            url=_normalise_url(result.get("href") or result.get("url") or ""),
            snippet=str(result.get("body") or "").strip(),
            content=str(result.get("body") or "").strip(),
            method="ddgs_snippet",
            backend=str(result.get("_backend") or "unknown"),
            relevance_score=float(result.get("_relevance_score") or 0.0),
        )
        for result in results
    ]
    return format_sources_as_markdown(sources)
