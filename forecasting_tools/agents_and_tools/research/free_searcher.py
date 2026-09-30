"""FreeSearcher - a free stand-in for ``forecasting_tools.SmartSearcher``.

Pipeline
--------
1. **Plan**: the query-planner LLM (OpenRouter's Auto Router by default,
   ``FREE_SEARCHER_PLANNER_MODEL``) proposes extra search queries. The original question is *always* query #1, verbatim, and a
   deterministic fallback is used if planning fails.
2. **Gather** (concurrently): scraped web pages (``research.scraper``), Google
   News RSS, GDELT, DDGS news, Wikipedia intros, Polymarket and Manifold
   prices. Every source is allowed to fail independently.
3. **Condense**: a free model (``FREE_SEARCHER_MODEL``) writes a "research assistant to a
   superforecaster" brief with dates and sources. The raw evidence is appended.

``await FreeSearcher().invoke(prompt)`` mirrors ``SmartSearcher.invoke`` so it
can be dropped into any ``ForecastBot.run_research``.

``llm`` and ``planner`` each accept a model name, a ``GeneralLlm`` (anything with
``async invoke(prompt) -> str``), or a plain ``async (prompt) -> str`` callable.

Needs the optional ``free-search`` extra for scraped web pages and DDGS news:
``pip install 'forecasting-tools[free-search]'``
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from forecasting_tools.agents_and_tools.research import free_sources as fs
from forecasting_tools.agents_and_tools.research.free_sources import SourceItem

if TYPE_CHECKING:
    from forecasting_tools.ai_models.general_llm import GeneralLlm

logger = logging.getLogger(__name__)

LlmFn = Callable[[str], Awaitable[str]]
WebSearchFn = Callable[[str, int, bool, str | None], str]

NO_RESEARCH = "NO_RESEARCH_AVAILABLE"

# OpenRouter's Auto Router bills at the routed model's price, so it is NOT
# guaranteed free. Set FREE_SEARCHER_PLANNER_MODEL to a `:free` model (or
# `openrouter/openrouter/free`) to keep query planning free-only.
DEFAULT_PLANNER_MODEL = "openrouter/openrouter/auto"
DEFAULT_LLM_MODEL = "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"


def _to_llm_fn(
    model: Any, *, temperature: float, default_model_name: str
) -> LlmFn:
    """Turn a model name / GeneralLlm-like / async callable into ``LlmFn``."""
    if model is None:
        model = default_model_name
    if isinstance(model, str):
        from forecasting_tools.ai_models.general_llm import (  # noqa: PLC0415
            GeneralLlm,
        )

        model = GeneralLlm(model=model, temperature=temperature)
    if hasattr(type(model), "invoke"):
        return model.invoke
    if callable(model):
        return model
    raise TypeError(f"Unsupported model spec: {model!r}")

PLANNER_PROMPT = """\
Today is {today}. You write web search queries for a research assistant who \
supports a superforecaster.

Question: {question}
Resolution criteria: {criteria}
Background: {background}

Write {n} distinct search queries that would surface the most useful evidence: \
the latest news, the current status of the key quantity or event, official \
sources, and base rates or historical precedent.
Rules: one query per line, no numbering, no quotes, no commentary, each under \
12 words, use specific names, places and dates from the question.
"""

CONDENSE_PROMPT = """\
You are an assistant to a superforecaster.
The superforecaster will give you a question they intend to forecast on.
To be a great assistant, you generate a concise but detailed rundown of the \
most relevant news, including if the question would resolve Yes or No based \
on current information.
You do not produce forecasts yourself.

Today is {today}.

Question:
{question}

This question's outcome will be determined by the specific criteria below:
{criteria}

{fine_print}

Question background:
{background}

{question_context}

Identify the facts and developments most relevant to this exact question and \
its resolution criteria. Preserve important dates, numbers and named sources. \
Note meaningful disagreement between sources. Discard anything that does not \
help determine how this question will resolve.
Prediction-market prices are evidence about *related* questions, not about \
this one: say whether each market's wording and close date match the \
resolution criteria before leaning on it. Do not invent facts and do not \
produce a forecast.

Evidence:
{evidence}
"""


# ---------------------------------------------------------------------------
# Query planning helpers (pure functions, unit-tested)
# ---------------------------------------------------------------------------


def parse_queries(text: str) -> list[str]:
    """Extract clean one-per-line queries from an LLM reply."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S | re.I)
    queries: list[str] = []
    for raw in text.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]+|\d+[.)])\s*", "", raw).strip()
        line = line.strip("\"'`").strip()
        if not (3 <= len(line) <= 200) or line.endswith(":"):
            continue
        if line.lower().startswith(("here are", "here is", "sure,", "sure!")):
            continue
        queries.append(line)
    return queries


def dedupe_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for query in queries:
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            out.append(query)
    return out


def fallback_queries(question_text: str) -> list[str]:
    """Deterministic variants (same rules as ``research.pipeline``)."""
    if not question_text:
        return []
    queries = [question_text]
    base = question_text[:-1] if question_text.endswith("?") else question_text
    words = base.split()
    if len(words) > 6:
        queries.append(" ".join(words[:8]))
        queries.append(" ".join(words[:6]) + " latest news")
    return dedupe_queries(queries)


# ---------------------------------------------------------------------------
# Evidence container
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    news: list[SourceItem] = field(default_factory=list)
    markets: list[SourceItem] = field(default_factory=list)
    wiki: list[SourceItem] = field(default_factory=list)
    web: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.news or self.markets or self.wiki or self.web)


def _fmt_date(value: datetime | None) -> str:
    return value.strftime("%Y-%m-%d") if value else "date unknown"


def format_evidence(evidence: Evidence) -> str:
    """Render evidence as sectioned plain text for the condenser / forecaster."""
    sections: list[str] = []
    if evidence.markets:
        lines = []
        for m in evidence.markets:
            price = (
                f"{m.probability:.0%} Yes" if m.probability is not None else m.snippet
            )
            extra = []
            if m.volume:
                extra.append(f"volume {m.volume:,.0f}")
            if m.close_time:
                extra.append(f"closes {_fmt_date(m.close_time)}")
            tail = f" ({', '.join(extra)})" if extra else ""
            lines.append(f'- {m.source}: "{m.title}" -> {price}{tail} {m.url}'.strip())
        sections.append("## PREDICTION MARKETS (related questions)\n" + "\n".join(lines))
    if evidence.news:
        lines = []
        for n in evidence.news:
            snippet = f" - {n.snippet}" if n.snippet else ""
            lines.append(
                f"- [{_fmt_date(n.date)}] {n.title} ({n.source}){snippet} {n.url}".strip()
            )
        sections.append("## RECENT NEWS\n" + "\n".join(lines))
    if evidence.wiki:
        lines = [f"### {w.title} ({w.url})\n{w.snippet}" for w in evidence.wiki]
        sections.append("## BACKGROUND (Wikipedia)\n" + "\n\n".join(lines))
    if evidence.web:
        sections.append("## WEB PAGES\n" + "\n\n===\n\n".join(evidence.web))
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# FreeSearcher
# ---------------------------------------------------------------------------

DEFAULT_SOURCES = frozenset(
    {"web", "google_news", "ddgs_news", "gdelt", "wikipedia", "polymarket", "manifold"}
)
# Sources whose content cannot be date-filtered are skipped when ``as_of`` is
# set, otherwise a backtest would see post-cutoff information.
_UNDATED_SOURCES = frozenset({"web", "wikipedia", "polymarket", "manifold"})


class FreeSearcher:
    def __init__(
        self,
        llm: "str | GeneralLlm | LlmFn | None" = None,
        planner: "str | GeneralLlm | LlmFn | None" = None,
        *,
        num_queries: int = 3,
        sites_per_query: int = 4,
        use_planner: bool = True,
        sources: frozenset[str] | set[str] = DEFAULT_SOURCES,
        as_of: datetime | None = None,
        web_search_fn: WebSearchFn | None = None,
        source_timeout: float = 90.0,
        max_news: int = 15,
        max_markets: int = 6,
        web_concurrency: int = 2,
    ) -> None:
        self._llm_spec = llm
        self._planner_spec = planner
        self._llm_fn: LlmFn | None = None
        self._planner_fn: LlmFn | None = None
        self.num_queries = max(1, num_queries)
        self.sites_per_query = sites_per_query
        self.use_planner = use_planner
        self.sources = set(sources)
        self.as_of = as_of
        self._web_search_fn = web_search_fn
        self.source_timeout = source_timeout
        self.max_news = max_news
        self.max_markets = max_markets
        self.web_concurrency = max(1, web_concurrency)
        if as_of is not None:
            self.sources -= _UNDATED_SOURCES

    # -- LLM hooks ---------------------------------------------------------

    async def _call_llm(self, prompt: str) -> str:
        if self._llm_fn is None:
            self._llm_fn = _to_llm_fn(
                self._llm_spec,
                temperature=0.15,
                default_model_name=os.getenv("FREE_SEARCHER_MODEL", DEFAULT_LLM_MODEL),
            )
        return await self._llm_fn(prompt)

    async def _call_planner(self, prompt: str) -> str:
        if self._planner_fn is None:
            self._planner_fn = _to_llm_fn(
                self._planner_spec,
                temperature=0.3,
                default_model_name=os.getenv(
                    "FREE_SEARCHER_PLANNER_MODEL", DEFAULT_PLANNER_MODEL
                ),
            )
        return await self._planner_fn(prompt)

    def _search_web(self, query: str, relevance_text: str) -> str:
        fn = self._web_search_fn
        if fn is None:
            from forecasting_tools.agents_and_tools.research.free_web_scraper import (  # noqa: PLC0415,E501
                web_search,
            )

            fn = web_search
        return fn(query, self.sites_per_query, True, relevance_text)

    # -- 1. plan -----------------------------------------------------------

    def _today(self) -> str:
        return (self.as_of or datetime.now(timezone.utc)).strftime("%Y-%m-%d")

    async def plan_queries(
        self,
        question_text: str,
        resolution_criteria: str = "",
        background: str = "",
    ) -> list[str]:
        """Original question first, then planner queries, capped at N."""
        if not question_text:
            return []
        planned: list[str] = []
        if self.use_planner and self.num_queries > 1:
            prompt = PLANNER_PROMPT.format(
                today=self._today(),
                question=question_text,
                criteria=(resolution_criteria or "n/a")[:1500],
                background=(background or "n/a")[:1500],
                n=self.num_queries - 1,
            )
            for name, call in (
                ("planner", self._call_planner),
                ("free model", self._call_llm),
            ):
                try:
                    planned = parse_queries(await call(prompt))
                except Exception as exc:  # noqa: BLE001 - planning is best-effort
                    logger.warning(
                        "[FREE-SEARCHER] query planning with %s failed: %s", name, exc
                    )
                    continue
                if planned:
                    break
        if not planned:
            planned = fallback_queries(question_text)[1:]
        return dedupe_queries([question_text, *planned])[: self.num_queries]

    # -- 2. gather ---------------------------------------------------------

    async def _guard(self, name: str, awaitable: Awaitable, evidence: Evidence):
        try:
            return await asyncio.wait_for(awaitable, timeout=self.source_timeout)
        except Exception as exc:  # noqa: BLE001 - every source may fail
            message = f"{name}: {type(exc).__name__}: {exc}"
            logger.warning("[FREE-SEARCHER] %s", message)
            evidence.errors.append(message)
            return None

    async def gather(self, question_text: str, queries: list[str]) -> Evidence:
        evidence = Evidence()
        terms = fs.keyword_terms(question_text)
        kw = " ".join(terms) or question_text[:100]
        planned = queries[1:3]
        web_gate = asyncio.Semaphore(self.web_concurrency)

        async def web_one(query: str) -> str | None:
            async with web_gate:
                return await asyncio.to_thread(
                    self._search_web, query, question_text
                )

        jobs: list[tuple[str, Awaitable]] = []
        if "web" in self.sources:
            jobs += [("web", web_one(q)) for q in queries]
        if "google_news" in self.sources:
            jobs += [
                ("google_news", asyncio.to_thread(fs.google_news_rss, q))
                for q in dict.fromkeys([kw, *planned])
            ]
        if "ddgs_news" in self.sources:
            jobs.append(("ddgs_news", asyncio.to_thread(fs.ddgs_news, kw)))
        if "gdelt" in self.sources:
            jobs.append(("gdelt", asyncio.to_thread(fs.gdelt_articles, kw)))
        if "wikipedia" in self.sources:
            jobs.append(("wikipedia", asyncio.to_thread(fs.wikipedia_extracts, kw)))
        if "polymarket" in self.sources:
            jobs.append(("polymarket", asyncio.to_thread(fs.polymarket_markets, kw)))
        if "manifold" in self.sources:
            jobs.append(("manifold", asyncio.to_thread(fs.manifold_markets, kw)))

        results = await asyncio.gather(
            *(self._guard(name, aw, evidence) for name, aw in jobs)
        )
        news: list[SourceItem] = []
        markets: list[SourceItem] = []
        for (name, _), result in zip(jobs, results):
            if not result:
                continue
            if name == "web":
                text = str(result).strip()
                if text and text != NO_RESEARCH:
                    evidence.web.append(text)
            elif name == "wikipedia":
                evidence.wiki.extend(result)
            elif name in ("polymarket", "manifold"):
                markets.extend(result)
            else:
                news.extend(result)

        evidence.news = self._select_news(news)
        evidence.markets = self._select_markets(markets, terms)
        return evidence

    def _select_news(self, items: list[SourceItem]) -> list[SourceItem]:
        seen: set[str] = set()
        kept: list[SourceItem] = []
        for item in items:
            if self.as_of is not None and (
                item.date is None or item.date > self.as_of
            ):
                continue
            key = re.sub(r"\W+", " ", item.title.lower()).strip() or item.url
            if key in seen:
                continue
            seen.add(key)
            kept.append(item)
        floor = datetime.min.replace(tzinfo=timezone.utc)
        kept.sort(key=lambda i: i.date or floor, reverse=True)
        return kept[: self.max_news]

    def _select_markets(
        self, items: list[SourceItem], terms: list[str]
    ) -> list[SourceItem]:
        kept: list[SourceItem] = []
        seen: set[str] = set()
        for item in items:
            hits, ratio = fs.term_overlap(terms, item.title)
            if hits < 2 or ratio < 0.3:
                continue
            key = item.title.casefold()
            if key in seen:
                continue
            seen.add(key)
            kept.append(item)
        kept.sort(key=lambda i: i.volume or 0.0, reverse=True)
        return kept[: self.max_markets]

    # -- 3. condense -------------------------------------------------------

    async def research(
        self,
        question_text: str,
        resolution_criteria: str = "",
        background: str = "",
        fine_print: str = "",
        question_context: str = "",
    ) -> str:
        if not question_text:
            return f"RESEARCH STATUS: {NO_RESEARCH}"

        queries = await self.plan_queries(
            question_text, resolution_criteria, background
        )
        logger.info("[FREE-SEARCHER] queries: %s", queries)
        evidence = await self.gather(question_text, queries)
        logger.info(
            "[FREE-SEARCHER] news=%d markets=%d wiki=%d web=%d errors=%d",
            len(evidence.news),
            len(evidence.markets),
            len(evidence.wiki),
            len(evidence.web),
            len(evidence.errors),
        )
        if evidence.empty:
            return (
                f"RESEARCH STATUS: {NO_RESEARCH}\n\n"
                "All free sources returned nothing usable. "
                "Do not treat this as evidence that no information exists."
            )

        raw = format_evidence(evidence)
        prompt = CONDENSE_PROMPT.format(
            today=self._today(),
            question=question_text,
            criteria=resolution_criteria or "n/a",
            fine_print=fine_print or "",
            background=background or "n/a",
            question_context=question_context or "",
            evidence=raw[:30000],
        )
        try:
            summary = (await self._call_llm(prompt)).strip()
        except Exception as exc:  # noqa: BLE001 - never lose gathered evidence
            logger.warning("[FREE-SEARCHER] condensing failed: %s", exc)
            summary = ""
        if not summary:
            summary = (
                "Research was retrieved, but summarisation failed. "
                "The raw evidence is supplied below for the forecaster."
            )
        return f"{summary}\n\nRAW EVIDENCE:\n{raw[:30000]}"

    async def invoke(self, prompt: str) -> str:
        """``SmartSearcher``-compatible entry point: prompt in, brief out."""
        return await self.research(prompt)
