# FreeSearcher

A keyless, free stand-in for `SmartSearcher` (which needs Exa). It plans queries
with an LLM, gathers evidence from free sources in parallel, and condenses it.

```
question -> PLAN (LLM adds queries; original question is always query #1)
         -> GATHER web pages (DDGS+Trafilatura) | Google News RSS | GDELT | DDGS news
                   Wikipedia | Polymarket | Manifold
         -> CONDENSE (free model writes a "superforecaster's assistant" brief)
```

## Install

```bash
pip install 'forecasting-tools[free-search]'      # ddgs, trafilatura, beautifulsoup4
# in this repo:  poetry install --all-extras
```

## Use it in a bot

```python
from forecasting_tools import TemplateBot, GeneralLlm

bot = TemplateBot(
    llms={
        "default": GeneralLlm(model="openrouter/nvidia/nemotron-3-ultra-550b-a55b:free"),
        "summarizer": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "parser": "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
        "researcher": "free-searcher",        # or "free-searcher/<litellm model name>"
    },
)
```

Note: `TemplateBot`'s *default* llms are paid (e.g. with only `OPENROUTER_API_KEY`
set, the default researcher is `openai/gpt-4o-search-preview`). Set every purpose
explicitly, as above, to stay free.

Standalone / any other bot:

```python
from forecasting_tools import FreeSearcher

brief = await FreeSearcher().invoke("Will X happen before 2027?")        # SmartSearcher-style
brief = await FreeSearcher(llm=my_general_llm, planner="openrouter/openrouter/free").research(
    question_text=q.question_text, resolution_criteria=q.resolution_criteria or "",
    background=q.background_info or "", fine_print=q.fine_print or "")
```

`llm` and `planner` accept a model name, a `GeneralLlm`, or any `async (prompt) -> str`.

## Environment variables (all optional)

| Variable | Default | Purpose |
|---|---|---|
| `FREE_SEARCHER_PLANNER_MODEL` | `openrouter/openrouter/auto` | Writes the extra search queries. The Auto Router bills at the routed model's price, so it is **not** guaranteed free; use a `:free` model or `openrouter/openrouter/free` to stay free. |
| `FREE_SEARCHER_MODEL` | `openrouter/nvidia/nemotron-3-ultra-550b-a55b:free` | Condenses the evidence (used when `llm=` is not given). |

If the planner fails, the free model tries next, then deterministic query variants.

## Caveats
- Free endpoints rate-limit and change. Each source can fail independently and is logged, never fatal. GDELT is throttled to about 1 request / 5.5 s.
- DDGS gets throttled hard from CI IPs; keep `_max_concurrent_questions` low.
- `as_of=<datetime>` disables sources that cannot be date-filtered (web pages, Wikipedia, markets) and drops news after the cutoff, for backtests.
- Tests: `pytest code_tests/unit_tests/test_free_searcher` (offline; HTTP is mocked).
