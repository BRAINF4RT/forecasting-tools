import asyncio
import unittest
from datetime import datetime, timezone
from unittest import mock

from forecasting_tools.agents_and_tools.research import free_searcher as fsr
from forecasting_tools.agents_and_tools.research import free_sources as fs
from forecasting_tools.agents_and_tools.research.free_sources import SourceItem

Q = "Will the Federal Reserve cut interest rates at its November 2026 meeting?"
UTC = timezone.utc


def news(title, day, url=None, source="Reuters"):
    return SourceItem(kind="news", source=source, title=title,
                      url=url or f"https://n/{title}", date=datetime(2026, 9, day, tzinfo=UTC))


def market(title, prob, volume, source="Polymarket"):
    return SourceItem(kind="market", source=source, title=title, url="https://m/" + title,
                      probability=prob, volume=volume, close_time=datetime(2026, 11, 5, tzinfo=UTC))


def fake_web(query, n, scrape, relevance):
    return f"### Page for {query}\nSource: https://web/{abs(hash(query)) % 100}\nbody text"


class Sources:
    """Patch every free source with canned data."""

    def __init__(self, **overrides):
        self.patches = {
            "google_news_rss": [news("Fed holds rates", 29), news("Fed holds rates", 28)],
            "ddgs_news": [news("Powell speech", 30, source="AP")],
            "gdelt_articles": [news("Old story", 10), SourceItem("news", "x.com", "Undated", "https://u")],
            "wikipedia_extracts": [SourceItem("wiki", "Wikipedia", "Federal Reserve", "https://w", snippet="The Fed is...")],
            "polymarket_markets": [market("Fed rate cut in November 2026?", 0.62, 5000),
                                   market("Will Bitcoin hit 200k?", 0.1, 99999)],
            "manifold_markets": [market("Fed cuts by November 2026", 0.55, 800, "Manifold")],
        }
        self.patches.update(overrides)
        self._ctx = []

    def __enter__(self):
        for name, value in self.patches.items():
            if isinstance(value, Exception):
                m = mock.patch.object(fs, name, side_effect=value)
            else:
                m = mock.patch.object(fs, name, return_value=value)
            m.start()
            self._ctx.append(m)
        return self

    def __exit__(self, *exc):
        for m in self._ctx:
            m.stop()


class QueryPlanning(unittest.IsolatedAsyncioTestCase):
    def test_parse_queries_cleans_llm_noise(self):
        text = ("<think>hmm</think>Here are the queries:\n1. fed november rate cut odds\n"
                "- \"powell rate guidance\"\n\n* fed funds futures 2026\nQueries:\nx")
        self.assertEqual(fsr.parse_queries(text),
                         ["fed november rate cut odds", "powell rate guidance", "fed funds futures 2026"])

    def test_fallback_queries_match_deterministic_rules(self):
        self.assertEqual(fsr.fallback_queries("Short one?"), ["Short one?"])
        out = fsr.fallback_queries(Q)
        self.assertEqual(out[0], Q)
        self.assertEqual(out[1], "Will the Federal Reserve cut interest rates at")
        self.assertEqual(out[2], "Will the Federal Reserve cut interest latest news")

    async def test_original_question_always_first_and_capped(self):
        async def planner(prompt):
            self.assertIn("Today is", prompt)
            self.assertIn(Q, prompt)
            return "1. fed november odds\n2. powell guidance\n3. extra query\n4. another"
        s = fsr.FreeSearcher(planner=planner, num_queries=3)
        self.assertEqual(await s.plan_queries(Q), [Q, "fed november odds", "powell guidance"])

    async def test_planner_failure_uses_deterministic_fallback(self):
        async def broken(prompt):
            raise RuntimeError("router down")
        s = fsr.FreeSearcher(llm=broken, planner=broken, num_queries=3)
        queries = await s.plan_queries(Q)
        self.assertEqual(queries, fsr.fallback_queries(Q)[:3])

    async def test_planner_failure_falls_back_to_free_model_first(self):
        async def broken(prompt):
            raise RuntimeError("router down")
        async def free_llm(prompt):
            return "free model query one\nfree model query two"
        s = fsr.FreeSearcher(llm=free_llm, planner=broken, num_queries=3)
        self.assertEqual(await s.plan_queries(Q),
                         [Q, "free model query one", "free model query two"])

    async def test_planner_can_be_disabled(self):
        planner = mock.AsyncMock(return_value="x query")
        s = fsr.FreeSearcher(planner=planner, use_planner=False)
        await s.plan_queries(Q)
        planner.assert_not_called()

    async def test_duplicate_of_question_is_dropped(self):
        async def planner(prompt):
            return Q.upper() + "\nnew angle query"
        s = fsr.FreeSearcher(planner=planner, num_queries=3)
        self.assertEqual(await s.plan_queries(Q), [Q, Q.upper().replace(Q.upper(), "new angle query")])


class ModelSpecs(unittest.IsolatedAsyncioTestCase):
    async def test_plain_async_callable_is_used_directly(self):
        async def fn(prompt):
            return "called"
        wrapped = fsr._to_llm_fn(fn, temperature=0.1, default_model_name="x")
        self.assertEqual(await wrapped("p"), "called")

    async def test_object_with_invoke_is_wrapped(self):
        class FakeGeneralLlm:
            async def invoke(self, prompt):
                return "invoked " + prompt
        wrapped = fsr._to_llm_fn(FakeGeneralLlm(), temperature=0.1, default_model_name="x")
        self.assertEqual(await wrapped("p"), "invoked p")

    async def test_model_name_builds_general_llm_lazily(self):
        created = {}
        class FakeGeneralLlm:
            def __init__(self, model, temperature):
                created.update(model=model, temperature=temperature)
            async def invoke(self, prompt):
                return "ok"
        fake_module = mock.MagicMock(GeneralLlm=FakeGeneralLlm)
        with mock.patch.dict("sys.modules",
                             {"forecasting_tools.ai_models.general_llm": fake_module}):
            wrapped = fsr._to_llm_fn(None, temperature=0.3,
                                     default_model_name="openrouter/openrouter/auto")
            self.assertEqual(await wrapped("p"), "ok")
        self.assertEqual(created, {"model": "openrouter/openrouter/auto", "temperature": 0.3})

    def test_rejects_unsupported_spec(self):
        with self.assertRaises(TypeError):
            fsr._to_llm_fn(42, temperature=0.1, default_model_name="x")

    async def test_env_overrides_default_models(self):
        seen = []
        class FakeGeneralLlm:
            def __init__(self, model, temperature):
                seen.append(model)
            async def invoke(self, prompt):
                return "q1"
        fake_module = mock.MagicMock(GeneralLlm=FakeGeneralLlm)
        env = {"FREE_SEARCHER_PLANNER_MODEL": "openrouter/some/planner:free",
               "FREE_SEARCHER_MODEL": "openrouter/some/writer:free"}
        with mock.patch.dict("os.environ", env), mock.patch.dict(
                "sys.modules", {"forecasting_tools.ai_models.general_llm": fake_module}):
            s = fsr.FreeSearcher(num_queries=2)
            await s._call_planner("p")
            await s._call_llm("p")
        self.assertEqual(seen, ["openrouter/some/planner:free", "openrouter/some/writer:free"])


class Gathering(unittest.IsolatedAsyncioTestCase):
    async def test_gather_merges_dedupes_and_filters(self):
        s = fsr.FreeSearcher(web_search_fn=fake_web)
        with Sources():
            ev = await s.gather(Q, [Q, "fed november odds"])
        titles = [n.title for n in ev.news]
        self.assertEqual(titles.count("Fed holds rates"), 1)  # deduped
        self.assertEqual(titles[0], "Powell speech")  # newest first
        self.assertEqual(titles[-1], "Undated")  # undated last
        self.assertEqual([m.title for m in ev.markets],
                         ["Fed rate cut in November 2026?", "Fed cuts by November 2026"])
        self.assertEqual(len(ev.wiki), 1)
        self.assertEqual(len(ev.web), 2)
        self.assertEqual(ev.errors, [])

    async def test_each_source_can_fail_independently(self):
        s = fsr.FreeSearcher(web_search_fn=fake_web)
        with Sources(gdelt_articles=fs.SourceError("rate limited"),
                     polymarket_markets=RuntimeError("boom")):
            ev = await s.gather(Q, [Q])
        self.assertEqual(len(ev.errors), 2)
        self.assertTrue(any("gdelt" in e for e in ev.errors))
        self.assertTrue(ev.news and ev.wiki)

    async def test_slow_source_times_out(self):
        def slow(*a):
            import time; time.sleep(0.5); return []
        s = fsr.FreeSearcher(web_search_fn=fake_web, source_timeout=0.05,
                             sources={"wikipedia"})
        with Sources(wikipedia_extracts=[]), mock.patch.object(fs, "wikipedia_extracts", side_effect=slow):
            ev = await s.gather(Q, [Q])
        self.assertTrue(any("wikipedia" in e for e in ev.errors))

    async def test_web_results_marked_unavailable_are_ignored(self):
        s = fsr.FreeSearcher(web_search_fn=lambda *a: "NO_RESEARCH_AVAILABLE", sources={"web"})
        ev = await s.gather(Q, [Q])
        self.assertTrue(ev.empty)

    async def test_as_of_blocks_undated_sources_and_future_news(self):
        cutoff = datetime(2026, 9, 25, tzinfo=UTC)
        s = fsr.FreeSearcher(as_of=cutoff, web_search_fn=fake_web)
        self.assertEqual(s.sources, {"google_news", "ddgs_news", "gdelt"})
        with Sources():
            ev = await s.gather(Q, [Q])
        self.assertEqual([n.title for n in ev.news], ["Old story"])
        self.assertFalse(ev.markets or ev.wiki or ev.web)


class EndToEnd(unittest.IsolatedAsyncioTestCase):
    async def test_research_builds_prompt_and_appends_raw_evidence(self):
        seen = {}
        async def llm(prompt):
            seen["prompt"] = prompt
            return "SUMMARY: markets lean yes."
        async def planner(prompt):
            return "fed november odds"
        s = fsr.FreeSearcher(llm=llm, planner=planner, web_search_fn=fake_web)
        with Sources():
            out = await s.research(Q, resolution_criteria="Resolves yes if the FOMC cuts.",
                                   background="Bg", fine_print="FP", question_context="CTX")
        self.assertTrue(out.startswith("SUMMARY: markets lean yes."))
        self.assertIn("RAW EVIDENCE:", out)
        for needle in ("PREDICTION MARKETS", "RECENT NEWS", "BACKGROUND (Wikipedia)", "WEB PAGES", "62% Yes"):
            self.assertIn(needle, out)
        p = seen["prompt"]
        self.assertIn("Resolves yes if the FOMC cuts.", p)
        self.assertIn("do not produce a forecast", p.lower())
        self.assertIn("CTX", p)

    async def test_llm_failure_still_returns_evidence(self):
        async def llm(prompt):
            raise RuntimeError("all models rate limited")
        s = fsr.FreeSearcher(llm=llm, use_planner=False, web_search_fn=fake_web)
        with Sources():
            out = await s.research(Q)
        self.assertIn("summarisation failed", out)
        self.assertIn("RECENT NEWS", out)

    async def test_no_evidence_reports_unavailable_without_calling_llm(self):
        llm = mock.AsyncMock(return_value="x")
        s = fsr.FreeSearcher(llm=llm, use_planner=False, web_search_fn=lambda *a: "NO_RESEARCH_AVAILABLE")
        empty = {n: [] for n in ("google_news_rss", "ddgs_news", "gdelt_articles",
                                 "wikipedia_extracts", "polymarket_markets", "manifold_markets")}
        with Sources(**empty):
            out = await s.research(Q)
        self.assertIn("NO_RESEARCH_AVAILABLE", out)
        llm.assert_not_called()

    async def test_invoke_is_smartsearcher_compatible(self):
        async def llm(prompt):
            return "ok"
        s = fsr.FreeSearcher(llm=llm, use_planner=False, web_search_fn=fake_web)
        with Sources():
            self.assertTrue((await s.invoke(Q)).startswith("ok"))

    async def test_empty_question(self):
        self.assertIn("NO_RESEARCH_AVAILABLE", await fsr.FreeSearcher().research(""))


if __name__ == "__main__":
    unittest.main()
