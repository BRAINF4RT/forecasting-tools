import json
import unittest
from datetime import datetime, timezone
from unittest import mock

from forecasting_tools.agents_and_tools.research import free_sources as fs

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>Fed holds rates steady - Reuters</title>
<link>https://news.google.com/rss/articles/abc</link>
<pubDate>Tue, 29 Sep 2026 14:00:00 GMT</pubDate>
<description>&lt;a href="x"&gt;Fed holds rates steady&lt;/a&gt;&amp;nbsp;&amp;nbsp;&lt;font&gt;Reuters&lt;/font&gt;</description>
<source url="https://reuters.com">Reuters</source></item>
<item><title></title><link>https://x</link></item>
<item><title>Second story</title><link>https://news.google.com/rss/articles/def</link>
<pubDate>Mon, 28 Sep 2026 09:30:00 GMT</pubDate><description>Some detail here</description></item>
</channel></rss>"""


class Resp:
    def __init__(self, status=200, content=b"", data=None, text=""):
        self.status_code = status
        self.content = content
        self._data = data
        self.text = text or content.decode("utf-8", "ignore")

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data


def patched_get(response):
    return mock.patch.object(fs.requests, "get", return_value=response)


class ParsingHelpers(unittest.TestCase):
    def test_parse_date_formats(self):
        utc = timezone.utc
        self.assertEqual(
            fs._parse_date("Tue, 29 Sep 2026 14:00:00 GMT"),
            datetime(2026, 9, 29, 14, 0, tzinfo=utc),
        )
        self.assertEqual(
            fs._parse_date("20260930T120000Z"),
            datetime(2026, 9, 30, 12, 0, tzinfo=utc),
        )
        self.assertEqual(
            fs._parse_date("2026-09-30T12:00:00+02:00"),
            datetime(2026, 9, 30, 10, 0, tzinfo=utc),
        )
        self.assertEqual(
            fs._parse_date(1790000000000).tzinfo, utc
        )
        self.assertIsNone(fs._parse_date("garbage"))
        self.assertIsNone(fs._parse_date(None))

    def test_keyword_terms(self):
        q = "Will the Federal Reserve cut interest rates at its November 2026 meeting?"
        self.assertEqual(
            fs.keyword_terms(q),
            ["federal", "reserve", "cut", "interest", "rates", "november", "2026", "meeting"],
        )
        self.assertEqual(fs.keyword_query("Will it?"), "")

    def test_term_overlap(self):
        hits, ratio = fs.term_overlap(["fed", "cut", "2026", "bitcoin"], "Fed cut in 2026")
        self.assertEqual(hits, 3)
        self.assertAlmostEqual(ratio, 0.75)
        self.assertEqual(fs.term_overlap([], "x"), (0, 0.0))

    def test_term_overlap_folds_plurals(self):
        hits, _ = fs.term_overlap(["rates", "cut"], "Fed rate cuts")
        self.assertEqual(hits, 2)


class HttpLayer(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(fs.time, "sleep")
        p.start()
        self.addCleanup(p.stop)

    def test_retries_then_raises_on_5xx(self):
        with patched_get(Resp(status=503)) as get:
            with self.assertRaises(fs.SourceError):
                fs._get("https://example.com", retries=2)
        self.assertEqual(get.call_count, 2)

    def test_4xx_raises_immediately(self):
        with patched_get(Resp(status=404)) as get:
            with self.assertRaises(fs.SourceError):
                fs._get("https://example.com", retries=3)
        self.assertEqual(get.call_count, 1)

    def test_recovers_after_transient_error(self):
        with mock.patch.object(
            fs.requests, "get", side_effect=[Resp(status=500), Resp(content=b"ok")]
        ):
            self.assertEqual(fs._get("https://example.com").content, b"ok")


class NewsSources(unittest.TestCase):
    def setUp(self):
        fs.GDELT_MIN_INTERVAL = 0
        p = mock.patch.object(fs.time, "sleep")
        p.start()
        self.addCleanup(p.stop)

    def test_google_news_rss(self):
        with patched_get(Resp(content=RSS)) as get:
            items = fs.google_news_rss("fed rates", max_results=5)
        self.assertEqual(get.call_args.kwargs["params"]["q"], "fed rates when:14d")
        self.assertEqual([i.title for i in items], ["Fed holds rates steady", "Second story"])
        first = items[0]
        self.assertEqual(first.source, "Reuters")
        self.assertEqual(first.snippet, "")
        self.assertEqual(first.date, datetime(2026, 9, 29, 14, 0, tzinfo=timezone.utc))
        self.assertEqual(items[1].source, "Google News")
        self.assertEqual(items[1].snippet, "Some detail here")

    def test_google_news_invalid_xml(self):
        with patched_get(Resp(content=b"<rss><item>")):
            with self.assertRaises(fs.SourceError):
                fs.google_news_rss("x")

    def test_gdelt(self):
        data = {"articles": [
            {"url": "https://a.com/1", "title": "Story A", "seendate": "20260930T101500Z", "domain": "a.com"},
            {"url": "", "title": "no url"},
        ]}
        with patched_get(Resp(data=data)) as get:
            items = fs.gdelt_articles("fed rates")
        self.assertIn("sourcelang:english", get.call_args.kwargs["params"]["query"])
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].source, "a.com")
        self.assertEqual(items[0].date, datetime(2026, 9, 30, 10, 15, tzinfo=timezone.utc))

    def test_gdelt_non_json_rate_limit(self):
        with patched_get(Resp(text="Please limit requests to one every 5 seconds")):
            with self.assertRaises(fs.SourceError):
                fs.gdelt_articles("fed rates")

    def test_empty_queries_short_circuit(self):
        with patched_get(Resp()) as get:
            self.assertEqual(fs.google_news_rss("  "), [])
            self.assertEqual(fs.gdelt_articles(""), [])
            self.assertEqual(fs.wikipedia_extracts(""), [])
            self.assertEqual(fs.polymarket_markets(""), [])
            self.assertEqual(fs.manifold_markets(""), [])
        get.assert_not_called()


class BackgroundAndMarkets(unittest.TestCase):
    def setUp(self):
        p = mock.patch.object(fs.time, "sleep")
        p.start()
        self.addCleanup(p.stop)

    def test_wikipedia_orders_by_search_rank(self):
        data = {"query": {"pages": {
            "2": {"title": "Second Page", "index": 2, "extract": "two"},
            "1": {"title": "First Page", "index": 1, "extract": "one " * 500},
            "3": {"title": "Empty", "index": 3, "extract": ""},
        }}}
        with patched_get(Resp(data=data)):
            items = fs.wikipedia_extracts("thing", chars=100)
        self.assertEqual([i.title for i in items], ["First Page", "Second Page"])
        self.assertEqual(len(items[0].snippet), 100)
        self.assertEqual(items[0].url, "https://en.wikipedia.org/wiki/First_Page")

    def test_polymarket(self):
        data = {"events": [{"slug": "fed-nov", "markets": [
            {"question": "Fed cuts in November?", "outcomes": json.dumps(["Yes", "No"]),
             "outcomePrices": json.dumps(["0.62", "0.38"]), "volumeNum": 125000.5,
             "endDate": "2026-11-05T00:00:00Z"},
            {"question": "Closed one", "closed": True, "outcomes": "[]", "outcomePrices": "[]"},
            {"question": "Who wins?", "outcomes": json.dumps(["A", "B"]),
             "outcomePrices": json.dumps(["0.7", "0.3"])},
        ]}]}
        with patched_get(Resp(data=data)):
            items = fs.polymarket_markets("fed cut")
        self.assertEqual([i.title for i in items], ["Fed cuts in November?", "Who wins?"])
        self.assertAlmostEqual(items[0].probability, 0.62)
        self.assertEqual(items[0].volume, 125000.5)
        self.assertEqual(items[0].url, "https://polymarket.com/event/fed-nov")
        self.assertEqual(items[0].close_time.year, 2026)
        self.assertIsNone(items[1].probability)
        self.assertEqual(items[1].snippet, "Outcomes: A 70%, B 30%")

    def test_manifold(self):
        rows = [
            {"question": "Will X happen?", "probability": 0.31, "url": "https://manifold.markets/a/x",
             "volume": 900, "closeTime": 1793000000000, "outcomeType": "BINARY"},
            {"question": "Resolved", "probability": 1, "isResolved": True, "outcomeType": "BINARY"},
            {"question": "Multi", "outcomeType": "MULTIPLE_CHOICE"},
        ]
        with patched_get(Resp(data=rows)):
            items = fs.manifold_markets("x")
        self.assertEqual(len(items), 1)
        self.assertAlmostEqual(items[0].probability, 0.31)
        self.assertEqual(items[0].source, "Manifold")

    def test_manifold_unexpected_shape(self):
        with patched_get(Resp(data={"error": "nope"})):
            with self.assertRaises(fs.SourceError):
                fs.manifold_markets("x")


if __name__ == "__main__":
    unittest.main()
