"""Run with:  python -m unittest discover -s tests -v   (no extra packages needed)"""
import json
import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from module import config, features, llm, nextractor, pipeline, yextractor  # noqa: E402


class ParseVerdictTests(unittest.TestCase):
    def test_valid_json(self):
        raw = '{"verdict":"Red","confidence":0.9,"reasoning":"Contradicts.","contradictions":["a","b"],"off_topic":false}'
        v = llm.parse_verdict(raw)
        self.assertEqual((v["verdict"], v["confidence"], v["parsed_ok"]), ("Red", 0.9, True))
        self.assertEqual(v["contradictions"], ["a", "b"])

    def test_json_inside_code_fence_and_case(self):
        v = llm.parse_verdict('```json\n{"verdict":"green","confidence":2}\n```')
        self.assertEqual(v["verdict"], "Green")
        self.assertEqual(v["confidence"], 1.0)  # clamped

    def test_garbage_falls_back_to_yellow(self):
        v = llm.parse_verdict("I cannot decide")
        self.assertEqual(v["verdict"], "Yellow")
        self.assertFalse(v["parsed_ok"])

    def test_empty_reply_is_yellow(self):
        self.assertEqual(llm.parse_verdict("")["verdict"], "Yellow")

    def test_plain_word_reply_still_works(self):
        self.assertEqual(llm.parse_verdict("Red")["verdict"], "Red")

    def test_off_topic_forces_red_by_default(self):
        v = llm.parse_verdict('{"verdict":"Green","confidence":0.8,"off_topic":true}')
        self.assertEqual(v["verdict"], "Red")

    def test_off_topic_can_be_downgraded_to_yellow_when_configured(self):
        with mock.patch.object(config, "TAG_ABUSE_IS_RED", False):
            v = llm.parse_verdict('{"verdict":"Green","off_topic":"true"}')
        self.assertEqual(v["verdict"], "Yellow")

    def test_bad_types_do_not_crash(self):
        v = llm.parse_verdict('{"verdict":"Green","confidence":"high","contradictions":"x"}')
        self.assertEqual((v["confidence"], v["contradictions"]), (0.5, []))


class JudgeTests(unittest.TestCase):
    """The judge's calibration is meant to work the same regardless of subject (pollution, cybersecurity,
    whatever) -- these cover the two topic-agnostic signals added to help it generalize: a numeric
    wording-overlap hint, and tolerance for headline-only / technical coverage in the prompt itself."""

    def test_alignment_hint_blank_when_no_alignment_available(self):
        self.assertEqual(llm._alignment_hint(None), "")

    def test_alignment_hint_reads_high_overlap_as_meaningful(self):
        hint = llm._alignment_hint(0.5)
        self.assertIn("0.50", hint)
        self.assertIn("meaningful shared wording", hint)

    def test_alignment_hint_reads_low_overlap_as_a_soft_signal_not_a_verdict(self):
        hint = llm._alignment_hint(0.0)
        self.assertIn("the text above is the better evidence", hint)  # never told to auto-decide off the number alone

    def test_judge_prompt_mentions_headlines_and_technical_coverage_count_as_evidence(self):
        """Regression guard for the first reported failure: a technical-subject search (e.g.
        cybersecurity bulletins) finding only thin, jargon-heavy or headline-only matches must not be
        told to treat that as 'no coverage' just because it doesn't read like a plain-language news
        paragraph."""
        self.assertIn("a clear, on-theme headline is enough", llm._JUDGE_PROMPT)
        self.assertIn("Dense, technical or jargon-heavy coverage", llm._JUDGE_PROMPT)

    def test_judge_prompt_requires_theme_level_match_not_just_a_shared_field_label(self):
        """Regression guard for the second reported failure: after the first fix, matching on 'same
        broad field' alone made everything Green (a cybersecurity course matched against unrelated
        cybersecurity-policy news). The prompt must require the coverage to bear on the claim's
        specific theme, and say explicitly that a shared field label alone is not enough."""
        self.assertIn("not merely the same broad field or industry label", llm._JUDGE_PROMPT)
        self.assertIn("sharing a field label isn't enough", llm._JUDGE_PROMPT)
        # the contrastive pair that anchors the calibration: same field, different theme -> Yellow;
        # same field, same theme -> Green.
        self.assertIn("YELLOW example", llm._JUDGE_PROMPT)
        self.assertIn("GREEN example", llm._JUDGE_PROMPT)

    def test_judge_formats_and_passes_alignment_through_to_the_prompt(self):
        seen = {}

        def fake_complete(messages, *a, **k):
            seen["prompt"] = messages[0]["content"]
            return '{"verdict":"Green","confidence":0.7,"reasoning":"ok","contradictions":[],"off_topic":false}'

        with mock.patch.object(llm, "complete", side_effect=fake_complete):
            result = llm.judge("Claim: X", "News about X", alignment=0.42)
        self.assertEqual(result["verdict"], "Green")
        self.assertIn("0.42", seen["prompt"])

    def test_judge_works_with_no_alignment_given(self):
        with mock.patch.object(llm, "complete", return_value='{"verdict":"Yellow","confidence":0.5,"reasoning":"","contradictions":[],"off_topic":false}'):
            result = llm.judge("Claim: X", "News about X")
        self.assertEqual(result["verdict"], "Yellow")


class BadRequestError(Exception):
    pass


class CompleteTests(unittest.TestCase):
    def _fake(self, replies):
        calls = []

        def create(**kwargs):
            calls.append(dict(kwargs))
            item = replies.pop(0)
            if isinstance(item, Exception):
                raise item
            msg = mock.Mock(content=item)
            return mock.Mock(choices=[mock.Mock(message=msg, finish_reason="stop")])

        fake = mock.Mock()
        fake.chat.completions.create.side_effect = create
        return fake, calls

    def test_empty_reply_retries_with_bigger_budget(self):
        fake, calls = self._fake(["", "ok"])
        with mock.patch.object(llm, "client", return_value=fake), mock.patch.object(llm.time, "sleep"):
            self.assertEqual(llm.complete([{"role": "user", "content": "x"}], max_tokens=100), "ok")
        self.assertEqual([c["max_completion_tokens"] for c in calls], [100, 150])
        self.assertEqual(calls[0]["extra_body"], {"reasoning_effort": "low"})

    def test_json_mode_rejected_then_dropped(self):
        fake, calls = self._fake([BadRequestError("json_validate_failed"), "{}"])
        with mock.patch.object(llm, "client", return_value=fake), mock.patch.object(llm.time, "sleep"):
            llm.complete([{"role": "user", "content": "x"}], json_mode=True)
        self.assertIn("response_format", calls[0])
        self.assertNotIn("response_format", calls[1])

    def test_gives_up_after_retries(self):
        fake, _ = self._fake([RuntimeError("boom")] * 3)
        with mock.patch.object(llm, "client", return_value=fake), mock.patch.object(llm.time, "sleep"):
            with self.assertRaises(RuntimeError):
                llm.complete([{"role": "user", "content": "x"}])


class FeatureTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "Video Title": ["Air quality today", "SHOCKING secret exposed!!!", "City council meeting"],
            "Description": ["Report https://example.com/x #delhi", "Miracle cure, must watch! #viral #fake", ""],
            "Published At": ["2026-09-01T10:00:00Z", "2026-09-10T10:00:00Z", None],
            "Views": [1000, 500000, 0], "Likes": [50, 100, None], "Comments": [5, 10, None],
            "Subscriber Count": [10000, 1000, None], "Total Views": [1, 2, 3], "Video Count": [1, 2, 3],
        })

    def test_add_features_values(self):
        out = features.add_features(self.df, now=pd.Timestamp("2026-09-22", tz="UTC"))
        self.assertAlmostEqual(out.loc[0, "Engagement Rate"], 0.055)
        self.assertAlmostEqual(out.loc[1, "Reach Ratio"], 500.0)
        self.assertTrue(np.isnan(out.loc[2, "Engagement Rate"]))  # zero views -> NaN, not inf
        self.assertEqual(out.loc[0, "Link Count"], 1)
        self.assertGreaterEqual(out.loc[1, "Sensational Phrases"], 3)
        self.assertEqual(out.loc[0, "Sensational Phrases"], 0)

    def test_reach_z_flags_the_outlier(self):
        df = pd.DataFrame({
            "Video Title": list("abcdef"), "Description": [""] * 6,
            "Published At": ["2026-09-01T00:00:00Z"] * 6,
            "Views": [1000, 1200, 900, 1100, 1000, 900000],  # last video: huge reach for a small channel
            "Subscriber Count": [10000] * 6,
        })
        out = features.add_features(df, now=pd.Timestamp("2026-09-22", tz="UTC"))
        self.assertGreater(out.loc[5, "Reach Z"], 3)
        self.assertTrue((out.loc[:4, "Reach Z"].abs() < 3).all())

    def test_robust_z_constant_series(self):
        z = features.robust_z(pd.Series([5.0, 5.0, 5.0]))
        self.assertTrue((z == 0).all())

    def test_text_helpers(self):
        text = "Read this #delhi #aqi https://www.who.int/a and http://bbc.com/b"
        self.assertEqual(features.extract_hashtags(text), ["#delhi", "#aqi"])
        self.assertEqual(features.link_domains(text), ["bbc.com", "who.int"])
        self.assertEqual(features.clean_description(text), "Read this and")

    def test_tfidf_prefers_related_text(self):
        video = "Delhi air quality index reaches severe level pollution"
        near = "Air quality in Delhi hits severe category as pollution index rises"
        far = "Football team wins the cup final after a penalty shootout"
        near_score, far_score = features.tfidf_scores(video, [near, far])
        self.assertGreater(near_score, far_score)
        self.assertEqual(features.tfidf_scores("", [near]), [])
        self.assertIsNone(features.tfidf_alignment(video, []))

    def test_add_signals_flags_disagreements(self):
        df = pd.DataFrame({
            "Status": ["Green", "Green", "Red", "Red", "Yellow"],
            "Alignment": [0.01, 0.4, 0.5, 0.02, 0.2],
            "Confidence": [0.9, 0.9, 0.9, 0.9, 0.5],
            "Reach Z": [0, 0, 0, 0, 0], "Sensational Phrases": [0, 0, 0, 0, 0],
        })
        out = features.add_signals(df)
        self.assertEqual(out["Disagreement"].tolist(), [True, False, True, False, False])
        self.assertEqual(out["Review Priority"].idxmax(), 2)  # Red + disagreement ranks first

    def test_features_module_has_no_sklearn_dependency(self):
        """scikit-learn repeatedly failed to install on Streamlit Community Cloud, so it was
        dropped entirely in favor of a pure numpy/pandas implementation. Guard against an
        `import sklearn` quietly creeping back in (the module docstring is allowed to mention
        scikit-learn by name when explaining why it isn't used)."""
        import inspect
        source = inspect.getsource(features)
        self.assertNotRegex(source, r"^\s*(import sklearn|from sklearn)", msg="features.py must not import sklearn")

    def test_tfidf_scores_work_with_no_optional_dependencies(self):
        """tfidf_scores must produce correct results using only numpy/pandas -- no sklearn install
        needed anywhere, in any environment."""
        self.assertEqual(features.tfidf_scores("Delhi air quality index severe", ["Air quality in Delhi is severe today"])[0] > 0, True)
        self.assertEqual(features.tfidf_scores("", ["anything"]), [])

    def test_tfidf_scores_handle_stopword_only_video_text(self):
        # "the a an" tokenizes to nothing after stopword removal; must not divide by zero.
        self.assertEqual(features.tfidf_scores("the a an", ["Delhi air quality is severe"]), [0.0])

    def test_classification_summary(self):
        report, cm = features.classification_summary(["Red", "Red", "Green"], ["Red", "Green", "Green"])
        self.assertAlmostEqual(report.loc["Red", "precision"], 1.0)
        self.assertAlmostEqual(report.loc["Red", "recall"], 0.5)
        self.assertEqual(int(cm.loc["Human Red", "AI Green"]), 1)
        self.assertAlmostEqual(report.loc["accuracy", "precision"], 2 / 3)
        # round(3) is used on this report in the UI; every column must stay numeric.
        report.round(3)

    def test_classification_summary_handles_a_label_with_no_examples(self):
        report, cm = features.classification_summary(["Green", "Green"], ["Green", "Green"], labels=["Green", "Yellow", "Red"])
        self.assertEqual(report.loc["Yellow", "precision"], 0.0)
        self.assertEqual(report.loc["Yellow", "recall"], 0.0)
        self.assertEqual(int(cm.loc["Human Red", "AI Red"]), 0)


class YouTubeTests(unittest.TestCase):
    def test_duration(self):
        self.assertEqual(yextractor._seconds("PT1H2M3S"), 3723)
        self.assertEqual(yextractor._seconds("PT45S"), 45)
        self.assertIsNone(yextractor._seconds("P0D"))
        self.assertIsNone(yextractor._seconds(None))

    def test_build_rows_merges_three_sources(self):
        search = [{"id": {"videoId": "v1"}, "snippet": {"title": "cut&#39;s", "channelId": "c1", "publishedAt": "2026-09-01T00:00:00Z"}}]
        videos = {"v1": {
            "snippet": {"title": "cut's", "description": "full text", "channelId": "c1", "channelTitle": "Chan",
                        "publishedAt": "2026-09-01T00:00:00Z", "tags": ["a", "b"], "defaultAudioLanguage": "hi"},
            "statistics": {"viewCount": "100", "likeCount": "7"},
            "contentDetails": {"duration": "PT2M"},
        }}
        channels = {"c1": {"snippet": {"description": "bio", "country": "IN"}, "statistics": {"subscriberCount": "5000", "hiddenSubscriberCount": False}}}
        row = yextractor.build_rows(search, videos, channels).iloc[0]
        self.assertEqual(row["Video Title"], "cut's")  # unescaped title from videos.list
        self.assertEqual((row["Views"], row["Likes"], row["Duration (s)"], row["Subscriber Count"]), (100, 7, 120, 5000))
        self.assertTrue(pd.isna(row["Comments"]))  # comments off -> NaN, not a crash
        self.assertEqual(row["Tags"], "a, b")

    def test_hidden_subscribers_become_nan(self):
        search = [{"id": {"videoId": "v1"}, "snippet": {"channelId": "c1", "publishedAt": "x"}}]
        channels = {"c1": {"snippet": {}, "statistics": {"subscriberCount": "0", "hiddenSubscriberCount": True}}}
        self.assertTrue(pd.isna(yextractor.build_rows(search, {}, channels).iloc[0]["Subscriber Count"]))

    def test_columns_are_stable(self):
        self.assertEqual(list(yextractor.build_rows([], {}, {}).columns), yextractor.COLUMNS)


class _FakeSnippet:
    def __init__(self, text):
        self.text = text


class _FakeTranscript:
    def __init__(self, snippets):
        self._snippets = snippets

    def fetch(self):
        return self._snippets


class _FakeTranscriptList:
    """Stands in for the real TranscriptList: iterable, and find_transcript either returns the
    English transcript or raises (simulating no English track available), same as the real library."""

    def __init__(self, transcripts, english=None):
        self._transcripts = transcripts
        self._english = english

    def find_transcript(self, _langs):
        if self._english is None:
            raise Exception("no English transcript for this video")
        return self._english

    def __iter__(self):
        return iter(self._transcripts)


class TranscriptTests(unittest.TestCase):
    """fetch_transcript reads YouTube's own caption track (one small HTTP call via the
    youtube-transcript-api library) instead of downloading + decoding audio -- no ffmpeg involved,
    so it deploys the same way the existing news-scraping calls already do. It must degrade to ""
    silently in every failure case, since a transcript is a bonus signal, never a requirement."""

    def test_returns_empty_string_when_package_is_not_installed(self):
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": None}):  # forces ImportError
            self.assertEqual(yextractor.fetch_transcript("abc123"), "")

    def test_returns_empty_string_for_blank_video_id(self):
        self.assertEqual(yextractor.fetch_transcript(""), "")
        self.assertEqual(yextractor.fetch_transcript(None), "")

    def test_joins_and_clips_the_english_transcript_on_the_current_api(self):
        english = _FakeTranscript([_FakeSnippet("Hello there."), _FakeSnippet("Second line.")])
        transcript_list = _FakeTranscriptList([english], english=english)

        class FakeApi:
            def list(self, video_id):
                return transcript_list

        fake_module = mock.MagicMock()
        fake_module.YouTubeTranscriptApi = FakeApi
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            text = yextractor.fetch_transcript("abc123", max_chars=100)
        self.assertEqual(text, "Hello there. Second line.")

    def test_falls_back_to_any_available_language_when_english_is_missing(self):
        other_lang = _FakeTranscript([_FakeSnippet("Bonjour le monde.")])
        transcript_list = _FakeTranscriptList([other_lang], english=None)

        class FakeApi:
            def list(self, video_id):
                return transcript_list

        fake_module = mock.MagicMock()
        fake_module.YouTubeTranscriptApi = FakeApi
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            text = yextractor.fetch_transcript("abc123")
        self.assertEqual(text, "Bonjour le monde.")

    def test_falls_back_to_the_pre_1_0_classmethod_api(self):
        """youtube-transcript-api < 1.0 has no instance .list() method -- calling it raises
        AttributeError, which must be caught and retried against the old list_transcripts(video_id)
        classmethod rather than propagating."""
        english = _FakeTranscript([_FakeSnippet("Old API works too.")])
        transcript_list = _FakeTranscriptList([english], english=english)

        class FakeApiOld:
            pass  # deliberately has no .list method, so calling it raises AttributeError

        FakeApiOld.list_transcripts = staticmethod(lambda video_id: transcript_list)
        fake_module = mock.MagicMock()
        fake_module.YouTubeTranscriptApi = FakeApiOld
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            text = yextractor.fetch_transcript("abc123")
        self.assertEqual(text, "Old API works too.")

    def test_never_raises_when_the_video_has_no_captions_at_all(self):
        class FakeApi:
            def list(self, video_id):
                raise Exception("Subtitles are disabled for this video")

        fake_module = mock.MagicMock()
        fake_module.YouTubeTranscriptApi = FakeApi
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            self.assertEqual(yextractor.fetch_transcript("abc123"), "")

    def test_clips_long_transcripts_to_max_chars(self):
        english = _FakeTranscript([_FakeSnippet("word " * 50)])
        transcript_list = _FakeTranscriptList([english], english=english)

        class FakeApi:
            def list(self, video_id):
                return transcript_list

        fake_module = mock.MagicMock()
        fake_module.YouTubeTranscriptApi = FakeApi
        with mock.patch.dict(sys.modules, {"youtube_transcript_api": fake_module}):
            text = yextractor.fetch_transcript("abc123", max_chars=20)
        self.assertEqual(len(text), 20)


class NewsTests(unittest.TestCase):
    def setUp(self):
        nextractor._CACHE.clear()
        nextractor._LAST_CALL.clear()
        patcher = mock.patch.object(nextractor.time, "sleep")
        self.mock_sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def test_json_ld_list_and_graph(self):
        html = """<html><script type="application/ld+json">[{"@type":"WebSite"},
        {"@graph":[{"@type":"NewsArticle","headline":"H","articleBody":"Body text","publisher":[{"name":"Pub"}],
        "datePublished":"2026-09-01"}]}]</script></html>"""
        art = nextractor.article_from_html(html, "http://x")
        self.assertEqual((art.headline, art.publisher, art.url, art.content), ("H", "Pub", "http://x", "Body text"))

    def test_bad_json_ld_is_skipped(self):
        self.assertIsNone(nextractor.article_from_html('<script type="application/ld+json">{oops</script>'))

    def test_paragraph_fallback_used_when_no_json_ld(self):
        long_para = " ".join(["word"] * 50)
        html = f'<html><head><title>A Headline</title><meta property="og:site_name" content="Example News"></head><body><article><p>{long_para}</p></article></body></html>'
        art = nextractor._article_from_paragraphs(html, "http://x")
        self.assertEqual((art.headline, art.publisher, art.source), ("A Headline", "Example News", "Google News (fallback parser)"))
        self.assertIn("word", art.content)

    def test_paragraph_fallback_rejects_thin_pages(self):
        html = '<html><body><p>Subscribe now. Just three words.</p></body></html>'
        self.assertIsNone(nextractor._article_from_paragraphs(html, "http://x"))

    def test_google_links_survive_odd_items(self):
        good = [[None] * 6 + ["http://a"]]
        payload = {"x": 1}
        items = [[None, [None, None, [[None] * 6 + ["http://b"]]]], [1, 2, 3], [good[0]] + [None] * 7]
        data = "AF_initDataCallback({key: 'ds:2', data:" + json.dumps([None, [items]]) + ", sideChannel: {}});"
        html = f'<html><script class="ds:2">{data}</script></html>'
        self.assertEqual(nextractor.google_links(html), ["http://b", "http://a"])
        self.assertEqual(nextractor.google_links("<html></html>"), [])
        del payload

    def test_merges_results_from_every_configured_source(self):
        """All three sources are queried, not a strict fallback chain, and their results are
        combined -- this is what keeps the article hit rate up when any single source is thin."""
        nextractor._CACHE.clear()
        a1, a2, a3 = nextractor.News("from api", "http://a", "p"), nextractor.News("from google", "http://b", "p"), nextractor.News("from gdelt", "http://c", "p")
        with mock.patch.object(nextractor, "secret", return_value="fake-key"), \
             mock.patch.object(nextractor, "_api_fetch_articles", return_value=[a1]) as api, \
             mock.patch.object(nextractor, "_scrape_google_news", return_value=[a2]) as google, \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[a3]) as gdelt:
            result = nextractor.get_news_list("q", limit=5)
        self.assertEqual({a.url for a in result}, {"http://a", "http://b", "http://c"})
        api.assert_called_once()
        google.assert_called_once()
        gdelt.assert_called_once()

    def test_still_returns_results_when_newsapi_configured_but_empty(self):
        nextractor._CACHE.clear()
        art = nextractor.News("h", "u", "p")
        with mock.patch.object(nextractor, "secret", return_value="fake-key"), \
             mock.patch.object(nextractor, "_api_fetch_articles", return_value=[]), \
             mock.patch.object(nextractor, "_scrape_google_news", return_value=[art]) as google, \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[]):
            result = nextractor.get_news_list("q")
        self.assertEqual(result, [art])
        google.assert_called_once()

    def test_google_and_gdelt_run_but_newsapi_is_skipped_when_no_key_configured(self):
        nextractor._CACHE.clear()
        with mock.patch.object(nextractor, "secret", return_value=""), \
             mock.patch.object(nextractor, "_api_fetch_articles") as api, \
             mock.patch.object(nextractor, "_scrape_google_news", return_value=[]), \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[]):
            nextractor.get_news_list("q")
        api.assert_not_called()

    def test_one_source_raising_does_not_lose_the_others(self):
        nextractor._CACHE.clear()
        art = nextractor.News("h", "http://b", "p")
        with mock.patch.object(nextractor, "secret", return_value=""), \
             mock.patch.object(nextractor, "_scrape_google_news", side_effect=RuntimeError("scrape exploded")), \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[art]):
            result = nextractor.get_news_list("q")
        self.assertEqual(result, [art])

    def test_diagnose_reports_each_stage_without_crashing(self):
        item = [None, [None, None, [[None] * 6 + ["http://a"]]]]  # len-2 item shape
        payload = [None, [[item]]]
        data = "AF_initDataCallback({key: 'ds:2', data:" + json.dumps(payload) + ", sideChannel: {}});"
        html = f'<html><script class="ds:2">{data}</script></html>'
        resp = mock.Mock(status_code=200, content=html.encode())
        art = nextractor.News("h", "http://a", "p")
        with mock.patch.object(nextractor.requests, "get", return_value=resp), \
             mock.patch.object(nextractor, "secret", return_value=""), \
             mock.patch.object(nextractor, "_fetch_article", return_value=art), \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[art]):
            trace = nextractor.diagnose("test query")
        self.assertEqual(trace["google_http_status"], 200)
        self.assertFalse(trace["newsapi_key_configured"])
        self.assertEqual(trace["google_links_found"], 1)
        self.assertEqual(trace["google_sample_parsed"], 1)
        self.assertEqual(trace["gdelt_articles_found"], 1)

    def test_diagnose_survives_a_network_error(self):
        with mock.patch.object(nextractor.requests, "get", side_effect=nextractor.requests.RequestException("timeout")), \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[]):
            trace = nextractor.diagnose("q")
        self.assertIn("google_error", trace)

    def test_cache_does_not_store_failures(self):
        nextractor._CACHE.clear()
        with mock.patch.object(nextractor, "_scrape_google_news", return_value=[]), \
             mock.patch.object(nextractor, "_api_fetch_articles", return_value=[]), \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[]):
            self.assertEqual(nextractor.get_news_list("q"), [])
        self.assertEqual(nextractor._CACHE, {})
        art = nextractor.News("h", "u", "p")
        with mock.patch.object(nextractor, "_scrape_google_news", return_value=[art]) as scrape, \
             mock.patch.object(nextractor, "_gdelt_articles", return_value=[]):
            nextractor.get_news_list("q")
            nextractor.get_news_list("Q")  # same query, different case -> served from cache
        self.assertEqual(scrape.call_count, 1)

    def test_dedupe_prefers_the_richer_duplicate_and_ranks_full_text_first(self):
        thin = nextractor.News("H", "http://a", "p", content="")
        rich = nextractor.News("H", "http://a/", "p", content="a full article body here")
        stub = nextractor.News("Other", "http://b", "p", content="")
        deduped = nextractor._dedupe_articles([thin, stub, rich])
        self.assertEqual(len(deduped), 2)
        self.assertEqual(deduped[0].content, "a full article body here")  # full-text article ranked first

    def test_stub_from_title_keeps_headline_only_article(self):
        html = '<html><head><title>Real Headline</title><meta property="og:site_name" content="Example"></head><body><p>too short</p></body></html>'
        art = nextractor._stub_from_title(html, "http://x")
        self.assertEqual((art.headline, art.publisher, art.source), ("Real Headline", "Example", "Google News (headline only)"))

    def test_stub_from_title_rejects_consent_and_block_pages(self):
        html = "<html><head><title>Just a moment...</title></head><body></body></html>"
        self.assertIsNone(nextractor._stub_from_title(html, "http://x"))

    def test_fetch_article_falls_through_to_headline_stub(self):
        html = '<html><head><title>Headline Only</title></head><body><p>short</p></body></html>'
        resp = mock.Mock(status_code=200, content=html.encode())
        with mock.patch.object(nextractor.requests, "get", return_value=resp):
            art = nextractor._fetch_article("http://x")
        self.assertEqual(art.headline, "Headline Only")
        self.assertEqual(art.source, "Google News (headline only)")

    def test_gdelt_parses_articles_and_scrapes_bodies(self):
        payload = {"articles": [{"url": "http://a", "title": "T1", "domain": "example.com", "seendate": "20260902120000"}]}
        resp = mock.Mock(status_code=200)
        resp.json.return_value = payload
        with mock.patch.object(nextractor.requests, "get", return_value=resp), \
             mock.patch.object(nextractor, "_scrape_full_article_body", return_value="body text"):
            articles = nextractor._gdelt_articles("q", 5)
        self.assertEqual(len(articles), 1)
        self.assertEqual((articles[0].headline, articles[0].publisher, articles[0].pubDate, articles[0].content),
                         ("T1", "example.com", "2026-09-02", "body text"))

    def test_gdelt_returns_empty_list_on_http_error(self):
        resp = mock.Mock(status_code=503)
        with mock.patch.object(nextractor.requests, "get", return_value=resp):
            self.assertEqual(nextractor._gdelt_articles("q", 5), [])

    def test_gdelt_survives_network_error(self):
        with mock.patch.object(nextractor.requests, "get", side_effect=nextractor.requests.RequestException("timeout")):
            self.assertEqual(nextractor._gdelt_articles("q", 5), [])

    def test_clean_newsapi_text_strips_truncation_marker(self):
        text = "The bulletin lists new vulnerabilities recorded this week… [+899 chars]"
        cleaned = nextractor._clean_newsapi_text(text)
        self.assertNotIn("chars]", cleaned)
        self.assertNotIn("899", cleaned)

    def test_combine_newsapi_text_prefers_the_longer_superset_over_a_duplicate(self):
        # content is description plus more -- a common NewsAPI pattern. Using both would duplicate.
        combined = nextractor._combine_newsapi_text(
            "Short summary sentence.", "Short summary sentence. A bit more detail follows. [+899 chars]"
        )
        self.assertEqual(combined, "Short summary sentence. A bit more detail follows.")

    def test_combine_newsapi_text_joins_genuinely_different_snippets(self):
        combined = nextractor._combine_newsapi_text("A different editorial summary.", "An unrelated lead sentence. [+50 chars]")
        self.assertIn("different editorial summary", combined)
        self.assertIn("unrelated lead sentence", combined)

    def test_combine_newsapi_text_handles_missing_fields(self):
        self.assertEqual(nextractor._combine_newsapi_text("Only a description.", ""), "Only a description.")
        self.assertEqual(nextractor._combine_newsapi_text("", "Only content. [+5 chars]"), "Only content.")
        self.assertEqual(nextractor._combine_newsapi_text("", ""), "")

    def test_scrape_full_article_body_falls_back_to_table_cells(self):
        """A government bulletin or spec page often has its real content in a table, not <p> tags --
        a near-empty paragraph scrape should not be accepted as the final answer."""
        html = (
            "<html><body><table>"
            "<tr><td>CVE-2026-1234 affects widely used network appliances and was disclosed this week.</td></tr>"
            "<tr><td>Administrators are advised to apply the vendor patch as soon as possible.</td></tr>"
            "</table></body></html>"
        )
        resp = mock.Mock(status_code=200, content=html.encode())
        with mock.patch.object(nextractor.requests, "get", return_value=resp):
            body = nextractor._scrape_full_article_body("http://x")
        self.assertIn("CVE-2026-1234", body)
        self.assertIn("vendor patch", body)

    def test_api_fetch_articles_combines_description_and_content_when_scrape_fails(self):
        """When the full page can't be scraped, NewsAPI's own description and (truncation-marker
        stripped) content are combined instead of showing the raw '[+N chars]' artifact or settling
        for description alone -- more real text for the judge to actually verify the claim against."""
        payload = {
            "status": "ok",
            "articles": [{
                "title": "T", "url": "http://a", "source": {"name": "Pub"},
                "description": "Short summary sentence.",
                "content": "Short summary sentence. A bit more detail follows here. [+899 chars]",
                "publishedAt": "2026-09-01",
            }],
        }
        resp = mock.Mock(status_code=200)
        resp.json.return_value = payload
        with mock.patch.object(nextractor, "secret", return_value="fake-key"), \
             mock.patch.object(nextractor.requests, "get", return_value=resp), \
             mock.patch.object(nextractor, "_scrape_full_article_body", return_value=""):
            articles = nextractor._api_fetch_articles("q", 5, None, None)
        self.assertEqual(len(articles), 1)
        self.assertNotIn("chars]", articles[0].content)
        self.assertIn("more detail", articles[0].content)
        self.assertIn("Short summary sentence", articles[0].content)  # content is a superset of the description here

    def test_throttled_get_retries_on_429_and_then_succeeds(self):
        """This is the actual bug that was making every live request to Google News and GDELT come
        back HTTP 429: a batch of videos fires many requests at the same host in quick succession
        with no pacing at all. A 429 is retried with backoff instead of being treated as a dead end."""
        limited = mock.Mock(status_code=429, headers={"Retry-After": "0"})
        ok = mock.Mock(status_code=200, headers={})
        with mock.patch.object(nextractor.requests, "get", side_effect=[limited, ok]) as get:
            response = nextractor._throttled_get("https://news.google.com/search?q=x")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(get.call_count, 2)
        self.mock_sleep.assert_called()

    def test_throttled_get_gives_up_after_max_retries(self):
        limited = mock.Mock(status_code=429, headers={})
        with mock.patch.object(nextractor.requests, "get", return_value=limited) as get:
            response = nextractor._throttled_get("https://api.gdeltproject.org/api/v2/doc/doc")
        self.assertEqual(response.status_code, 429)
        self.assertEqual(get.call_count, 1 + nextractor._MAX_429_RETRIES)

    def test_throttled_get_paces_successive_calls_to_the_same_host(self):
        ok = mock.Mock(status_code=200, headers={})
        with mock.patch.object(nextractor.requests, "get", return_value=ok):
            nextractor._throttled_get("https://news.google.com/search?q=a")
            nextractor._throttled_get("https://news.google.com/search?q=b")
        self.mock_sleep.assert_called()
        self.assertGreater(self.mock_sleep.call_args[0][0], 0)

    def test_throttled_get_does_not_pace_unlisted_hosts(self):
        """Pacing is per-host, not global -- the many different publisher domains article bodies are
        fetched from must not be slowed down by the throttling applied to the three search/API hosts."""
        ok = mock.Mock(status_code=200, headers={})
        with mock.patch.object(nextractor.requests, "get", return_value=ok):
            nextractor._throttled_get("https://example-news-site.com/article-1")
            nextractor._throttled_get("https://example-news-site.com/article-2")
        self.mock_sleep.assert_not_called()


class PipelineTests(unittest.TestCase):
    ROW = {"Video ID": "v1", "Video Title": "Delhi AQI severe", "Description": "Air quality is bad", "Published At": "2026-09-01T00:00:00Z"}

    def test_context_window(self):
        start, end = pipeline.context_window("2026-09-10T00:00:00Z")
        self.assertEqual(start, "2026-09-07")
        self.assertLessEqual(end, pd.Timestamp.now(tz="UTC").date().isoformat())
        self.assertEqual(pipeline.context_window(None), (None, None))

    def test_find_context_falls_back_to_raw_title_when_translated_query_finds_nothing(self):
        """Each tier (translated query, raw title, a broadened 2-3 word slice) only runs when the
        ones before it came up thin, not on every video regardless of how it did -- running every
        tier unconditionally is what previously overwhelmed Google News and GDELT with requests and
        got every query rate-limited (HTTP 429)."""
        from_title = [nextractor.News("H1", "http://a", "p"), nextractor.News("H2", "http://b", "p")]
        with mock.patch.object(llm, "translate_query", return_value="short query"), \
             mock.patch.object(nextractor, "get_news_list", side_effect=[[], from_title]) as get_news:
            query, articles = pipeline.find_context("Some video title")
        self.assertEqual(get_news.call_count, 2)  # translated query found nothing, so the raw title was also tried
        self.assertEqual({a.url for a in articles}, {"http://a", "http://b"})

    def test_find_context_skips_further_search_once_two_or_more_articles_are_found(self):
        two_articles = [nextractor.News("H0", "http://0", "p"), nextractor.News("H1", "http://1", "p")]
        with mock.patch.object(llm, "translate_query", return_value="short query"), \
             mock.patch.object(nextractor, "get_news_list", return_value=two_articles) as get_news:
            pipeline.find_context("Some video title")
        get_news.assert_called_once()  # two hits is enough; no need for the raw-title or broadened tiers

    def test_find_context_tries_a_broadened_query_when_still_thin_after_two_tiers(self):
        """A single stray hit isn't enough to call it done -- a third, maximally broad search (just
        the first few words of the topic) is tried so a thin result doesn't mean an empty report."""
        broad_hits = [nextractor.News("H1", "http://a", "p"), nextractor.News("H2", "http://b", "p")]
        with mock.patch.object(llm, "translate_query", return_value="short air quality query"), \
             mock.patch.object(nextractor, "get_news_list", side_effect=[
                 [],  # translated query: nothing
                 [nextractor.News("H0", "http://0", "p")],  # raw title: one hit, still thin
                 broad_hits,  # broadened query: more
             ]) as get_news:
            query, articles = pipeline.find_context("Some video title")
        self.assertEqual(get_news.call_count, 3)
        self.assertEqual({a.url for a in articles}, {"http://0", "http://a", "http://b"})

    def test_broaden_takes_the_first_few_words(self):
        self.assertEqual(pipeline._broaden("Delhi air quality index worsens sharply"), "Delhi air quality")
        self.assertEqual(pipeline._broaden("onlyoneword"), "onlyoneword")

    def test_find_context_cleans_pipe_separated_branding_out_of_the_query(self):
        """Real YouTube titles look like 'Topic | Channel Name | #tag', where the pipes and hashtags
        are a poor search query as-is (some engines even read a bare '|' as an OR operator)."""
        noisy = "SSC CGL 2026 | Air Quality Index (AQI) & Air Pollutants | Ishendu Sir | Simplicrack"
        self.assertNotIn("|", pipeline.clean_query_text(noisy))
        with mock.patch.object(llm, "translate_query") as translate, \
             mock.patch.object(nextractor, "get_news_list", return_value=[nextractor.News("H", "http://a", "p")]):
            pipeline.find_context(noisy)
        cleaned_arg = translate.call_args[0][0]
        self.assertNotIn("|", cleaned_arg)

    def test_clean_query_text_strips_hashtags_and_separators(self):
        self.assertEqual(pipeline.clean_query_text("Delhi AQI worsens #shorts #airpollution"), "Delhi AQI worsens")
        self.assertEqual(pipeline.clean_query_text("Topic A | Topic B • Topic C"), "Topic A Topic B Topic C")

    def test_happy_path(self):
        art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "Delhi air quality index severe pollution", "2026-09-02")
        with mock.patch.object(llm, "extract_claim", return_value="Claim: AQI is severe"), \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI severe"), \
             mock.patch.object(llm, "summarize", return_value="AQI severe"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Green", "confidence": 0.8, "reasoning": "ok", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]):
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual((res["Status"], res["Articles Found"], res["Confidence"]), ("Green", 1, 0.8))
        self.assertGreater(res["Alignment"], 0)
        self.assertEqual(res["Articles"][0]["headline"], "AQI severe in Delhi")
        self.assertFalse(res["Used Transcript"])  # fetch_transcript wasn't mocked -> no package -> ""

    def test_transcript_is_folded_into_claim_extraction_when_available(self):
        """The richer source of a real, checkable claim is what the speaker actually says, not the
        title -- when a transcript is available it must reach extract_claim and be flagged so the
        report can show the person where the claim actually came from."""
        art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "Delhi air quality severe", "2026-09-02")
        with mock.patch.object(yextractor, "fetch_transcript", return_value="The AQI hit 420 today in Delhi."), \
             mock.patch.object(llm, "extract_claim") as extract_claim, \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
             mock.patch.object(llm, "summarize", return_value="summary"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Green", "confidence": 0.7, "reasoning": "ok", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]):
            extract_claim.return_value = "Claim: AQI hit 420"
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertTrue(res["Used Transcript"])
        self.assertIn("The AQI hit 420 today in Delhi.", extract_claim.call_args[0][0])

    def test_transcript_fetch_failure_does_not_block_analysis(self):
        """A transcript is a bonus signal, never a requirement -- if fetching it blows up for any
        reason, the video must still be analysed from title + description exactly as before."""
        art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "Delhi air quality severe", "2026-09-02")
        with mock.patch.object(yextractor, "fetch_transcript", side_effect=RuntimeError("boom")), \
             mock.patch.object(llm, "extract_claim", return_value="Claim: AQI severe"), \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
             mock.patch.object(llm, "summarize", return_value="summary"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Green", "confidence": 0.7, "reasoning": "ok", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]):
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Green")
        self.assertEqual(res["Error"], "")
        self.assertFalse(res["Used Transcript"])

    def test_low_confidence_yellow_can_be_escalated_to_red_by_the_coin_flip(self):
        art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "Delhi air quality severe", "2026-09-02")
        with mock.patch.object(llm, "extract_claim", return_value="Claim: AQI severe"), \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
             mock.patch.object(llm, "summarize", return_value="summary"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Yellow", "confidence": 0.3, "reasoning": "unclear", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]), \
             mock.patch.object(pipeline.random, "random", return_value=0.0):  # "wins" the coin flip
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Red")
        self.assertIn("unclear", res["Reasoning"])  # the judge's real reasoning is kept, not replaced
        self.assertIn("borderline", res["Reasoning"].lower())  # and honestly labelled as a coin-flip escalation

    def test_low_confidence_yellow_stays_yellow_when_the_coin_flip_misses(self):
        art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "Delhi air quality severe", "2026-09-02")
        with mock.patch.object(llm, "extract_claim", return_value="Claim: AQI severe"), \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
             mock.patch.object(llm, "summarize", return_value="summary"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Yellow", "confidence": 0.3, "reasoning": "unclear", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]), \
             mock.patch.object(pipeline.random, "random", return_value=0.99):  # "loses" the coin flip
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Yellow")

    def test_high_confidence_yellow_is_never_escalated(self):
        art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "Delhi air quality severe", "2026-09-02")
        with mock.patch.object(llm, "extract_claim", return_value="Claim: AQI severe"), \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
             mock.patch.object(llm, "summarize", return_value="summary"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Yellow", "confidence": 0.9, "reasoning": "clear enough", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]), \
             mock.patch.object(pipeline.random, "random", return_value=0.0):  # would "win" if it were eligible
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Yellow")

    def test_no_coverage_found_yellow_is_never_escalated_even_at_zero_confidence(self):
        """The 'no coverage found' case has Confidence=None, not a low number -- escalating it would
        contradict the project's own stated principle that missing coverage is not evidence of
        anything, so it must be structurally ineligible, not just unlikely."""
        with mock.patch.object(llm, "extract_claim", return_value="Claim: x"), \
             mock.patch.object(llm, "translate_query", return_value="q"), \
             mock.patch.object(nextractor, "get_news_list", return_value=[]), \
             mock.patch.object(pipeline.random, "random", return_value=0.0):
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Yellow")

    def test_green_and_red_verdicts_are_never_touched_by_escalation(self):
        for verdict, confidence in (("Green", 0.2), ("Red", 0.1)):
            art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "x", "2026-09-02")
            with mock.patch.object(llm, "extract_claim", return_value="Claim: AQI severe"), \
                 mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
                 mock.patch.object(llm, "summarize", return_value="summary"), \
                 mock.patch.object(llm, "judge", return_value={"verdict": verdict, "confidence": confidence, "reasoning": "r", "contradictions": [], "off_topic": False}), \
                 mock.patch.object(nextractor, "get_news_list", return_value=[art]), \
                 mock.patch.object(pipeline.random, "random", return_value=0.0):
                res = pipeline.analyse_video(dict(self.ROW))
            self.assertEqual(res["Status"], verdict)

    def test_escalation_can_be_disabled_via_config(self):
        art = nextractor.News("AQI severe in Delhi", "http://a", "Pub", "d", "x", "2026-09-02")
        with mock.patch.object(llm, "extract_claim", return_value="Claim: AQI severe"), \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
             mock.patch.object(llm, "summarize", return_value="summary"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Yellow", "confidence": 0.3, "reasoning": "unclear", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]), \
             mock.patch.object(pipeline.random, "random", return_value=0.0), \
             mock.patch.object(config, "BORDERLINE_YELLOW_ESCALATION_CHANCE", 0.0):
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Yellow")

    def test_tfidf_failure_does_not_block_the_judge(self):
        """Defense in depth: even though tfidf_scores no longer depends on any optional package,
        it's a supporting signal, not a required one, and must never be able to stop the LLM judge
        from running. (This class of bug previously made every video silently fall back to Yellow
        when the alignment step failed before the judge was called.)"""
        art = nextractor.News("Delhi AQI severe", "http://a", "Pub", "d", "Delhi air quality severe today", "2026-09-02")
        with mock.patch.object(llm, "extract_claim", return_value="Claim: AQI severe"), \
             mock.patch.object(llm, "translate_query", return_value="Delhi AQI"), \
             mock.patch.object(llm, "summarize", return_value="summary"), \
             mock.patch.object(llm, "judge", return_value={"verdict": "Red", "confidence": 0.9, "reasoning": "contradicts", "contradictions": [], "off_topic": False}), \
             mock.patch.object(nextractor, "get_news_list", return_value=[art]), \
             mock.patch("module.pipeline.tfidf_scores", side_effect=ModuleNotFoundError("No module named 'sklearn'")):
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Red")  # judge ran and its verdict was used, not the Yellow default
        self.assertEqual(res["Error"], "")
        self.assertIsNone(res["Alignment"])

    def test_no_news_is_yellow_not_red(self):
        with mock.patch.object(llm, "extract_claim", return_value="Claim: x"), \
             mock.patch.object(llm, "translate_query", return_value="q"), \
             mock.patch.object(nextractor, "get_news_list", return_value=[]):
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Yellow")
        self.assertIn("not evidence", res["Reasoning"])

    def test_llm_failure_is_contained(self):
        with mock.patch.object(llm, "extract_claim", side_effect=RuntimeError("rate limited")):
            res = pipeline.analyse_video(dict(self.ROW))
        self.assertEqual(res["Status"], "Yellow")
        self.assertIn("rate limited", res["Error"])

    def test_empty_video(self):
        res = pipeline.analyse_video({"Video ID": "v2", "Video Title": " ", "Description": ""})
        self.assertEqual(res["Status"], "Yellow")


class ConfigTests(unittest.TestCase):
    def test_secret_falls_back_to_env(self):
        with mock.patch.dict(os.environ, {"NS_TEST_KEY": "abc"}):
            self.assertEqual(config.secret("NS_TEST_KEY"), "abc")
        self.assertEqual(config.secret("NS_MISSING_KEY", "dflt"), "dflt")


class AutomateDetailReportTests(unittest.TestCase):
    """Regression test for the KeyError crash on Automate's Detail Report table.

    state.set_results() builds results_df from pipeline.analyse_video() output, which only ever
    carries 'Video ID' plus analysis fields -- never 'Video Title'/'Video URL'. The page must join
    back to the videos table for those columns rather than reading them off results_df directly.
    We can't import module.state here (it imports streamlit, not installed in this sandbox), so
    this mirrors state.set_results' shape and the exact merge pages/06Automate.py now performs.
    """

    def test_detail_report_merge_does_not_raise_keyerror(self):
        videos = pd.DataFrame({
            "Video ID": ["v1", "v2"],
            "Video Title": ["Air quality today", "City council meeting"],
            "Video URL": ["https://youtube.com/watch?v=v1", "https://youtube.com/watch?v=v2"],
        })
        # Shape of what pipeline.analyse_video returns, as state.set_results() would store it.
        results = pd.DataFrame([
            {"Video ID": "v1", "Status": "Green", "Error": ""},
            {"Video ID": "v2", "Status": "Yellow", "Error": ""},
        ])

        detail = videos[["Video ID", "Video Title", "Video URL"]].merge(results[["Video ID", "Status"]], on="Video ID")

        self.assertEqual(list(detail["Status"]), ["Green", "Yellow"])
        self.assertIn("Video Title", detail.columns)
        self.assertIn("Video URL", detail.columns)

    def test_results_dataframe_alone_lacks_video_title(self):
        """Documents the actual root cause: results_df never has these columns on its own."""
        results = pd.DataFrame([{"Video ID": "v1", "Status": "Green", "Error": ""}])
        self.assertNotIn("Video Title", results.columns)
        self.assertNotIn("Video URL", results.columns)


if __name__ == "__main__":
    unittest.main()
