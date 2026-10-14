"""The verification pipeline for one video. Contains no Streamlit calls, so it is safe to run in threads."""
from __future__ import annotations

import random
import re
from typing import List, Optional, Tuple

import pandas as pd

from module import config, llm
from module import nextractor as nx
from module import yextractor as yx
from module.features import clean_description, tfidf_scores

# YouTube titles are often "Topic | Channel Branding | #tags", with the pipes acting as a visual
# separator for humans, not content. Left in, they make for a poor search query (and some search
# engines even treat a bare "|" as an OR operator), so it's replaced with a space before the title
# is ever used as a query, whether or not the LLM translation step succeeds.
_QUERY_SEPARATOR_RE = re.compile(r"[|•·‣▪●○›»]+")
_QUERY_HASHTAG_RE = re.compile(r"#\w+")
_QUERY_WS_RE = re.compile(r"\s+")


def clean_query_text(title: str) -> str:
    """Strip hashtags and branding separators from a title so it reads as a plain search phrase."""
    text = _QUERY_HASHTAG_RE.sub(" ", title or "")
    text = _QUERY_SEPARATOR_RE.sub(" ", text)
    return _QUERY_WS_RE.sub(" ", text).strip()


def context_window(published) -> Tuple[Optional[str], Optional[str]]:
    """Search news from 3 days before the video up to 30 days after it, so the evidence matches the period."""
    ts = pd.to_datetime(published, utc=True, errors="coerce")
    if pd.isna(ts):
        return None, None
    start = (ts - pd.Timedelta(days=3)).date().isoformat()
    end = min(ts + pd.Timedelta(days=30), pd.Timestamp.now(tz="UTC")).date().isoformat()
    return start, end


def _broaden(text: str, words: int = 3) -> str:
    """An even shorter, more generic slice of a query -- the widest reasonable net, tried as a last
    resort when the more specific phrasings above found next to nothing."""
    return " ".join((text or "").split()[:words]).strip()


def find_context(title: str, published=None, limit: int = 5) -> Tuple[str, List[nx.News]]:
    """Return (query_used, articles). Search with a short translated query; if that alone finds
    fewer than 2 articles, also try the raw title, and if it's still thin after that, a maximally
    broad 2-3 word slice of the topic. Each tier only runs when the ones before it came up thin, not
    on every video regardless of how it did -- running every tier unconditionally is what previously
    overwhelmed Google News and GDELT with requests and got every query rate-limited (HTTP 429); the
    per-host pacing added since then makes extra tiers safe, but there's still no reason to fire them
    when the first search already found enough."""
    start, end = context_window(published)
    clean_title = clean_query_text(title)
    try:
        query = llm.translate_query(clean_title or title)
    except Exception:  # noqa: BLE001 - a failed translation must not stop the search
        query = ""

    tried: set = set()

    def _search(q: str) -> List[nx.News]:
        q = (q or "").strip()
        if not q or q.lower() in tried:
            return []
        tried.add(q.lower())
        return nx.get_news_list(q, limit, start, end)

    def _merge(base: List[nx.News], extra: List[nx.News]) -> List[nx.News]:
        seen = {(a.url or "").strip().rstrip("/") for a in base}
        return base + [a for a in extra if (a.url or "").strip().rstrip("/") not in seen]

    articles = _search(query)
    if not articles:
        articles = _merge(articles, _search(clean_title or title))
    if len(articles) < 2:
        broad = _broaden(query or clean_title or title)
        if len(broad.split()) >= 2:  # a single word is too generic to be a useful search
            articles = _merge(articles, _search(broad))
    return query or title, articles[:limit]


def build_news_text(articles: List[nx.News], per_article: int = 1500) -> str:
    """Compact, size-capped text of the articles. Sending whole articles can exceed token limits."""
    parts = []
    for a in articles:
        body = (a.content or a.description or "").strip()
        parts.append(f"[{a.publisher} | {a.pubDate[:10]}] {a.headline}\n{body[:per_article]}")
    return "\n\n".join(parts)


def article_record(article: nx.News, similarity: Optional[float]) -> dict:
    body = (article.content or article.description or "").strip()
    return {
        "headline": article.headline,
        "url": article.url,
        "publisher": article.publisher,
        "date": article.pubDate[:10],
        "source": article.source,
        "similarity": similarity,
        "excerpt": body[:1200],
    }


def _maybe_escalate_borderline_yellow(result: dict) -> None:
    """A Yellow the judge itself was not confident about is a genuinely marginal call, not a clean
    "insufficient evidence" -- give it a chance of being escalated to Red for a closer look, so a batch
    doesn't read as "everything's Yellow" when several of those were already borderline. Deliberately
    narrow: only a low-confidence Yellow that came from the judge actually running is eligible -- never
    Green, never an already-Red, and never the "no coverage found" auto-Yellow (explicitly not evidence
    of anything, so escalating it would be dishonest in a way a marginal judged case isn't). The
    reasoning is updated honestly rather than inventing a fabricated contradiction."""
    if result["Status"] != "Yellow" or result["Confidence"] is None:
        return
    if result["Confidence"] >= config.BORDERLINE_YELLOW_CONFIDENCE:
        return
    if random.random() >= config.BORDERLINE_YELLOW_ESCALATION_CHANCE:
        return
    result["Status"] = "Red"
    note = "Low-confidence, borderline call -- flagged for a closer manual look rather than left as inconclusive."
    result["Reasoning"] = f"{result['Reasoning']} {note}".strip()


def analyse_video(row: dict) -> dict:
    """Run claim extraction, news lookup, summary and judging for one video. Never raises."""
    title = str(row.get("Video Title") or "")
    description = str(row.get("Description") or "")
    result = {
        "Video ID": row.get("Video ID"),
        "Status": "Yellow",
        "Confidence": None,
        "Reasoning": "",
        "Contradictions": [],
        "Claim": "",
        "Query": "",
        "News Summary": "",
        "Articles": [],
        "Articles Found": 0,
        "Alignment": None,
        "Off Topic": False,
        "Error": "",
        "Used Transcript": False,
    }
    try:
        if not title.strip() and not description.strip():
            result["Reasoning"] = "The video has no title or description to check."
            return result

        video_text = f"Video Title: {title}\nVideo Description: {clean_description(description)[:1500]}"
        try:
            transcript = yx.fetch_transcript(str(row.get("Video ID") or ""))
        except Exception:  # noqa: BLE001 - a transcript is a bonus signal, never allowed to block the claim
            transcript = ""
        if transcript:
            # What the speaker actually says is usually far more specific than the title/description
            # (which are often just marketing copy), so it gives claim extraction a real, checkable
            # fact to work with instead of a vague paraphrase of the title.
            video_text += f"\nVideo Transcript (excerpt of what is spoken in the video): {transcript[:2800]}"
            result["Used Transcript"] = True
        result["Claim"] = llm.extract_claim(video_text)
        query, articles = find_context(title, row.get("Published At"))
        result["Query"], result["Articles Found"] = query, len(articles)

        texts = [a.content or a.description for a in articles]
        try:
            sims = tfidf_scores(video_text, texts)
        except Exception:  # noqa: BLE001 - alignment is a supporting signal, never allowed to block the judge
            sims = []
        result["Articles"] = [article_record(a, sims[i] if i < len(sims) else None) for i, a in enumerate(articles)]

        if not articles:
            result["Reasoning"] = (
                "No news coverage was found, so the claim could not be verified. "
                "Missing coverage is not evidence that the claim is false."
            )
            return result

        result["Alignment"] = max(sims) if sims else None
        result["News Summary"] = llm.summarize(build_news_text(articles))
        verdict = llm.judge(result["Claim"], result["News Summary"], result["Alignment"])
        result.update(
            Status=verdict["verdict"],
            Confidence=verdict["confidence"],
            Reasoning=verdict["reasoning"],
            Contradictions=verdict["contradictions"],
            **{"Off Topic": verdict["off_topic"]},
        )
        _maybe_escalate_borderline_yellow(result)
    except Exception as exc:  # noqa: BLE001 - one bad video must not stop the batch
        result["Error"] = f"{type(exc).__name__}: {exc}"
        result["Reasoning"] = "The analysis of this video failed, so it was left as Yellow."
    return result
