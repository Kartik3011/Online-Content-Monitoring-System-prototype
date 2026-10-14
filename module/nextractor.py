"""News context retrieval: Google News page scrape first, NewsAPI as the fallback."""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterator, List, Optional
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

from module.config import secret

NEWS_API_ENDPOINT = "https://newsapi.org/v2/everything"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
CACHE_TTL_SECONDS = 3600
_CACHE: dict = {}

# --------------------------------------------------------------------------- request pacing
# A whole batch of videos runs in one process, one after another. Without pacing, 8 videos is
# enough to fire a dozen-plus requests at the same search endpoint within a few seconds -- exactly
# the pattern that gets a shared cloud IP rate-limited (HTTP 429), which is what was happening here:
# every single Google News and GDELT call was coming back 429. Spacing is per *host*, not global, so
# the many different publisher domains article bodies are fetched from are never slowed down by it.
_THROTTLE_LOCK = threading.Lock()
_LAST_CALL: dict = {}
_MIN_INTERVAL = {"news.google.com": 5.0, "api.gdeltproject.org": 3.0, "newsapi.org": 1.0}
_MAX_429_RETRIES = 2


def _throttled_get(url: str, **kwargs):
    """requests.get with a per-host minimum interval and a short backoff-and-retry on HTTP 429."""
    host = urlparse(url).netloc
    min_interval = _MIN_INTERVAL.get(host, 0.0)
    if min_interval:
        with _THROTTLE_LOCK:
            now = time.time()
            wait = max(0.0, _LAST_CALL.get(host, 0.0) + min_interval - now)
            _LAST_CALL[host] = now + wait
        if wait:
            time.sleep(wait)

    response = requests.get(url, **kwargs)
    attempt = 0
    while response.status_code == 429 and attempt < _MAX_429_RETRIES:
        retry_after = response.headers.get("Retry-After", "")
        delay = float(retry_after) if retry_after.strip().isdigit() else 3.0 * (2**attempt)
        print(f"{host} returned HTTP 429 (rate limited); waiting {delay:.1f}s before retry {attempt + 1}/{_MAX_429_RETRIES}")
        time.sleep(delay)
        if min_interval:
            with _THROTTLE_LOCK:
                _LAST_CALL[host] = time.time()
        response = requests.get(url, **kwargs)
        attempt += 1
    return response


@dataclass
class News:
    """One news article. Field order matches the original class."""

    headline: str
    url: str
    publisher: str
    description: str = ""
    content: str = ""
    pubDate: str = ""
    source: str = ""


# --------------------------------------------------------------------------- Google News
def _iter_nodes(obj) -> Iterator[dict]:
    """Yield every dict in a JSON-LD document, including nodes inside @graph."""
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_nodes(item)
    elif isinstance(obj, dict):
        yield obj
        graph = obj.get("@graph")
        if isinstance(graph, list):
            yield from _iter_nodes(graph)


def _publisher(node: dict) -> str:
    pub = node.get("publisher")
    if isinstance(pub, list) and pub:
        pub = pub[0]
    if isinstance(pub, dict):
        return str(pub.get("name") or pub.get("url") or "N/A")
    return str(pub) if pub else "N/A"


def article_from_html(html: bytes | str, fallback_url: str = "N/A") -> Optional[News]:
    """Find the JSON-LD node that carries articleBody and turn it into a News object."""
    soup = BeautifulSoup(html, "html.parser")
    for script in soup.find_all("script", {"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _iter_nodes(data):
            if node.get("articleBody"):
                return News(
                    headline=str(node.get("headline") or "N/A"),
                    url=str(node.get("url") or fallback_url),
                    publisher=_publisher(node),
                    description=str(node.get("description") or ""),
                    content=str(node.get("articleBody") or ""),
                    pubDate=str(node.get("datePublished") or ""),
                    source="Google News",
                )
    return None


def _article_from_paragraphs(html: bytes | str, url: str) -> Optional[News]:
    """Fallback for pages with no JSON-LD articleBody: pull the <title> and the main paragraph text."""
    soup = BeautifulSoup(html, "html.parser")
    block = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
    paragraphs = (block or soup).find_all("p")
    body = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text().split()) > 10).strip()
    if len(body.split()) < 25:  # too little text to be a real article; likely a paywall or nav-only page
        return None
    title_tag = soup.find("meta", {"property": "og:title"}) or soup.find("title")
    headline = (title_tag.get("content") if title_tag and title_tag.name == "meta" else title_tag.get_text(strip=True)) if title_tag else "N/A"
    site_tag = soup.find("meta", {"property": "og:site_name"})
    return News(headline=headline or "N/A", url=url, publisher=(site_tag.get("content") if site_tag else "N/A") or "N/A",
                content=body, source="Google News (fallback parser)")


# Phrases that show up on consent walls, bot-check pages and error pages rather than real articles.
# A page matching one of these must not become a headline-only stub -- it isn't coverage of anything.
_BLOCK_PAGE_RE = re.compile(
    r"\b(?:just a moment|attention required|are you a human|verify you are human|access denied|"
    r"page not found|404 error|this site can.?t be reached|consent\.google|before you continue)\b",
    re.I,
)


def _stub_from_title(html: bytes | str, url: str) -> Optional[News]:
    """Last resort when neither JSON-LD nor paragraph text is available: keep the headline and
    publisher alone. A headline-only record still lets the judge see that real coverage exists,
    which beats silently dropping the link and reporting zero articles for the video."""
    soup = BeautifulSoup(html, "html.parser")
    title_tag = soup.find("meta", {"property": "og:title"}) or soup.find("title")
    headline = (title_tag.get("content") if title_tag and title_tag.name == "meta" else title_tag.get_text(strip=True)) if title_tag else ""
    headline = (headline or "").strip()
    if not headline or _BLOCK_PAGE_RE.search(headline) or _BLOCK_PAGE_RE.search(str(html)[:500]):
        return None
    desc_tag = soup.find("meta", {"property": "og:description"}) or soup.find("meta", {"name": "description"})
    description = (desc_tag.get("content") if desc_tag else "") or ""
    site_tag = soup.find("meta", {"property": "og:site_name"})
    return News(headline=headline, url=url, publisher=(site_tag.get("content") if site_tag else "N/A") or "N/A",
                description=description.strip(), source="Google News (headline only)")


def _fetch_article(url: str) -> Optional[News]:
    try:
        response = _throttled_get(url, headers=HEADERS, allow_redirects=True, timeout=10)
        if response.status_code != 200:
            print(f"news article fetch: HTTP {response.status_code} for {url}")
            return None
        article = article_from_html(response.content, fallback_url=url)
        article = article or _article_from_paragraphs(response.content, url)
        article = article or _stub_from_title(response.content, url)
        return article
    except requests.RequestException as exc:
        print(f"news article fetch failed for {url}: {type(exc).__name__}: {exc}")
        return None


def google_links(html: bytes | str, cap: int = 25) -> List[str]:
    """Pull article links out of the Google News search page. Returns [] if the layout changed."""
    soup = BeautifulSoup(html, "html.parser")
    script = soup.find("script", {"class": "ds:2"})
    if not script or not script.string:
        print("google_links: no <script class=\"ds:2\"> found on the page (layout may have changed, or the page is a consent/CAPTCHA wall)")
        return []
    data = script.string
    start, end = data.find("data:["), data.rfind("]")
    if start == -1 or end == -1:
        print("google_links: 'data:[' marker not found inside the ds:2 script")
        return []
    try:
        items = json.loads(data[start + 5 : end + 1])[1][0][:cap]
    except (json.JSONDecodeError, IndexError, TypeError, KeyError) as exc:
        print(f"google_links: could not parse the ds:2 payload: {type(exc).__name__}: {exc}")
        return []

    links = []
    for item in items:
        try:
            if len(item) == 2:
                links.append(item[1][2][0][6])
            elif len(item) == 8:
                links.append(item[0][6])
        except (IndexError, TypeError, KeyError):
            continue  # one odd item must not discard the rest
    return links


def _scrape_google_news(query: str, limit: int) -> List[News]:
    url = f"https://news.google.com/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN%3Aen"
    try:
        response = _throttled_get(url, headers=HEADERS, allow_redirects=True, timeout=10)
        if response.status_code != 200:
            print(f"Google News search returned HTTP {response.status_code} for query '{query}'")
            return []
        links = google_links(response.content)[: limit * 3]
        if not links:
            print(f"Google News search returned 0 links for query '{query}'")
            return []
        with ThreadPoolExecutor(max_workers=5) as pool:
            fetched = list(pool.map(_fetch_article, links))
        articles = [a for a in fetched if a][:limit]
        if not articles:
            print(f"Google News found {len(links)} links for '{query}' but none of the article pages could be parsed")
        return articles
    except Exception as exc:  # noqa: BLE001
        print(f"Google News scraper error for '{query}': {exc}")
        return []


# --------------------------------------------------------------------------- NewsAPI
# NewsAPI's own free-tier "content" field is truncated to ~200 characters with a literal trailing
# "... [+912 chars]" marker baked into the string -- not useful text, just noise if shown as-is.
_NEWSAPI_TRUNCATION_RE = re.compile(r"\s*\[\+\d+\s*chars\]\s*$")


def _clean_newsapi_text(text: str) -> str:
    return _NEWSAPI_TRUNCATION_RE.sub("", text or "").strip()


def _combine_newsapi_text(description: str, raw_content: str) -> str:
    """NewsAPI's "description" and "content" fields often overlap (content is usually a longer lead
    that includes the description's own wording) but are sometimes genuinely different snippets.
    Avoid gluing a duplicate together, but don't throw away either one when they actually differ."""
    description = (description or "").strip()
    snippet = _clean_newsapi_text(raw_content)
    if not snippet:
        return description
    if not description or snippet.startswith(description) or description in snippet:
        return snippet  # snippet is the same text or a superset of it: more words, not a duplicate
    if description.startswith(snippet) or snippet in description:
        return description
    return f"{description} {snippet}".strip()


def _scrape_full_article_body(url: str) -> str:
    """Best-effort extraction of the main text of an article page."""
    try:
        response = _throttled_get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        block = soup.find("article") or soup.find("main") or soup.find("div", class_="content")
        paragraphs = (block or soup).find_all("p")
        body = " ".join(p.get_text(" ", strip=True) for p in paragraphs if len(p.get_text().split()) > 10).strip()
        if len(body.split()) < 40:
            # Some pages (government bulletins, spec sheets, listicles) carry their real content in
            # tables or lists rather than <p> tags. A near-empty paragraph scrape is a sign to also
            # pull from those, rather than accepting a few words and falling back to a worse source.
            extra = (block or soup).find_all(["li", "td"])
            extra_text = " ".join(e.get_text(" ", strip=True) for e in extra if len(e.get_text().split()) > 4)
            body = (body + " " + extra_text).strip()
        return body
    except Exception:  # noqa: BLE001
        return ""


def _newsapi_call(params: dict) -> Optional[dict]:
    try:
        response = _throttled_get(NEWS_API_ENDPOINT, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data if data.get("status") == "ok" else None
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 426:
            print(
                "NewsAPI error: 426 Upgrade Required. This is NewsAPI's own free-plan restriction, not a bug here: "
                "the free Developer plan is for local testing only and actively refuses requests from a deployed "
                "app. A paid NewsAPI plan is required for it to work once deployed; Google News and GDELT do not "
                "have this restriction."
            )
        else:
            print(f"NewsAPI error: {exc}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"NewsAPI error: {exc}")
        return None


def _api_fetch_articles(query: str, limit: int, from_date: Optional[str], to_date: Optional[str]) -> List[News]:
    key = secret("NEWS_API_KEY")
    if not key:
        print("NEWS_API_KEY is not configured. Skipping the NewsAPI fallback.")
        return []

    params = {"q": query, "apiKey": key, "sortBy": "relevancy", "language": "en", "pageSize": limit}
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    data = _newsapi_call(params)
    if data is None and (from_date or to_date):
        # Free plans limit how far back you can search. Retry without the date window.
        params.pop("from", None)
        params.pop("to", None)
        data = _newsapi_call(params)
    if not data or not data.get("articles"):
        return []

    articles = data["articles"]
    urls = [a.get("url") or "" for a in articles]
    with ThreadPoolExecutor(max_workers=min(5, len(urls))) as pool:
        bodies = list(pool.map(_scrape_full_article_body, urls))

    news_list = []
    for a, body in zip(articles, bodies):
        description = (a.get("description") or "").strip()
        if not body:
            # The full page couldn't be scraped (blocked, paywalled, or just no <p> tags to find).
            # NewsAPI's own "content" field is truncated to ~200 characters with a "[+N chars]"
            # marker baked in -- combining it (cleaned) with "description" gives the judge more real
            # text to verify against than either field alone, without pretending it's the full article.
            body = _combine_newsapi_text(description, a.get("content") or "")
        news_list.append(News(
            headline=a.get("title") or "N/A",
            url=a.get("url") or "N/A",
            publisher=(a.get("source") or {}).get("name") or "N/A",
            description=description,
            content=body or description,
            pubDate=a.get("publishedAt") or "",
            source="NewsAPI",
        ))
    return news_list


# --------------------------------------------------------------------------- GDELT
GDELT_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"


def _gdelt_articles(query: str, limit: int) -> List[News]:
    """GDELT's Doc API: a free, key-less, global news index. No official uptime guarantee, but it
    needs no account and nothing to configure, so it works the same in local dev and on Streamlit
    Cloud -- a useful third source when a NEWS_API_KEY isn't set or Google's page layout blocks the
    scraper for a particular host. It only indexes roughly the last three months of coverage."""
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(max(limit * 3, 10), 50),
        "format": "json",
        "sort": "hybridrel",
    }
    try:
        response = _throttled_get(GDELT_ENDPOINT, params=params, headers=HEADERS, timeout=12)
        if response.status_code != 200:
            print(f"GDELT returned HTTP {response.status_code} for query '{query}'")
            return []
        try:
            data = response.json()
        except ValueError:
            print(f"GDELT returned a non-JSON response for query '{query}' (likely rate-limited)")
            return []
    except requests.RequestException as exc:
        print(f"GDELT request failed for '{query}': {type(exc).__name__}: {exc}")
        return []

    items = (data or {}).get("articles") or []
    if not items:
        return []
    items = items[:limit]

    urls = [item.get("url") or "" for item in items]
    with ThreadPoolExecutor(max_workers=min(5, len(urls)) or 1) as pool:
        bodies = list(pool.map(_scrape_full_article_body, urls))

    articles = []
    for item, body in zip(items, bodies):
        url = item.get("url") or ""
        headline = item.get("title") or "N/A"
        if not url or not headline or headline == "N/A":
            continue
        seen = str(item.get("seendate") or "")
        pub_date = f"{seen[0:4]}-{seen[4:6]}-{seen[6:8]}" if len(seen) >= 8 else ""
        articles.append(News(
            headline=headline, url=url, publisher=item.get("domain") or "N/A",
            content=body, pubDate=pub_date, source="GDELT",
        ))
    return articles


# --------------------------------------------------------------------------- merging
def _dedupe_articles(articles: List[News]) -> List[News]:
    """Combine results from several sources, keeping the richest version of each duplicate (the
    same story often turns up from more than one source) and ranking full-text articles above
    headline-only stubs."""
    best: dict = {}
    for article in articles:
        key = (article.url or "").strip().rstrip("/") or f"{article.publisher}|{article.headline}".lower()
        current = best.get(key)
        if current is None or len(article.content or "") > len(current.content or ""):
            best[key] = article
    ordered = list(best.values())
    ordered.sort(key=lambda a: len((a.content or a.description or "").strip()) > 0, reverse=True)
    return ordered


# --------------------------------------------------------------------------- diagnostics
def diagnose(query: str) -> dict:
    """Run one query through every step of get_news_list and report exactly what happened at each step.
    Meant for the Current Context Report page's "Test the news search" tool, or for manual debugging."""
    trace: dict = {"query": query}

    url = f"https://news.google.com/search?q={quote(query)}&hl=en-IN&gl=IN&ceid=IN%3Aen"
    try:
        response = _throttled_get(url, headers=HEADERS, allow_redirects=True, timeout=10)
        trace["google_http_status"] = response.status_code
        if response.status_code == 200:
            links = google_links(response.content)
            trace["google_ds2_script_found"] = bool(BeautifulSoup(response.content, "html.parser").find("script", {"class": "ds:2"}))
            trace["google_links_found"] = len(links)
            if links:
                sample = links[:5]
                fetched = [_fetch_article(u) for u in sample]
                trace["google_sample_checked"] = len(sample)
                trace["google_sample_parsed"] = sum(1 for a in fetched if a)
        else:
            trace["google_note"] = "A non-200 status usually means a block page, rate limit, or consent wall."
    except requests.RequestException as exc:
        trace["google_error"] = f"{type(exc).__name__}: {exc}"

    key = secret("NEWS_API_KEY")
    trace["newsapi_key_configured"] = bool(key)
    if key:
        data = _newsapi_call({"q": query, "apiKey": key, "pageSize": 5, "language": "en", "sortBy": "relevancy"})
        trace["newsapi_ok"] = data is not None
        trace["newsapi_articles_found"] = len(data.get("articles", [])) if data else 0
        if data is None:
            trace["newsapi_note"] = (
                "If this fails here but works when you run the app locally, it's almost certainly NewsAPI's free "
                "Developer plan refusing a deployed app (HTTP 426) -- see the server logs. That plan is restricted "
                "to local testing by NewsAPI itself; it needs a paid plan to work once deployed."
            )

    try:
        gdelt = _gdelt_articles(query, 5)
        trace["gdelt_articles_found"] = len(gdelt)
    except Exception as exc:  # noqa: BLE001
        trace["gdelt_error"] = f"{type(exc).__name__}: {exc}"

    return trace


# --------------------------------------------------------------------------- public API
def get_news_list(
    query: str,
    limit: int = 5,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> List[News]:
    """Look up news for a query across every source available, and merge what comes back.

    Three independent sources are tried, not as a strict fallback chain but combined: NewsAPI (an
    official, stable API, used when a NEWS_API_KEY is configured), a Google News page scrape (wide
    coverage, but a scrape that can silently break on layout changes or cloud-host rate limiting),
    and GDELT's Doc API (free, key-less, so it works the same locally and on a hosted deployment).
    Any one of them failing or returning nothing still leaves the other two, which is what keeps the
    hit rate up in practice -- Google alone, in particular, drops a fair number of links whose target
    page blocks scrapers or needs JavaScript. Results are de-duplicated by URL. from_date / to_date
    (YYYY-MM-DD) only apply to NewsAPI, which is the only source that accepts a date window.
    """
    query = (query or "").strip()
    if not query:
        return []

    key = (query.lower(), limit, from_date, to_date)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_SECONDS:
        return hit[1]

    fetch_limit = max(limit, 5)  # fetch a bit extra per source so de-duping still leaves `limit` good results
    jobs = [lambda: _scrape_google_news(query, fetch_limit), lambda: _gdelt_articles(query, fetch_limit)]
    if secret("NEWS_API_KEY"):
        jobs.append(lambda: _api_fetch_articles(query, fetch_limit, from_date, to_date))

    gathered: List[News] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        for future in [pool.submit(job) for job in jobs]:
            try:
                gathered.extend(future.result())
            except Exception as exc:  # noqa: BLE001 - one source failing must not take the others down
                print(f"a news source raised while fetching '{query}': {type(exc).__name__}: {exc}")

    results = _dedupe_articles(gathered)[:limit]
    if results:  # never cache a failure
        _CACHE[key] = (time.time(), results)
    return results
