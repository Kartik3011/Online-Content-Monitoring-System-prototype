"""YouTube Data API access. Batched so one fetch costs about 102 quota units, not 100 + 2 per video."""
from __future__ import annotations

import re
from datetime import datetime
from functools import lru_cache
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from module.config import secret

COLUMNS = [
    "Video ID", "Video Title", "Description", "Video URL", "Published At",
    "Channel Title", "Channel ID", "Channel Description", "Channel URL",
    "Subscriber Count", "Total Views", "Video Count",
    "Views", "Likes", "Comments",
    "Duration (s)", "Tags", "Category ID", "Language", "Channel Created", "Country",
]


class YouTubeError(RuntimeError):
    """A problem with the YouTube API that the user can act on."""


@lru_cache(maxsize=1)
def _client():
    from googleapiclient.discovery import build

    key = secret("YOUTUBE_API_KEY")
    if not key:
        raise YouTubeError("YOUTUBE_API_KEY is missing. Add it to .streamlit/secrets.toml or your Streamlit Cloud secrets.")
    return build("youtube", "v3", developerKey=key, cache_discovery=False)


def _run(request):
    from googleapiclient.errors import HttpError

    try:
        return request.execute()
    except HttpError as exc:
        if "quota" in str(exc).lower():
            raise YouTubeError("The YouTube API quota for today is used up. It resets at midnight Pacific time.") from exc
        raise YouTubeError(f"YouTube API error: {exc}") from exc


_DURATION = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?")


def _seconds(iso: Optional[str]) -> Optional[int]:
    match = _DURATION.fullmatch(iso or "")
    if not match:
        return None
    hours, minutes, seconds = (int(x or 0) for x in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def _number(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _chunks(items: List[str], size: int = 50) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _iso(moment: Optional[datetime]) -> Optional[str]:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ") if moment else None


def _search(yt, query, lat, lon, radius, max_results, start, end) -> List[dict]:
    found, seen, token = [], set(), None
    while len(found) < max_results:
        params = {"part": "snippet", "q": query, "type": "video", "maxResults": min(50, max_results - len(found))}
        if token:
            params["pageToken"] = token
        if start:
            params["publishedAfter"] = _iso(start)
        if end:
            params["publishedBefore"] = _iso(end)
        if lat is not None and lon is not None:
            params["location"] = f"{lat},{lon}"
            params["locationRadius"] = radius
        response = _run(yt.search().list(**params))
        for item in response.get("items", []):
            vid = (item.get("id") or {}).get("videoId")
            if vid and vid not in seen:
                seen.add(vid)
                found.append(item)
        token = response.get("nextPageToken")
        if not token:
            break
    return found[:max_results]


def _details(yt, resource: str, part: str, ids: List[str]) -> dict:
    out = {}
    for chunk in _chunks(ids):
        response = _run(getattr(yt, resource)().list(part=part, id=",".join(chunk)))
        for item in response.get("items", []):
            out[item["id"]] = item
    return out


def build_rows(search_items: List[dict], videos: dict, channels: dict) -> pd.DataFrame:
    """Merge search hits, video details and channel details into one flat table."""
    rows = []
    for item in search_items:
        vid = item["id"]["videoId"]
        detail = videos.get(vid, {})
        snippet = detail.get("snippet") or item.get("snippet") or {}  # videos.list has the full description
        stats = detail.get("statistics", {})
        cid = snippet.get("channelId") or item["snippet"].get("channelId", "")
        channel = channels.get(cid, {})
        c_snippet, c_stats = channel.get("snippet", {}), channel.get("statistics", {})
        hidden = c_stats.get("hiddenSubscriberCount", False)
        rows.append({
            "Video ID": vid,
            "Video Title": snippet.get("title", ""),
            "Description": snippet.get("description", ""),
            "Video URL": f"https://www.youtube.com/watch?v={vid}",
            "Published At": snippet.get("publishedAt") or item["snippet"].get("publishedAt"),
            "Channel Title": snippet.get("channelTitle", ""),
            "Channel ID": cid,
            "Channel Description": c_snippet.get("description", ""),
            "Channel URL": f"https://www.youtube.com/channel/{cid}",
            "Subscriber Count": None if hidden else _number(c_stats.get("subscriberCount")),
            "Total Views": _number(c_stats.get("viewCount")),
            "Video Count": _number(c_stats.get("videoCount")),
            "Views": _number(stats.get("viewCount")),
            "Likes": _number(stats.get("likeCount")),
            "Comments": _number(stats.get("commentCount")),
            "Duration (s)": _seconds((detail.get("contentDetails") or {}).get("duration")),
            "Tags": ", ".join((snippet.get("tags") or [])[:15]),
            "Category ID": snippet.get("categoryId", ""),
            "Language": snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or "",
            "Channel Created": c_snippet.get("publishedAt"),
            "Country": c_snippet.get("country", ""),
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def fetch_videos(
    query: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    radius: str = "50km",
    max_results: int = 10,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Search YouTube and return (dataframe, notes). Notes are messages worth showing the user."""
    yt = _client()
    notes: List[str] = []
    located = lat is not None and lon is not None

    items = _search(yt, query, lat, lon, radius, max_results, start_date, end_date)
    if located and not items:
        notes.append("No geotagged videos matched this city, so the search ran without a location filter.")
        items = _search(yt, query, None, None, radius, max_results, start_date, end_date)
    elif located and len(items) < max_results:
        notes.append(f"Only {len(items)} geotagged videos matched. YouTube's location filter only returns videos that carry a location.")

    if not items:
        return pd.DataFrame(columns=COLUMNS), notes

    video_ids = [i["id"]["videoId"] for i in items]
    videos = _details(yt, "videos", "snippet,statistics,contentDetails", video_ids)
    channel_ids = sorted({(videos.get(v, {}).get("snippet") or i["snippet"]).get("channelId", "") for v, i in zip(video_ids, items)} - {""})
    channels = _details(yt, "channels", "snippet,statistics", channel_ids)
    return build_rows(items, videos, channels), notes


# --------------------------------------------------------------------------- transcripts
def _snippet_text(snippet) -> str:
    if isinstance(snippet, dict):
        return str(snippet.get("text") or "")
    return str(getattr(snippet, "text", "") or "")


def fetch_transcript(video_id: str, max_chars: int = 4000) -> str:
    """Best-effort fetch of a video's own spoken-word captions, straight from YouTube's caption
    track. Deliberately NOT audio download + ffmpeg + a speech-to-text model: that pipeline is what
    breaks on Streamlit Community Cloud (no ffmpeg binary, and YouTube blocks the kind of audio
    download that would need it). This instead reads the text track YouTube already generated, via
    one small HTTP call -- the same category of request as the news-scraping calls already used
    elsewhere in this app, so it deploys the same way they do.

    Not every video has this available (captions can be disabled, or not generated yet, or the video
    private/age-restricted), so this must never raise -- it returns "" and the rest of the pipeline
    falls back to title + description alone, exactly as before this existed.

    One honest limitation: YouTube rate-limits or blocks requests from IP ranges it recognizes as
    cloud-provider infrastructure (the library's own docs name AWS, GCP and Azure); Streamlit Community
    Cloud's shared IPs could plausibly get the same treatment even though this call itself needs no
    ffmpeg and isn't blocked the way audio download is. If that happens the symptom is simply this
    function returning "" in production even though it works locally -- degrading safely to the old
    title+description-only behavior, never crashing. There's no code fix for that short of a paid
    rotating-proxy service (the library supports one via WebshareProxyConfig, if ever needed)."""
    if not video_id:
        return ""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        return ""

    try:
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)  # youtube-transcript-api >= 1.0
        except AttributeError:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)  # youtube-transcript-api < 1.0

        try:
            transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except Exception:  # noqa: BLE001 - no English track: any language is still useful, claim
            transcript = next(iter(transcript_list))  # extraction already translates when needed

        snippets = transcript.fetch()
        text = " ".join(_snippet_text(s).strip() for s in snippets if _snippet_text(s).strip())
        return text[:max_chars]
    except Exception:  # noqa: BLE001 - captions disabled/missing/private, a version mismatch in the
        return ""       # library's API, a network hiccup -- a transcript is a bonus signal, never
                         # allowed to block the rest of the pipeline.
