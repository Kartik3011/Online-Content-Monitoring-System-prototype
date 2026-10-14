# NetShield 2.0

Checks YouTube videos against news coverage from the same period and returns a
Green / Yellow / Red verdict with a confidence score and reasoning.

## What changed from v1

See `CHANGES.md` for the full list. Headlines: session-based storage instead of
shared CSV files on disk (safe for multiple concurrent users), parallel video
processing, a judge model that returns structured JSON instead of a single
word, a human-review queue with precision/recall/F1 evaluation, and a new
**Analytics Lab** page aimed at a data-scientist audience.

## Setup

1. `pip install -r requirements.txt`
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:
   - `GROQ_API_KEY` — from [console.groq.com](https://console.groq.com)
   - `YOUTUBE_API_KEY` — a YouTube Data API v3 key from Google Cloud Console
   - `NEWS_API_KEY` — optional, only used if the Google News scrape returns nothing
3. `streamlit run Dashboard.py`

No `yt-dlp`, `ffmpeg` or manual binary downloads are required — v2 analyses
title, description and channel metadata rather than transcribing audio. See
"What was cut" below for why.

## Project layout

```
Dashboard.py                 Home page + help assistant
pages/
  01Request Analysis.py      Fetch videos from YouTube
  02Social Content Report.py Raw data, filters, charts
  03Content Forensic.py      Per-video text signals + on-demand AI summary
  04Current Context Report.py Automatic per-video news lookup (cached for the session)
  05Account Report.py        Status breakdown chart, full results table, red-flagged list
  06Automate.py               The AI pipeline — runs automatically, one video at a time, cached for the session
  07Analytics Lab.py          Channel breakdown, human review queue, feature distributions,
                               anomaly detection, text-similarity baseline, evaluation metrics
module/
  config.py       Model names, prompt version, secret lookup
  llm.py          One Groq/OpenAI client, retries, JSON verdict parsing
  nextractor.py   News search (Google News + NewsAPI fallback), cached, with a diagnose() tool
  yextractor.py   Batched YouTube Data API calls -> DataFrame
  features.py     Pure pandas/sklearn feature engineering (unit-tested)
  pipeline.py     Per-video analysis, thread-safe, no Streamlit calls
  state.py        Session-state storage (per-visitor, not shared files)
  ui.py           Shared CSS, page header, callouts, verdict pills, charts
tests/
  test_core.py    42 unit tests, run with:
                  python -m unittest discover -s tests -v
```

## Design choices carried over from the original project on purpose

A few pages deliberately match the original's interaction pattern rather than the more
"click to run" style of an earlier rewrite, because that's how the original behaves and
that behavior was specifically asked to be kept:
- **Current Context Report** and **Automate** run automatically over every fetched video —
  no button needed to start them. Results are cached in the session so navigating back to
  either page doesn't silently re-run (and re-bill) the whole pipeline; each has a
  Refresh/Re-run button for an explicit re-pass.
- **Automate** processes videos one at a time (not in parallel), so you can watch each
  video's claim, news summary and verdict appear in sequence, matching the original's feel.
- **Content Forensic** is the one place that stays opt-in: summarizing a video's description
  only happens when you press the button for that video, rather than running an LLM call
  for every video on every page load.
- A video whose title doesn't match what it actually discusses (e.g. devotional content
  under a factual news tag) is treated as content abuse and forced to **Red** — the
  original project's explicit rule (`module/config.py: TAG_ABUSE_IS_RED`).

## What was cut, and why

The original README advertised audio transcription (yt-dlp + ffmpeg +
AssemblyAI) as part of the pipeline, but no code in the repository actually
called AssemblyAI or ran a transcription step — `Content Forensic` only
cleaned and summarized the text description. Re-adding real transcription is
possible but changes the deployment story (binary dependencies, per-minute
transcription cost, much longer run times) enough that it's a deliberate
follow-up rather than something to silently keep half-implemented. Everything
in this version is a feature that actually runs.

## Known limitations

- Verdicts come from an LLM reading text; they are a triage signal, not proof.
  A Red verdict always deserves a human look before any account action.
- Only title, description and channel bio are analysed — not video or audio content.
- News coverage can be genuinely absent for local or very recent stories; that
  is scored Yellow, never Red.
- Data lives in each browser session, not a database. Closing the tab clears it
  (by design, so two visitors never see or overwrite each other's data).
