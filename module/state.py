"""Session-state helpers.

Data lives in each visitor's own session instead of shared CSV files on the server disk,
so two people using the deployed app can no longer overwrite each other's results.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from module import config
from module.features import add_features, add_signals

VERDICT_OPTIONS = ["Green", "Yellow", "Red"]


def set_videos(df: pd.DataFrame, params: dict) -> None:
    """Store a new dataset. Any earlier analysis belongs to the old data, so it is cleared."""
    st.session_state["videos_df"] = df
    st.session_state["fetch_params"] = params
    st.session_state.pop("results_df", None)
    st.session_state.pop("context", None)


def get_videos() -> Optional[pd.DataFrame]:
    df = st.session_state.get("videos_df")
    return df if df is not None and not df.empty else None


def get_params() -> dict:
    return st.session_state.get("fetch_params", {})


def set_results(results: List[dict]) -> None:
    df = pd.DataFrame(results)
    df["Human Verdict"] = None
    df["Run ID"] = uuid.uuid4().hex[:8]
    df["Analysed At"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    df["Prompt Version"] = config.PROMPT_VERSION
    df["Extraction Model"] = config.FAST_MODEL
    df["Judge Model"] = config.JUDGE_MODEL
    st.session_state["results_df"] = df


def get_results() -> Optional[pd.DataFrame]:
    df = st.session_state.get("results_df")
    return df if df is not None and not df.empty else None


def save_human_verdicts(verdicts: Dict[str, Optional[str]]) -> None:
    """verdicts maps Video ID to 'Green', 'Yellow', 'Red' or None."""
    results = get_results()
    if results is None:
        return
    merged = dict(zip(results["Video ID"], results["Human Verdict"]))
    merged.update({vid: (v if v in VERDICT_OPTIONS else None) for vid, v in verdicts.items()})
    results["Human Verdict"] = results["Video ID"].map(merged)
    st.session_state["results_df"] = results


def enriched() -> Optional[pd.DataFrame]:
    """Videos + engineered features, plus results and triage signals once an analysis has run."""
    videos = get_videos()
    if videos is None:
        return None
    out = add_features(videos)
    results = get_results()
    if results is not None:
        out = add_signals(out.merge(results, on="Video ID", how="left"))
    return out


def export_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten list columns so the table can be written to CSV."""
    out = df.drop(columns=[c for c in ("Articles",) if c in df], errors="ignore").copy()
    if "Contradictions" in out:
        out["Contradictions"] = out["Contradictions"].map(lambda x: " | ".join(x) if isinstance(x, list) else "")
    return out


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return export_frame(df).to_csv(index=False).encode("utf-8")
