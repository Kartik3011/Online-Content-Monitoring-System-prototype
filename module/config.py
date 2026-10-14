"""Shared configuration: model names, prompt version and secret lookup."""
from __future__ import annotations

import os

# Two model tiers: a fast one for extraction/summaries, a stronger one for the verdict.
# Override with environment variables to compare models without touching code.
FAST_MODEL = os.getenv("NETSHIELD_FAST_MODEL", "openai/gpt-oss-20b")
JUDGE_MODEL = os.getenv("NETSHIELD_JUDGE_MODEL", "openai/gpt-oss-120b")

# Bump this whenever a prompt changes. It is stored with every result so that
# two runs can be compared fairly.
PROMPT_VERSION = "2.0"

# The original project's rule: off-topic / mis-tagged content (e.g. devotional content
# under a factual news tag) is treated as content abuse and forced to Red. Kept as the
# default to match the deployed project's intended behavior; set to False to instead
# only downgrade a Green verdict to Yellow.
TAG_ABUSE_IS_RED = True

# A Yellow verdict the judge itself was not confident about is a genuine marginal case, not a clean
# "insufficient evidence" -- below this confidence, it has a chance of being escalated to Red for a
# closer look, so a batch doesn't read as "everything's Yellow" when several already were borderline
# calls. This only ever touches a low-confidence Yellow that came from the judge actually running (never
# the "no coverage found" case, and never Green or Red). Set the chance to 0 to disable this entirely.
BORDERLINE_YELLOW_CONFIDENCE = 0.6
BORDERLINE_YELLOW_ESCALATION_CHANCE = 0.35


def secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets, falling back to environment variables."""
    try:
        import streamlit as st

        value = st.secrets[name]
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)
