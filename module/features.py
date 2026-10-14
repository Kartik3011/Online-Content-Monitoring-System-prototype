"""Feature engineering and analytics. Pure pandas/numpy only — deliberately no scikit-learn.

scikit-learn was dropped as a dependency after it repeatedly failed to install on Streamlit
Community Cloud (ModuleNotFoundError at runtime despite being listed in requirements.txt).
TF-IDF/cosine-similarity and the precision/recall/F1/confusion-matrix evaluation are reimplemented
below with plain numpy and pandas so this class of deployment failure can't happen again.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd

URL_RE = re.compile(r"https?://[^\s<>\"')]+", re.I)
HASHTAG_RE = re.compile(r"#[^\s#]+")
# Deliberately narrow: words like "breaking" are normal in news titles and would be noise.
SENSATIONAL_RE = re.compile(
    r"\b(?:shocking|exposed|secrets?|banned|leaked|miracle|cure|scam|"
    r"you won'?t believe|must watch|share this|before it'?s deleted|"
    r"100% (?:true|real|sure)|doctors hate|they don'?t want you to know)\b",
    re.I,
)
NUMERIC_COLUMNS = ["Views", "Likes", "Comments", "Subscriber Count", "Total Views", "Video Count", "Duration (s)"]
STATUS_SCORE = {"Green": 0.0, "Yellow": 1.0, "Red": 3.0}


# --------------------------------------------------------------------------- text helpers
def clean_description(text) -> str:
    """Remove links and hashtags and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n", text).strip()


def extract_hashtags(text) -> List[str]:
    return HASHTAG_RE.findall(text) if isinstance(text, str) else []


def link_domains(text) -> List[str]:
    if not isinstance(text, str):
        return []
    domains = {urlparse(u).netloc.lower().removeprefix("www.") for u in URL_RE.findall(text)}
    return sorted(d for d in domains if d)


def find_phrases(text) -> List[str]:
    return sorted({m.lower() for m in SENSATIONAL_RE.findall(text)}) if isinstance(text, str) else []


def _caps_ratio(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    return sum(c.isupper() for c in letters) / len(letters) if letters else 0.0


# --------------------------------------------------------------------------- statistics
def robust_z(values: pd.Series) -> pd.Series:
    """Median/MAD z-score. Stable on tiny samples where the mean and std are dragged by one outlier."""
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = x.median()
    mad = (x - median).abs().median()
    if pd.notna(mad) and mad > 0:
        return 0.6745 * (x - median) / mad
    std = x.std(ddof=0)
    if pd.notna(std) and std > 0:
        return (x - x.mean()) / std
    return pd.Series(0.0, index=x.index).where(x.notna())


def add_features(df: pd.DataFrame, now: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Add engagement, reach and text-signal columns to the raw video table."""
    out = df.copy()
    for col in NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce") if col in out else np.nan
    for col in ("Video Title", "Description"):
        if col not in out:
            out[col] = ""
    out["Published At"] = pd.to_datetime(out["Published At"], utc=True, errors="coerce")
    now = now if now is not None else pd.Timestamp.now(tz="UTC")

    age = ((now - out["Published At"]).dt.total_seconds() / 86400).clip(lower=1)
    views = out["Views"].where(out["Views"] > 0)
    subs = out["Subscriber Count"].where(out["Subscriber Count"] > 0)
    out["Age (days)"] = age
    out["Engagement Rate"] = (out["Likes"].fillna(0) + out["Comments"].fillna(0)) / views
    out["Views per Day"] = out["Views"] / age
    out["Reach Ratio"] = out["Views"] / subs  # views relative to audience size

    title = out["Video Title"].fillna("").astype(str)
    desc = out["Description"].fillna("").astype(str)
    both = title + " " + desc
    out["Description Words"] = desc.str.split().str.len()
    out["Hashtag Count"] = desc.map(lambda t: len(HASHTAG_RE.findall(t)))
    out["Link Count"] = desc.map(lambda t: len(URL_RE.findall(t)))
    out["Exclamations"] = both.str.count("!")
    out["Caps Ratio"] = title.map(_caps_ratio)
    out["Sensational Phrases"] = both.map(lambda t: len(find_phrases(t)))
    out["Reach Z"] = robust_z(np.log10(out["Reach Ratio"]))
    return out


# --------------------------------------------------------------------------- text alignment baseline
# A compact English stopword list, not an exhaustive one -- enough to keep function words from
# drowning out the topical words that actually distinguish one article from another.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her
here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's
its itself let's me more most mustn't my myself no nor not of off on once only or other ought our
ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than
that that's the their theirs them themselves then there there's these they they'd they'll they're
they've this those through to too under until up very was wasn't we we'd we'll we're we've were
weren't what what's when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself yourselves
""".split())
_TOKEN_RE = re.compile(r"[A-Za-z]{2,}")


def _tokenize(text: str) -> List[str]:
    """Lowercase word tokens with stopwords removed, plus adjacent-word bigrams, mirroring a
    unigram+bigram TF-IDF vectorizer closely enough for this use (a sanity-check signal, not a
    precision text-similarity model)."""
    words = [w for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS]
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _tfidf_vectors(docs: List[str]) -> List[dict]:
    """TF-IDF weights per document, as sparse dicts of {term: weight}. Matches scikit-learn's
    TfidfVectorizer defaults closely: sublinear tf (1 + log(tf)), smooth idf, L2-normalized."""
    token_lists = [_tokenize(d) for d in docs]
    n_docs = len(docs)
    doc_freq = Counter()
    for tokens in token_lists:
        doc_freq.update(set(tokens))

    idf = {term: math.log((1 + n_docs) / (1 + df)) + 1.0 for term, df in doc_freq.items()}

    vectors = []
    for tokens in token_lists:
        tf = Counter(tokens)
        weights = {term: (1.0 + math.log(count)) * idf[term] for term, count in tf.items()}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        vectors.append({term: w / norm for term, w in weights.items()})
    return vectors


def _cosine(a: dict, b: dict) -> float:
    shared = a.keys() & b.keys()
    return sum(a[t] * b[t] for t in shared)  # vectors are already L2-normalized


def tfidf_scores(video_text: str, news_texts: Sequence[str]) -> List[float]:
    """Cosine similarity between the video text and each article (0 = no shared wording, 1 = identical).

    A lightweight, dependency-free sanity check on top of the LLM judge -- not a substitute for it.
    """
    docs = [t for t in news_texts if t and t.strip()]
    if not video_text or not video_text.strip() or not docs:
        return []
    vectors = _tfidf_vectors([video_text] + docs)
    video_vec, article_vecs = vectors[0], vectors[1:]
    if not video_vec:  # video text had no scorable tokens (e.g. all stopwords)
        return [0.0] * len(article_vecs)
    return [_cosine(video_vec, v) for v in article_vecs]


def tfidf_alignment(video_text: str, news_texts: Sequence[str]) -> Optional[float]:
    scores = tfidf_scores(video_text, news_texts)
    return max(scores) if scores else None


# --------------------------------------------------------------------------- review triage
def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Flag disagreements between the LLM verdict and the text baseline, and rank a review queue."""
    out = df.copy()
    if "Status" not in out:
        return out

    align = pd.to_numeric(out.get("Alignment"), errors="coerce")
    conf = pd.to_numeric(out.get("Confidence"), errors="coerce")
    low, high = align.quantile(0.25), align.quantile(0.75)
    green, red = out["Status"].eq("Green"), out["Status"].eq("Red")

    out["Disagreement"] = (green & (align <= low)) | (red & (align >= high))
    score = out["Status"].map(STATUS_SCORE).fillna(0.0)
    score += (green & (conf < 0.6)).astype(float)
    score += out["Disagreement"].astype(float)
    score += (out["Reach Z"].abs() > 3).fillna(False).astype(float)
    score += (out["Sensational Phrases"] >= 2).astype(float)
    out["Review Priority"] = score
    return out


# --------------------------------------------------------------------------- evaluation
def classification_summary(y_true: Sequence[str], y_pred: Sequence[str], labels=("Green", "Yellow", "Red")):
    """Per-class precision/recall/F1 and a confusion matrix (rows = human, columns = AI).

    A plain pandas/numpy reimplementation of sklearn.metrics.classification_report +
    confusion_matrix (see the module docstring for why scikit-learn was dropped).
    """
    labels = list(labels)
    y_true = pd.Series(list(y_true))
    y_pred = pd.Series(list(y_pred))

    matrix = (
        pd.crosstab(y_true, y_pred)
        .reindex(index=labels, columns=labels, fill_value=0)
        .rename(index=lambda l: f"Human {l}", columns=lambda l: f"AI {l}")
    )

    counts = matrix.to_numpy()
    rows = []
    total_support = 0
    for i, label in enumerate(labels):
        tp = counts[i, i]
        support = counts[i, :].sum()
        predicted = counts[:, i].sum()
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"precision": precision, "recall": recall, "f1-score": f1, "support": support})
        total_support += support

    report = pd.DataFrame(rows, index=labels)
    accuracy = (np.diag(counts).sum() / total_support) if total_support else 0.0
    macro = report[["precision", "recall", "f1-score"]].mean()
    weighted = (
        (report[["precision", "recall", "f1-score"]].T * report["support"]).T.sum() / total_support
        if total_support else pd.Series({"precision": 0.0, "recall": 0.0, "f1-score": 0.0})
    )
    report.loc["accuracy"] = {"precision": accuracy, "recall": accuracy, "f1-score": accuracy, "support": total_support}
    report.loc["macro avg"] = {**macro.to_dict(), "support": total_support}
    report.loc["weighted avg"] = {**weighted.to_dict(), "support": total_support}

    return report, matrix
