"""Shared look and feel for every NetShield page.

Design rules
- Colour carries meaning: saturated colour is reserved for verdicts (Green, Yellow, Red).
  Everything else stays in quiet blue-greys, so a verdict is always the first thing you see.
- One left edge: headings, text, tables and charts all start at the same x position.
  (The old pages added margin-left / margin to some blocks, which is why text looked indented.)
- Flat panels with a thin border. No hover zoom, no drop shadows.
- Panels use translucent fills, so they still read correctly if a viewer switches to the light theme.
"""
from __future__ import annotations

import html

import streamlit as st

STATUS_ORDER = ["Green", "Yellow", "Red"]
STATUS_COLORS = {"Green": "#2fbf71", "Yellow": "#e5a72e", "Red": "#e5484d"}
STATUS_EMOJI = {"Green": "🟢", "Yellow": "🟡", "Red": "🔴"}
STATUS_MEANING = {
    "Green": "The claim is supported by news coverage.",
    "Yellow": "There is not enough evidence to decide.",
    "Red": "The claim contradicts news coverage.",
}

_CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

:root {
  --ns-line: rgba(148,163,184,.28);
  --ns-tint: rgba(148,163,184,.07);
  --ns-mute: #8ea0b5;
  --ns-accent: #4f8ef7;
  --ns-green: #2fbf71;
  --ns-yellow: #e5a72e;
  --ns-red: #e5484d;
}

.stApp { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; }
.stApp h1, .stApp h2, .stApp h3 {
  font-family: 'Newsreader', Georgia, 'Times New Roman', serif;
  font-weight: 600; letter-spacing: -.01em;
}
.stApp h1 { font-size: 2.35rem; line-height: 1.15; padding: 0 0 .3rem; }
.stApp h2 { font-size: 1.6rem; }
.stApp h3 { font-size: 1.25rem; }

/* one consistent content column, no ad-hoc margins */
.block-container { max-width: 1180px; padding: 3.5rem 2.5rem 4rem !important; }

/* sidebar branding */
[data-testid="stSidebar"]::before {
  content: "NetShield 🛡️"; display: block;
  font: 600 1.6rem 'Newsreader', Georgia, serif;
  padding: 1.4rem 1.5rem .9rem; margin-bottom: .5rem;
  border-bottom: 1px solid var(--ns-line);
}
[data-testid="stSidebar"]::after {
  content: "Use the sidebar to move between pages.";
  position: absolute; left: 0; right: 0; bottom: .75rem;
  padding: .6rem 1rem; text-align: center; font-size: .8rem; color: var(--ns-mute);
}
.ns-side { border-top: 1px solid var(--ns-line); margin-top: 1rem; padding: .9rem .25rem 0; font-size: .9rem; }
.ns-side b { display: block; margin-bottom: .35rem; }
.ns-side div { display: flex; justify-content: space-between; padding: .1rem 0; color: var(--ns-mute); }
.ns-side span { color: inherit; font-variant-numeric: tabular-nums; }

/* buttons */
.stApp .stButton > button, .stApp .stDownloadButton > button, .stApp .stFormSubmitButton > button {
  border-radius: 6px; font-weight: 600;
}

/* page header */
.ns-lede { font-size: 1.08rem; line-height: 1.6; color: var(--ns-mute); max-width: 68ch; margin: .1rem 0 1.5rem; }

/* callouts */
.ns-callout {
  border-left: 3px solid var(--tone, var(--ns-accent)); background: var(--ns-tint);
  border-radius: 0 8px 8px 0; padding: .85rem 1.1rem; margin: 0 0 1.5rem;
}
.ns-callout .t { display: block; font-weight: 600; margin-bottom: .2rem; }
.ns-callout p { margin: .15rem 0; }
.ns-callout ul, .ns-callout ol { margin: .3rem 0 0; padding-left: 1.2rem; }
.ns-callout li { margin: .2rem 0; }
.ns-tone-info { --tone: var(--ns-accent); }
.ns-tone-warn { --tone: var(--ns-yellow); }
.ns-tone-ok   { --tone: var(--ns-green); }
.ns-tone-bad  { --tone: var(--ns-red); }

/* verdict pills and card headers */
.ns-pill {
  display: inline-flex; align-items: center; gap: .4rem; padding: .1rem .65rem;
  border: 1px solid var(--c); border-radius: 999px; color: var(--c);
  font-size: .85rem; font-weight: 600; white-space: nowrap;
}
.ns-pill i { width: .5rem; height: .5rem; border-radius: 50%; background: var(--c); }
.s-green { --c: var(--ns-green); } .s-yellow { --c: var(--ns-yellow); } .s-red { --c: var(--ns-red); }
.ns-vhead {
  display: flex; align-items: center; gap: .75rem; flex-wrap: wrap;
  border-left: 4px solid var(--c); padding: .1rem 0 .1rem .8rem; margin-bottom: .6rem;
}
.ns-vhead .n { color: var(--ns-mute); font-variant-numeric: tabular-nums; }
.ns-vhead .t { flex: 1 1 18rem; font-weight: 600; font-size: 1.05rem; }
.ns-vhead .t a { color: inherit; text-decoration: none; border-bottom: 1px solid var(--ns-line); }
.ns-vhead .c { color: var(--ns-mute); font-size: .9rem; }

/* home page */
.ns-hero { padding: .5rem 0 1.25rem; }
.stApp .ns-hero h1 {
  font-size: clamp(2.1rem, 4.2vw, 3.2rem); line-height: 1.1; letter-spacing: -.02em;
  max-width: 22ch; margin: 0 0 .9rem; padding: 0;
}
.ns-key { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1.25rem; margin: .25rem 0 2rem; }
.ns-key p { margin: .45rem 0 0; color: var(--ns-mute); font-size: .95rem; }
.ns-steps { list-style: none; padding: 0; margin: .25rem 0 2rem; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1.25rem; }
.ns-steps li {
  border-top: 2px solid var(--ns-line); padding: .8rem .8rem 1rem 0; margin: 0 -.8rem 0 0;
  display: flex; flex-direction: column; gap: .3rem; border-radius: 0 0 8px 8px;
  transition: border-color .18s ease, background-color .18s ease, transform .18s ease;
}
.ns-steps li:hover {
  border-top-color: var(--ns-accent); background: var(--ns-tint); transform: translateY(-3px);
  padding-left: .8rem; margin-right: 0;
}
.ns-steps .n { font-family: 'Newsreader', Georgia, serif; font-size: 1.7rem; line-height: 1; color: var(--ns-accent); transition: color .18s ease; }
.ns-steps b { font-size: 1.02rem; }
.ns-steps span.d { color: var(--ns-mute); font-size: .93rem; line-height: 1.5; }

.ns-mini-card {
  border: 1px solid var(--ns-line); background: var(--ns-tint); border-radius: 10px; padding: 1.1rem 1.2rem;
  height: 100%; box-sizing: border-box;
}
.ns-mini-card h4 { margin: 0 0 .5rem; font: 600 1rem 'IBM Plex Sans', sans-serif; }
.ns-mini-card p { margin: 0; color: var(--ns-mute); font-size: .92rem; line-height: 1.55; }

/* Equal-height cards: the row is wrapped in st.container(key="ns_cards"), which Streamlit marks
   with a .st-key-ns_cards class we can target. Without this, the card with the most text (and the
   other two) end up different heights, since st.columns() does not stretch columns by default. */
.st-key-ns_cards [data-testid="stHorizontalBlock"] { align-items: stretch; }
.st-key-ns_cards [data-testid="stColumn"],
.st-key-ns_cards [data-testid="stColumn"] > div,
.st-key-ns_cards [data-testid="stVerticalBlock"],
.st-key-ns_cards [data-testid="stVerticalBlockBorderWrapper"],
.st-key-ns_cards [data-testid="element-container"],
.st-key-ns_cards [data-testid="stMarkdown"],
.st-key-ns_cards [data-testid="stMarkdownContainer"] { height: 100%; }

@media (max-width: 900px) {
  .ns-steps { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ns-key { grid-template-columns: 1fr; }
  .block-container { padding: 3rem 1.25rem 3rem !important; }
}
@media (max-width: 560px) { .ns-steps { grid-template-columns: 1fr; } }
"""


def setup_page(title: str, icon: str = "🛡️") -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} | NetShield", page_icon=icon, layout="wide", initial_sidebar_state="expanded")
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)
    _sidebar_status()


def _sidebar_status() -> None:
    videos = st.session_state.get("videos_df")
    results = st.session_state.get("results_df")
    if videos is None and results is None:
        return
    rows = [("Videos loaded", len(videos) if videos is not None else 0)]
    if results is not None:
        rows.append(("Analysed", len(results)))
        rows.append(("Flagged red", int((results["Status"] == "Red").sum())))
    body = "".join(f"<div>{html.escape(k)}<span>{v}</span></div>" for k, v in rows)
    with st.sidebar:
        st.markdown(f'<div class="ns-side"><b>This session</b>{body}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- components
def page_header(title: str, lede: str = "") -> None:
    st.title(title)
    if lede:
        st.markdown(f'<p class="ns-lede">{html.escape(lede)}</p>', unsafe_allow_html=True)


def callout(title: str = "", items: list[str] | None = None, text: str = "", tone: str = "info", ordered: bool = False, footer: str = "") -> None:
    """A titled note. `items`/`text`/`footer` may contain simple inline HTML such as <b>; pass only trusted text.
    Render order: title, text, items, footer — footer is for a closing line that belongs after a list."""
    parts = []
    if title:
        parts.append(f'<span class="t">{html.escape(title)}</span>')
    if text:
        parts.append(f"<p>{text}</p>")
    if items:
        tag = "ol" if ordered else "ul"
        parts.append(f"<{tag}>" + "".join(f"<li>{i}</li>" for i in items) + f"</{tag}>")
    if footer:
        parts.append(f"<p><b>{footer}</b></p>")
    st.markdown(f'<div class="ns-callout ns-tone-{tone}">{"".join(parts)}</div>', unsafe_allow_html=True)


def pill(status: str) -> str:
    cls = {"Green": "s-green", "Yellow": "s-yellow", "Red": "s-red"}.get(status, "s-yellow")
    return f'<span class="ns-pill {cls}"><i></i>{html.escape(str(status))}</span>'


def verdict_header(index: int, title: str, status: str, confidence=None, url: str | None = None) -> str:
    cls = {"Green": "s-green", "Yellow": "s-yellow", "Red": "s-red"}.get(status, "s-yellow")
    name = html.escape(str(title))
    if url:
        name = f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener">{name}</a>'
    conf = ""
    if isinstance(confidence, (int, float)) and confidence == confidence:
        conf = f'<span class="c">confidence {confidence:.0%}</span>'
    return f'<div class="ns-vhead {cls}"><span class="n">{index}</span><span class="t">{name}</span>{pill(status)}{conf}</div>'


def md_escape(text) -> str:
    """Stop $ in titles from being read as LaTeX by st.markdown."""
    return str(text).replace("$", r"\$")


def kpis(items: list[tuple]) -> None:
    """Row of bordered metrics. Each item is (label, value) or (label, value, help)."""
    for col, item in zip(st.columns(len(items)), items):
        label, value = item[0], item[1]
        col.metric(label, value, help=item[2] if len(item) > 2 else None, border=True)


def empty_state(title: str, text: str, page: str, label: str) -> None:
    callout(title, text=text, tone="info")
    st.page_link(page, label=label)
    st.stop()


def style_fig(fig, height: int = 340):
    """Consistent Plotly styling: left-aligned title, legend above the plot, transparent background."""
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=70, b=8),
        title=dict(x=0, xanchor="left", y=0.97, yanchor="top", font=dict(size=15)),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", title=None),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.15)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,.15)", zeroline=False)
    return fig
