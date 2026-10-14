import streamlit as st

from module import config, llm, state, ui

ui.setup_page("Home", "🛡️")

# ------------------------------------------------------------------ hero
st.markdown(
    '<div class="ns-hero"><h1>Check what a video claims against what the news reports</h1>'
    '<p class="ns-lede">NetShield collects YouTube videos on a topic, finds news coverage from the same period, '
    "and asks a language model whether the video's main claim holds up. Every verdict comes with a confidence "
    "score and written reasoning, so a reviewer can agree or overrule it.</p></div>",
    unsafe_allow_html=True,
)
if st.button("Start an analysis", type="primary"):
    st.switch_page("pages/01Request Analysis.py")

# ------------------------------------------------------------------ why this matters (original wording, kept verbatim)
CARDS = [
    ("Fake News: A Growing Threat",
     "Fake news videos and blogs have become a major issue, leading to the spread of misinformation, social unrest, "
     "and a general erosion of trust in media sources. The misleading content has caused "
     "confusion and harm to society, and it is crucial to have reliable tools to identify these threats."),
    ("Impact of Fake News",
     "The consequences of fake news are far-reaching, affecting public opinion, political stability, and even public "
     "health. From misleading information about elections to harmful medical advice, fake news is a serious concern "
     "that needs to be addressed immediately."),
    ("How NetShield Helps",
     "NetShield aims to identify fake news by analyzing video content and blogs. Through the power of artificial "
     "intelligence and machine learning, we provide tools for detecting fraudulent content, empowering users to make "
     "informed decisions and contribute to a more informed society."),
]
with st.container(key="ns_cards"):
    cols = st.columns(3)
    for col, (title, body) in zip(cols, CARDS):
        with col:
            st.markdown(f'<div class="ns-mini-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ verdict key
st.subheader("What a verdict means")
key = "".join(f"<div>{ui.pill(s)}<p>{ui.STATUS_MEANING[s]}</p></div>" for s in ui.STATUS_ORDER)
st.markdown(f'<div class="ns-key">{key}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ pipeline
st.subheader("How it works")
steps = [
    ("Collect", "Pull videos for a topic, city and date range from YouTube, with view counts and channel statistics."),
    ("Extract the claim", "A language model reads the title and description and picks out the one checkable claim."),
    ("Find coverage", "NetShield searches news from the days around the upload and measures how closely each article matches."),
    ("Judge and explain", "A stronger model compares claim and coverage, then returns a verdict, a confidence score and its reasoning."),
]
items = "".join(f'<li><span class="n">{i}</span><b>{t}</b><span class="d">{d}</span></li>' for i, (t, d) in enumerate(steps, 1))
st.markdown(f'<ol class="ns-steps">{items}</ol>', unsafe_allow_html=True)

# ------------------------------------------------------------------ session snapshot
st.subheader("Your current session")
videos, results = state.get_videos(), state.get_results()
if videos is None:
    ui.callout(text="No data yet. Start an analysis to fetch videos; results appear here and in the sidebar.", tone="info")
else:
    tiles = [("Videos loaded", len(videos))]
    if results is not None:
        tiles += [
            ("Analysed", len(results)),
            ("Flagged red", int((results["Status"] == "Red").sum())),
            ("Average confidence", f"{results['Confidence'].mean():.0%}" if results["Confidence"].notna().any() else "n/a"),
        ]
    ui.kpis(tiles)

ui.callout(
    "A verdict is a triage signal, not a ruling",
    text=(
        "Green, Yellow and Red come from a language model comparing text, and it can be wrong. "
        "Missing news coverage is scored Yellow, never Red. Review every Red before acting on an account."
    ),
    tone="warn",
)

# ------------------------------------------------------------------ assistant
st.markdown(
    """<style>
    [data-testid="stPopover"] { position: fixed !important; bottom: 1.25rem; right: 2rem; left: auto !important; width: auto !important; z-index: 1001; }

    /* floating trigger: a pill-shaped button instead of a plain Streamlit one */
    [data-testid="stPopover"] > button {
      border-radius: 999px !important; padding: .7rem 1.4rem !important; font-weight: 600 !important;
      border: none !important; color: #fff !important;
      background: linear-gradient(135deg, var(--ns-accent), #3f6fd1) !important;
      box-shadow: 0 8px 22px rgba(79,142,247,.4); transition: transform .15s ease, box-shadow .15s ease;
    }
    [data-testid="stPopover"] > button:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(79,142,247,.5); }
    [data-testid="stPopover"] > button p { color: #fff !important; }
    [data-testid="stPopover"][aria-disabled="true"] > button,
    [data-testid="stPopover"] > button:disabled {
      background: var(--ns-tint) !important; box-shadow: none; color: var(--ns-mute) !important;
    }

    /* the panel that opens */
    div[data-testid="stPopoverBody"] {
      width: 380px !important; max-width: 90vw !important; padding: 0 !important;
      border-radius: 18px !important; border: 1px solid var(--ns-line) !important;
      box-shadow: 0 20px 55px rgba(0,0,0,.38) !important; overflow: hidden;
    }
    .st-key-ns_chat_header {
      display: flex; align-items: center; justify-content: space-between; gap: .5rem;
      padding: .85rem 1.1rem; border-bottom: 1px solid var(--ns-line); margin-bottom: .3rem;
      background: linear-gradient(135deg, rgba(79,142,247,.16), rgba(79,142,247,.02));
    }
    .ns-chat-title { font: 600 1.05rem 'Newsreader', Georgia, serif; display: block; }
    .ns-chat-sub { color: var(--ns-mute); font-size: .8rem; }
    div[data-testid="stPopoverBody"] [data-testid="stVerticalBlock"] { gap: .4rem; }

    /* chat bubbles -- the whole reply is one bubble, even when it has several paragraphs */
    div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] { padding: .15rem .9rem !important; gap: .5rem; }
    div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
      background: var(--ns-tint); border: 1px solid var(--ns-line); border-radius: 14px;
      padding: .55rem .85rem; margin: .15rem 0; line-height: 1.5; font-size: .9rem;
    }
    div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { margin: 0 0 .5rem; }
    div[data-testid="stPopoverBody"] [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child { margin-bottom: 0; }

    /* quick-start suggestion chips */
    .ns-chat-suggest { padding: 0 1.1rem; }
    .ns-chat-suggest [data-testid="stButton"] > button {
      border-radius: 999px !important; font-size: .8rem !important; padding: .3rem .8rem !important;
      background: var(--ns-tint) !important; border: 1px solid var(--ns-line) !important; color: inherit !important;
      font-weight: 500 !important;
    }
    .ns-chat-suggest [data-testid="stButton"] > button:hover { border-color: var(--ns-accent) !important; color: var(--ns-accent) !important; }

    /* clear-chat ghost button in the header */
    .st-key-ns_chat_header [data-testid="stButton"] > button {
      background: transparent !important; border: none !important; color: var(--ns-mute) !important;
      padding: .1rem .3rem !important; font-size: .78rem !important; font-weight: 500 !important;
    }
    .st-key-ns_chat_header [data-testid="stButton"] > button:hover { color: var(--ns-red) !important; }

    div[data-testid="stPopoverBody"] [data-testid="stChatInput"] { padding: .6rem .9rem .9rem; }
    </style>""",
    unsafe_allow_html=True,
)

SUGGESTED_QUESTIONS = [
    "How many pages does NetShield have?",
    "What does a Yellow verdict mean?",
    "How does the Automate pipeline work?",
]

KNOWLEDGE_BASE = """
WHAT NETSHIELD IS
NetShield is a misinformation-checking tool for YouTube. You give it a topic, and it pulls matching
videos, finds the news coverage published around the same time, and asks a language model whether
each video's main claim actually holds up against that coverage. Every verdict comes with a
confidence score and written reasoning, so a human reviewer can agree with it or overrule it — it's
built as a triage aid for a reviewer, not an automatic judge. Everything runs inside your own
browser session (Streamlit's session state); nothing you fetch or analyse is written to a shared
database or visible to other visitors, and it resets when the session ends.

PAGE COUNT
NetShield has 8 screens in total: the Home page (this one) plus 7 pages reachable from the sidebar,
in the order they normally get used:
1. Home — this page: the pitch, the verdict key, a snapshot of your current session, and this chat.
2. Request Analysis — where you start: fetch YouTube videos for a topic.
3. Social Content Report — the raw fetched data as a sortable table with KPIs and charts.
4. Content Forensic — text-pattern signals per video, plus an optional on-demand AI summary.
5. Current Context Report — the news articles found for each video, with full text and a wording-overlap score.
6. Automate — runs the full AI verification pipeline and produces Green/Yellow/Red verdicts.
7. Account Report — the final summary: status counts, the full results table, and a CSV download.
8. Analytics Lab — channel breakdown, a human-review queue, and deeper data-science tooling (anomaly
   detection, a text-similarity baseline, precision/recall/F1 evaluation against your own review).
If asked how many pages NetShield has, the answer is 8 (counting Home), or 7 if someone means only
the pages listed in the sidebar below Home — it's fine to explain it either way.

HOW A VERDICT IS REACHED
For each video, the pipeline runs: (1) a fast model reads the title and description and extracts the
one checkable factual claim; (2) NetShield searches for news coverage from the days around the video's
upload, trying several sources and query phrasings so a single missed search doesn't leave a video
with no context; (3) the found articles are summarized; (4) a stronger judge model compares the claim
against the news summary and returns a verdict, a confidence score, written reasoning, and any
contradictions it found.

VERDICT MEANINGS
- Green: the claim is supported by the news coverage found.
- Yellow: there isn't enough evidence to decide either way. This is also what a video gets when no
  matching news coverage could be found at all — missing coverage is never treated as proof the claim
  is false, so it's scored Yellow rather than Red.
- Red: either the claim contradicts the news coverage, or the video's content doesn't match what its
  title/tags claim it's about (for example devotional or unrelated content posted under a factual news
  tag) — that second case is treated as content abuse/mistagging and is always forced Red.
Every verdict is a triage signal meant to speed up a human reviewer, not a final ruling — a person
should look at every Red before acting on it.

PAGE-BY-PAGE DETAIL
- Request Analysis (the starting page): enter a keyword or hashtag, an optional city to geofilter by,
  a date range, and the maximum number of videos to fetch. It pulls matching videos from the YouTube
  Data API along with each channel's subscriber count and stats. 'Fetch Data & Go to Content Report'
  loads the videos and takes you to see them; 'Complete Analysis & Switch to Automate' loads them and
  jumps straight into running the AI pipeline.
- Social Content Report: the videos you fetched, as a filterable table, with KPIs like total views,
  engagement rate and views per day, plus charts. Nothing here calls the language model — it's all
  computed directly from the fetched data.
- Content Forensic: per-video text signals computed from the title and description — hashtag count,
  linked domains, capital-letter ratio, exclamation marks, and a list of sensational phrasing (words
  like "shocking" or "exposed"). It also offers an optional AI summary of the cleaned description and
  the channel's bio, which only runs when you press the button for that specific video.
- Current Context Report: automatically searches for news coverage of every fetched video. It
  combines several independent sources (a Google News search, the free GDELT news index, and NewsAPI
  when a key is configured) and merges what each one finds, rather than relying on just one source —
  that's what keeps the article hit rate reasonable even when one source comes back thin. Each found
  article is shown with its source, publish date, full text and a score for how closely its wording
  matches the video. There's also a "Test the news search" tool for diagnosing why a particular query
  isn't finding coverage.
- Automate: runs the full verification pipeline (claim extraction → news search → summarize → judge)
  for every fetched video, one at a time, and shows a card per video with its verdict as it completes.
  Results are cached for the session; use Re-run analysis to process everything again from scratch.
- Account Report: the final rollup once Automate has run — counts of Green/Yellow/Red, a chart, the
  full results table, and the list of Red-flagged videos with a CSV download.
- Analytics Lab: the most data-science-heavy page. A channel-level breakdown, a review queue where you
  record your own verdict per video, feature distributions, anomaly detection for videos that went
  unusually viral for a small channel's subscriber count, a TF-IDF text-similarity baseline computed
  independently of the AI judge (a sanity check on whether the verdict and the wording overlap agree),
  and classification metrics (precision, recall, F1, a confusion matrix) comparing the AI's verdicts
  against the verdicts you record by hand.

HOW IT'S BUILT (if asked about the tech stack)
Streamlit for the app itself; the YouTube Data API v3 for fetching videos and channel stats; Groq's
API for the two language models (a fast model for claim extraction and summarizing, a larger model as
the judge); a news-retrieval layer that combines a Google News scrape, the GDELT Doc API and NewsAPI;
and pandas/numpy/Plotly for the data-science tooling. The TF-IDF similarity scoring and the evaluation
metrics (precision/recall/F1/confusion matrix) are implemented directly in pandas and numpy rather
than with a machine-learning library, which also keeps the app lighter to deploy.

LIMITATIONS
Only the video's title and description are analysed — not the audio or the video itself. YouTube's
location filter only returns videos that were geotagged by their uploader, so a city search can miss
relevant videos that weren't tagged. News coverage can be thin or missing for very local or very
recent events, in which case the video is scored Yellow rather than guessed at.
"""

SYSTEM_PROMPT = f"""You are the help assistant for a website called NetShield, built into its Home page.
Answer using the information below about NetShield itself. You can reason over it freely — count
items it lists, summarize across sections, compare pages, or explain how parts of the pipeline relate
— you are not limited to repeating sentences verbatim. If something genuinely isn't covered by this
information and isn't about NetShield at all (e.g. general knowledge, something unrelated), say so
briefly and steer the person back to what you can help with. Be friendly, concise, and specific.
---
{KNOWLEDGE_BASE}
---"""

st.session_state.setdefault("chat_messages", [])


def _ask(question: str) -> None:
    st.session_state.chat_messages.append({"role": "user", "content": question})
    try:
        reply = llm.chat_reply(SYSTEM_PROMPT, st.session_state.chat_messages)
    except Exception as exc:  # noqa: BLE001
        reply = f"The assistant could not answer just now ({type(exc).__name__}). Try again in a moment."
    st.session_state.chat_messages.append({"role": "assistant", "content": reply})


if config.secret("GROQ_API_KEY"):
    with st.popover("💬 Ask NetShield"):
        with st.container(key="ns_chat_header"):
            head_l, head_r = st.columns([3, 1])
            with head_l:
                st.markdown('<span class="ns-chat-title">🛡️ NetShield Assistant</span>'
                            '<span class="ns-chat-sub">Ask about any page, verdict or term</span>', unsafe_allow_html=True)
            with head_r:
                if st.session_state.chat_messages and st.button("Clear ✕", key="clear_chat"):
                    st.session_state.chat_messages = []
                    st.rerun()

        box = st.container(height=340)
        with box:
            if not st.session_state.chat_messages:
                st.markdown(
                    '<p style="color:var(--ns-mute); font-size:.88rem; padding:.9rem 1.1rem 0;">'
                    "👋 Hi! I know every page of NetShield inside out — ask me anything, or try one of these:</p>",
                    unsafe_allow_html=True,
                )
            for message in st.session_state.chat_messages:
                with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🛡️"):
                    st.markdown(message["content"])

        if not st.session_state.chat_messages:
            st.markdown('<div class="ns-chat-suggest">', unsafe_allow_html=True)
            for i, question in enumerate(SUGGESTED_QUESTIONS):
                if st.button(question, key=f"suggest_{i}", use_container_width=True):
                    _ask(question)
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        if prompt := st.chat_input("Ask about a page or a term"):
            _ask(prompt)
            st.rerun()
else:
    st.popover("💬 Assistant unavailable", disabled=True, help="Add GROQ_API_KEY to the app secrets to enable it.")
