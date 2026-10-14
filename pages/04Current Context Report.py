import streamlit as st

from module import features, nextractor, pipeline, state, ui

ui.setup_page("Current Context Report", "📰")
ui.page_header("Actual Context Report", "Find the factual context for the videos you've fetched.")
ui.callout(
    "Contextual Analysis Overview",
    items=[
        "This page uses the video titles to search for related news articles from external web sources.",
        "Videos that yield matching news articles are displayed below, providing a factual baseline against "
        "which the video content can be compared.",
    ],
)

df = state.enriched()
if df is None:
    ui.empty_state("No video data yet", "Fetch some videos first, then look up their news context here.",
                   "pages/01Request Analysis.py", "Go to Request Analysis")

contexts = st.session_state.setdefault("context", {})
current_ids = tuple(df["Video ID"])
already_run = contexts.get("_for") == current_ids


def render_articles(records: list, video_key: str) -> None:
    if not records:
        st.info("No relevant news context found for this video after search attempts.")
        return
    st.markdown(f"**Total Articles Found:** {len(records)}")
    st.markdown("**Related News Articles (click to expand):**")
    for idx, art in enumerate(records, 1):
        with st.expander(f"Article {idx}: {art['headline']}"):
            st.markdown(f"* **Source:** {ui.md_escape(art['publisher'])}")
            st.markdown(f"* **Date Published:** {art['date'] or 'N/A'}")
            st.markdown(f"* **URL:** [View Full Article]({art['url']})")
            if art["similarity"] is not None:
                st.caption(f"Wording overlap with the video: {art['similarity']:.2f}")
            st.markdown("#### Article Content")
            # Keyed by video + position, not just the article URL: the same article can legitimately
            # turn up for more than one video (e.g. two videos on the same topic both matching the same
            # general news piece), and a key collision like that crashes the whole page.
            st.text_area(f"Content for {art['headline'][:80]}", art["excerpt"], height=300,
                        key=f"content_{video_key}_{idx}")


_, refresh_col = st.columns([3, 1])
rerun = refresh_col.button("Refresh all", help="Re-search news for every video below, discarding the cached results.")

st.subheader("Related Context Extraction Results")
if already_run and not rerun:
    st.caption("Showing cached results from this session. Select Refresh all to search again.")
st.markdown("---")

found_for = []
for k, row in enumerate(df.to_dict("records")):
    st.markdown(f"### Video {k + 1}: {ui.md_escape(row['Video Title'])}")

    if already_run and not rerun:
        records = contexts.get(row["Video ID"], {}).get("articles", [])
    else:
        with st.spinner(f"Processing video {k + 1}: translating and searching external context..."):
            query, articles = pipeline.find_context(row["Video Title"], row["Published At"])
            video_text = f"{row['Video Title']} {features.clean_description(row['Description'])}"
            sims = features.tfidf_scores(video_text, [a.content or a.description for a in articles])
            records = [pipeline.article_record(a, sims[i] if i < len(sims) else None) for i, a in enumerate(articles)]
        contexts[row["Video ID"]] = {"query": query, "articles": records}

    render_articles(records, video_key=row["Video ID"])
    if records:
        found_for.append(row["Video ID"])
    st.markdown("---")

contexts["_for"] = current_ids

if found_for:
    st.success(f"Contextual data extracted for {len(found_for)} of {len(df)} videos.")
else:
    st.info("No relevant news context found for any of the videos.")

with st.expander("Getting 'no articles found' for too many videos? Test the news search"):
    st.caption(
        "Runs one query against every news source this page uses -- a Google News scrape, the free "
        "GDELT index, and NewsAPI if a key is set -- and shows exactly what each one returned."
    )
    probe = st.text_input("Test query", "India air pollution")
    if st.button("Run diagnostic"):
        with st.spinner("Testing..."):
            trace = nextractor.diagnose(probe)
        st.json(trace)
        if trace.get("google_http_status") == 429 or trace.get("gdelt_error", "").find("429") != -1:
            ui.callout(
                text="HTTP 429 means that host is rate-limiting this app's IP right now. The app already paces its "
                     "own requests and backs off automatically on a 429 -- if you still see this, you're likely "
                     "testing right after a big batch run finished. Wait a minute and try again.",
                tone="warn",
            )
        if trace.get("newsapi_key_configured") and not trace.get("newsapi_ok"):
            ui.callout(
                text="NewsAPI configured but returning nothing: check the server logs for the exact status. A "
                     "common cause is HTTP 426 -- NewsAPI's free Developer plan is restricted to local testing by "
                     "NewsAPI itself and refuses requests from a deployed app; it needs a paid plan to work here.",
                tone="warn",
            )
        if not trace.get("newsapi_key_configured"):
            ui.callout(text="No NEWS_API_KEY is configured. The Google News scrape and GDELT still run without one.", tone="info")
