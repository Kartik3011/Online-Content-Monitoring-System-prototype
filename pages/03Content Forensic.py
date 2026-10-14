import pandas as pd
import streamlit as st

from module import features, llm, state, ui

ui.setup_page("Content Forensic", "🧪")
ui.page_header(
    "Content Forensic",
    "Text signals hidden in each video's title, description and channel bio. Signals are hints for a reviewer, not verdicts.",
)
ui.callout(
    "Forensic Analysis Overview",
    items=[
        "This page extracts the video Description and cleans it by removing hashtags and external links.",
        "The cleaned description and the Channel Description are combined, and the result can be summarized "
        "on demand to provide a concise context for manual review.",
        "The summarized description is displayed below and can be copied for your notes.",
    ],
)

df = state.enriched()
if df is None:
    ui.empty_state("No video data yet", "Fetch some videos first, then inspect their text here.",
                   "pages/01Request Analysis.py", "Go to Request Analysis")

# ------------------------------------------------------------------ overview table
st.subheader("Signals across all videos")
signal_cols = ["Video Title", "Channel Title", "Description Words", "Hashtag Count", "Link Count",
               "Exclamations", "Caps Ratio", "Sensational Phrases", "Duration (s)"]
st.dataframe(
    df[signal_cols].sort_values("Sensational Phrases", ascending=False),
    hide_index=True,
    width="stretch",
    column_config={
        "Caps Ratio": st.column_config.NumberColumn("Capitals in title", format="percent"),
        "Sensational Phrases": st.column_config.NumberColumn("Sensational phrases", help="Phrases such as 'shocking', 'exposed' or 'must watch'."),
        "Duration (s)": st.column_config.NumberColumn("Length (s)"),
    },
)

# ------------------------------------------------------------------ single video
st.subheader("Inspect one video")
ids = df["Video ID"].tolist()
labels = {vid: f"{i}. {title[:90]}" for i, (vid, title) in enumerate(zip(ids, df["Video Title"]), 1)}
chosen = st.selectbox("Video", ids, format_func=labels.get)
row = df[df["Video ID"] == chosen].iloc[0]

st.markdown(f"**{ui.md_escape(row['Video Title'])}**")
published = row["Published At"].strftime("%Y-%m-%d") if pd.notna(row["Published At"]) else "unknown date"
st.caption(f"{row['Channel Title']}  |  published {published}  |  [watch on YouTube]({row['Video URL']})")

left, right = st.columns([2, 3])
with left:
    hashtags = features.extract_hashtags(row["Description"])
    domains = features.link_domains(row["Description"])
    phrases = features.find_phrases(f"{row['Video Title']} {row['Description']}")
    st.markdown("**Hashtags**")
    st.write(" ".join(f"`{h}`" for h in hashtags) if hashtags else "None")
    st.markdown("**Websites linked in the description**")
    st.write(", ".join(f"`{d}`" for d in domains) if domains else "None")
    st.markdown("**Sensational phrases**")
    st.write(", ".join(f"`{p}`" for p in phrases) if phrases else "None")
    st.markdown("**Tags set by the uploader**")
    st.write(row["Tags"] if row["Tags"] else "None")

with right:
    cleaned = features.clean_description(row["Description"]) or "The description contains only links or hashtags."
    tab_text, tab_bio, tab_ai = st.tabs(["Cleaned description", "Channel bio", "AI summary"])
    with tab_text:
        st.caption("Links and hashtags removed.")
        st.text(cleaned)
    with tab_bio:
        st.text(row["Channel Description"] or "The channel has no bio.")
    with tab_ai:
        source = f"Video description: {cleaned}\n\nChannel bio: {row['Channel Description']}"
        if len(source.split()) < 20:
            st.info("There is too little text to summarize.")
        elif st.button("Summarize with AI", key=f"sum_{chosen}"):
            with st.spinner("Summarizing..."):
                try:
                    st.write(llm.summarize(source))
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Summarization failed: {exc}")
        else:
            st.caption("Hit the button above whenever you want a summary — it won't generate one on its own.")
