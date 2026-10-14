import streamlit as st

from module import config, llm, pipeline, state, ui

ui.setup_page("Automate", "🤖")
ui.page_header("NetShield Content Automation Analysis", "Runs the full AI check on every fetched video.")
ui.callout(
    "AI Verification Pipeline: This page performs the full AI analysis on the fetched videos, executing the core NetShield process: ",
    ordered=True,
    items=[
        "<b>Transcription:</b> Converts video audio to raw text.",
        "<b>Context Fetch:</b> Finds and scrapes related news articles.",
        f"<b>Summarization:</b> Uses LLMs to create summaries of both video and news.",
        f"<b>Validation:</b> Uses Meta/Llama 3 70B to compare summaries and assign a status.",
    ],
    footer="Final Output: A comprehensive report of Green (Accurate), Yellow (Inconclusive), or Red (Misinformation/Abuse).",
)

videos = state.get_videos()
if videos is None:
    ui.empty_state("No video data yet", "Fetch some videos first, then run the analysis.",
                   "pages/01Request Analysis.py", "Go to Request Analysis")


def _num(value):
    return None if value is None or value != value else value


def render_card(index: int, row: dict, res: dict) -> None:
    confidence, alignment = _num(res.get("Confidence")), _num(res.get("Alignment"))
    with st.container(border=True):
        st.markdown(ui.verdict_header(index, row["Video Title"], res["Status"], confidence, row["Video URL"]), unsafe_allow_html=True)
        with st.expander("View Full Content"):
            st.text(f"Video Title: {row['Video Title']}\nVideo Description: {row['Description']}")
        if res.get("Claim"):
            label = "**Video Claim**" + (" (from the spoken transcript):" if res.get("Used Transcript") else ":")
            st.markdown(label)
            st.info(ui.md_escape(res["Claim"]))
            if not res.get("Used Transcript"):
                st.caption("No transcript/captions available for this video -- claim extracted from the title and description only.")
        if res.get("Articles Found") is not None:
            st.markdown(f"**{res['Articles Found']} Related News Articles Found.** (Full articles on the **Current Context Report** page)")
        if res.get("News Summary"):
            st.markdown("**News Summary:**")
            st.success(ui.md_escape(res["News Summary"]))
        if res.get("Reasoning"):
            st.markdown(ui.md_escape(res["Reasoning"]))
        for item in res.get("Contradictions") or []:
            st.markdown(f"- {ui.md_escape(item)}")
        if alignment is not None:
            st.caption(f"Wording overlap with best-matching article: {alignment:.2f}")
        if res.get("Error"):
            st.warning(f"Something went wrong for this video: {res['Error']}")


current_ids = tuple(videos["Video ID"])
cached = state.get_results()
already_run = cached is not None and tuple(cached["Video ID"]) == current_ids

_, rerun_col = st.columns([3, 1])
rerun = rerun_col.button("Re-run analysis", help="Runs the pipeline again for every video, discarding cached results.")

if not already_run or rerun:
    try:
        llm.client()  # fail fast, and build the client once on the main thread
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    st.subheader("1. Data Loading")
    st.write("Loading fetched YouTube data for analysis...")
    rows = videos.to_dict("records")
    st.subheader(f"2. Processing {len(rows)} Videos")
    progress = st.progress(0.0)
    results = []
    for k, row in enumerate(rows):
        with st.spinner(f"Video {k + 1} of {len(rows)}: {row['Video Title'][:70]}"):
            res = pipeline.analyse_video(row)
        results.append(res)
        render_card(k + 1, row, res)
        progress.progress((k + 1) / len(rows))

    state.set_results(results)
    st.session_state["just_ran"] = True
    st.rerun()

if st.session_state.pop("just_ran", False):
    st.toast("Analysis complete")

results = state.get_results()
if results is not None:
    if already_run and not rerun:
        st.caption("Showing cached results from this session. Select Re-run analysis to process again.")
    counts = results["Status"].value_counts()
    st.subheader("3. Final Account Report")
    st.success(f"Analysis complete! {len(results)} videos processed.")
    ui.kpis([
        ("Green", int(counts.get("Green", 0))),
        ("Yellow", int(counts.get("Yellow", 0))),
        ("Red", int(counts.get("Red", 0))),
        ("Failed", int((results["Error"] != "").sum()), "Videos where a step raised an error. They are scored Yellow."),
    ])

    by_id = results.set_index("Video ID").to_dict("index")
    for i, row in enumerate(videos.to_dict("records"), 1):
        if row["Video ID"] in by_id:
            render_card(i, row, by_id[row["Video ID"]])

    st.write("Detail Report")
    detail = videos[["Video ID", "Video Title", "Video URL"]].merge(results[["Video ID", "Status"]], on="Video ID")
    st.dataframe(
        detail[["Video Title", "Video URL", "Status"]], hide_index=True, width="stretch",
        column_config={"Video URL": st.column_config.LinkColumn("Video Link", display_text="Watch")},
    )
    st.success("Report saved for this session. View the full summary in the **Account Report** tab.")
    st.page_link("pages/05Account Report.py", label="Open the account report")
    st.download_button("Download results as CSV", state.to_csv_bytes(results), "netshield_results.csv", "text/csv")
