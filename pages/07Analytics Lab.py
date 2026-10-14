import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from module import features, state, ui

ui.setup_page("Analytics Lab", "📈")
ui.page_header(
    "Analytics Lab",
    "Engineered features, anomaly detection, an unsupervised text baseline, and evaluation against human review.",
)

df = state.enriched()
if df is None:
    ui.empty_state("No video data yet", "Fetch some videos first, then explore the features here.",
                   "pages/01Request Analysis.py", "Go to Request Analysis")

tab_review, tab_features, tab_anomaly, tab_baseline, tab_eval = st.tabs(
    ["Channels & review queue", "Feature distributions", "Anomaly detection", "Text-similarity baseline", "Evaluation vs. human review"]
)

# ------------------------------------------------------------------ channels & review queue
with tab_review:
    st.markdown("**Channels**")
    channels = (
        df.groupby(["Channel Title", "Channel URL"], as_index=False)
        .agg(Videos=("Video ID", "count"),
             Red=("Status", lambda s: int((s == "Red").sum()) if "Status" in df else 0),
             Yellow=("Status", lambda s: int((s == "Yellow").sum()) if "Status" in df else 0),
             Green=("Status", lambda s: int((s == "Green").sum()) if "Status" in df else 0),
             Views=("Views", "sum"),
             Subscribers=("Subscriber Count", "max"))
    )
    st.dataframe(
        channels.sort_values(["Red", "Videos"], ascending=False), hide_index=True, width="stretch",
        column_config={
            "Channel URL": st.column_config.LinkColumn("Channel", display_text="Open"),
            "Views": st.column_config.NumberColumn(format="localized"),
            "Subscribers": st.column_config.NumberColumn(format="localized"),
        },
    )

    st.markdown("**Review queue**")
    st.caption(
        "Sorted by review priority when an analysis has run: red verdicts, low-confidence greens, disagreements "
        "with the text baseline, unusual reach and sensational wording all push a video up. Record your own "
        "verdict here to build the evaluation set used in the tab to the right."
    )
    if "Status" not in df:
        st.info("Run the AI analysis on the Automate page to populate the review queue.")
    else:
        queue = df.sort_values("Review Priority", ascending=False)
        columns = ["Human Verdict", "Status", "Confidence", "Review Priority", "Video Title", "Channel Title", "Reasoning", "Video URL"]
        grid = queue.set_index("Video ID")[columns]
        with st.form("review_form"):
            edited = st.data_editor(
                grid, width="stretch", disabled=[c for c in columns if c != "Human Verdict"],
                column_config={
                    "Human Verdict": st.column_config.SelectboxColumn("Your verdict", options=state.VERDICT_OPTIONS, required=False),
                    "Status": st.column_config.TextColumn("AI verdict"),
                    "Confidence": st.column_config.NumberColumn(format="percent"),
                    "Review Priority": st.column_config.NumberColumn("Priority", format="%.1f"),
                    "Video URL": st.column_config.LinkColumn("Link", display_text="Watch"),
                },
                key="review_editor",
            )
            if st.form_submit_button("Save reviews", type="primary"):
                state.save_human_verdicts(edited["Human Verdict"].to_dict())
                st.success("Reviews saved for this session.")

# ------------------------------------------------------------------ features
with tab_features:
    st.caption("Every signal here is plain math on the raw fetch (see `module/features.py`) — nothing here calls the LLM.")
    numeric = df.select_dtypes("number").columns.tolist()
    default = [c for c in ("Views", "Engagement Rate", "Reach Ratio", "Views per Day") if c in numeric]
    metric = st.selectbox("Metric", numeric, index=numeric.index(default[0]) if default else 0)
    color = "Status" if "Status" in df else None

    values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    use_log = values.gt(0).all() and values.max() / max(values.min(), 1e-9) > 100
    plot_df = df.copy()
    if use_log:
        plot_df[metric] = plot_df[metric].where(plot_df[metric] > 0)

    fig = px.histogram(plot_df, x=metric, color=color, log_x=use_log, nbins=25,
                       color_discrete_map=ui.STATUS_COLORS if color else None, title=f"Distribution of {metric}")
    st.plotly_chart(ui.style_fig(fig))
    if use_log:
        st.caption("Shown on a log scale: this metric spans more than two orders of magnitude.")

    st.markdown("**Correlations between numeric signals**")
    corr_cols = [c for c in numeric if df[c].notna().sum() >= 3]
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        heat = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, aspect="auto")
        st.plotly_chart(ui.style_fig(heat, 420))
    else:
        st.info("Not enough numeric variety yet to compute correlations.")

# ------------------------------------------------------------------ anomaly detection
with tab_anomaly:
    st.markdown("**Unusually viral videos for a small channel**")
    st.caption(
        "Reach Ratio = views ÷ subscribers. Reach Z is a robust (median/MAD) z-score of log(Reach Ratio) — "
        "it flags videos that reached far more people than their channel's size would suggest, without being "
        "thrown off by a single extreme outlier the way a mean/std z-score would be."
    )
    threshold = st.slider("Flag videos with |Reach Z| above", 1.0, 5.0, 3.0, 0.5)
    flagged = df[df["Reach Z"].abs() >= threshold].sort_values("Reach Z", key=abs, ascending=False)
    st.metric("Flagged videos", len(flagged))
    if flagged.empty:
        st.info("No videos cross this threshold. Try a lower value.")
    else:
        cols = ["Video Title", "Channel Title", "Views", "Subscriber Count", "Reach Ratio", "Reach Z"]
        if "Status" in flagged:
            cols.append("Status")
        st.dataframe(flagged[cols], hide_index=True, width="stretch",
                    column_config={"Reach Z": st.column_config.NumberColumn(format="%.2f"),
                                   "Reach Ratio": st.column_config.NumberColumn(format="%.1f")})

    fig = px.scatter(df, x="Subscriber Count", y="Views", log_x=True, log_y=True, hover_name="Video Title",
                     color=df["Reach Z"].abs() >= threshold, color_discrete_map={True: "#e5484d", False: "#8ea0b5"},
                     title="Views vs. subscribers (red = flagged)")
    fig.update_layout(showlegend=False)
    st.plotly_chart(ui.style_fig(fig))

# ------------------------------------------------------------------ baseline
with tab_baseline:
    st.markdown("**Does the LLM verdict agree with a simple text baseline?**")
    st.caption(
        "Alignment is the best TF-IDF cosine similarity between the video's text and its news articles, computed "
        "independently of the LLM judge. It is a sanity check, not a substitute: pure wording overlap misses "
        "paraphrase and can be fooled by a topically similar but factually different article."
    )
    if "Alignment" not in df or df["Alignment"].notna().sum() == 0:
        st.info("Run the AI analysis on the Automate page to see this comparison.")
    else:
        scored = df.dropna(subset=["Alignment"])
        fig = px.strip(scored, x="Status", y="Alignment", color="Status", category_orders={"Status": ui.STATUS_ORDER},
                       color_discrete_map=ui.STATUS_COLORS, hover_name="Video Title", title="Wording overlap by verdict")
        fig.update_traces(jitter=0.35, marker=dict(size=9))
        st.plotly_chart(ui.style_fig(fig))

        disagreements = scored[scored["Disagreement"]]
        st.metric("Disagreements flagged", len(disagreements),
                  help="Green verdicts with low wording overlap, or Red verdicts with high wording overlap.")
        if not disagreements.empty:
            st.dataframe(
                disagreements[["Video Title", "Status", "Confidence", "Alignment", "Reasoning"]],
                hide_index=True, width="stretch",
                column_config={"Confidence": st.column_config.NumberColumn(format="percent"),
                               "Alignment": st.column_config.NumberColumn(format="%.2f")},
            )

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.markdown("**How well does the AI verdict match your own review?**")
    labelled = df.dropna(subset=["Human Verdict"]) if "Human Verdict" in df else pd.DataFrame()
    st.caption("Record verdicts in the review queue (the Channels & review queue tab) to build this evaluation set.")
    if labelled.empty:
        st.info("No videos have a recorded human verdict yet.")
    else:
        st.metric("Labelled videos", len(labelled))
        report, matrix = features.classification_summary(labelled["Human Verdict"], labelled["Status"])
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Precision, recall, F1 by class**")
            st.dataframe(report.round(3), width="stretch")
        with right:
            st.markdown("**Confusion matrix** (rows: human, columns: AI)")
            st.dataframe(matrix, width="stretch")
        heat = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", aspect="auto")
        st.plotly_chart(ui.style_fig(heat, 360))

st.download_button("Download this dataset with all engineered features", state.to_csv_bytes(df),
                   "netshield_features.csv", "text/csv")
