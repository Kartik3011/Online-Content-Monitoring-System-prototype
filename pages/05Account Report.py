import plotly.express as px
import streamlit as st

from module import state, ui

ui.setup_page("Account Report", "📑")
ui.page_header("Account Report", "The final results after the Automate page has finished.")
ui.callout(
    "Misinformation Account Analysis",
    items=[
        "This report sorts accounts based on the level of misinformation detected in their content (Red, Yellow, Green).",
        "The chart below shows the breakdown of videos by severity status.",
        "Videos marked Red are automatically flagged for immediate review.",
    ],
)

df = state.enriched()
if df is None or "Status" not in df:
    ui.callout(
        text="No 'Accountreport' found for this session. Please run the complete AUTOMATE step to generate the analysis.",
        tone="warn",
    )
    st.page_link("pages/06Automate.py", label="Go to Automate Page")
    st.stop()

first = df.iloc[0]
st.caption(f"Run {first['Run ID']}  |  {first['Analysed At']}  |  judge {first['Judge Model']}  |  prompt version {first['Prompt Version']}")

st.dataframe(
    df[["Video Title", "Channel Title", "Status", "Confidence", "Reasoning", "Video URL"]],
    hide_index=True, width="stretch",
    column_config={
        "Confidence": st.column_config.NumberColumn(format="percent"),
        "Video URL": st.column_config.LinkColumn("Link", display_text="Watch"),
    },
)
st.download_button("Download full report", state.to_csv_bytes(df), "netshield_report.csv", "text/csv")

st.markdown("---")
st.subheader("Filter Status and Visualize Breakdown")
chosen = st.multiselect("Choose Status for Visualization", ui.STATUS_ORDER, ui.STATUS_ORDER)

if not chosen:
    st.error("Please select at least one status to visualize.")
else:
    filtered = df[df["Status"].isin(chosen)]
    counts = filtered["Status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]

    st.markdown("### Status Count Breakdown")
    fig = px.bar(counts, x="Status", y="Count", color="Status", color_discrete_map=ui.STATUS_COLORS,
                category_orders={"Status": ui.STATUS_ORDER}, title="Account Status Distribution")
    fig.update_layout(showlegend=False)
    st.plotly_chart(ui.style_fig(fig, 320))

    st.markdown("---")
    st.subheader("Red Flagged Accounts for Termination 🚨")
    red = df[df["Status"] == "Red"]
    if red.empty:
        st.info("No accounts currently flagged with 'Red' status.")
    else:
        ui.callout(
            text="This is a shortlist for a person to check. A red verdict is a model's judgement of text, "
                 "not evidence that an account should be removed.",
            tone="warn",
        )
        st.dataframe(
            red[["Video Title", "Channel Title", "Confidence", "Reasoning", "Video URL", "Channel URL"]],
            hide_index=True, width="stretch",
            column_config={
                "Confidence": st.column_config.NumberColumn(format="percent"),
                "Video URL": st.column_config.LinkColumn("Video", display_text="Watch"),
                "Channel URL": st.column_config.LinkColumn("Channel", display_text="Open"),
            },
        )
        st.download_button("Download red-flagged list (terminate.csv)", state.to_csv_bytes(red), "terminate.csv", "text/csv")

st.page_link("pages/07Analytics Lab.py", label="Want per-channel breakdowns and human-review scoring? Open Analytics Lab")
