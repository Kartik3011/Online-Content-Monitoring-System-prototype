import plotly.express as px
import streamlit as st

from module import state, ui

ui.setup_page("Social Content Report", "📊")
ui.page_header(
    "Social Content Report",
    "The raw data returned by YouTube, with engagement measures added. No AI is used on this page.",
)
ui.callout(
    "Report Overview",
    items=[
        "This page provides a comprehensive report based on the data fetched from YouTube.",
        "Please see the filters to analyze video view trends across selected channels.",
    ],
)

df = state.enriched()
if df is None:
    ui.empty_state("No video data yet", "Fetch some videos first, then come back to explore them.",
                   "pages/01Request Analysis.py", "Go to Request Analysis")

channels = st.multiselect("Filter by channel", sorted(df["Channel Title"].unique()), placeholder="All channels")
view = df[df["Channel Title"].isin(channels)] if channels else df
if view.empty:
    st.warning("No videos match this filter.")
    st.stop()

median_eng = view["Engagement Rate"].median()
ui.kpis([
    ("Videos", len(view)),
    ("Channels", view["Channel ID"].nunique()),
    ("Total views", f"{int(view['Views'].sum()):,}"),
    ("Median engagement", f"{median_eng:.1%}" if median_eng == median_eng else "n/a",
     "(likes + comments) / views. Videos with zero views are left out."),
    ("Median views per day", f"{view['Views per Day'].median():,.0f}"),
])

st.subheader("Videos")
table_cols = ["Video Title", "Channel Title", "Published At", "Views", "Likes", "Comments",
              "Engagement Rate", "Views per Day", "Subscriber Count", "Video URL"]
st.dataframe(
    view[table_cols],
    hide_index=True,
    width="stretch",
    column_config={
        "Video URL": st.column_config.LinkColumn("Link", display_text="Watch"),
        "Engagement Rate": st.column_config.NumberColumn(format="percent"),
        "Views per Day": st.column_config.NumberColumn(format="localized"),
        "Published At": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
    },
)
st.download_button("Download as CSV", state.to_csv_bytes(view), "videos.csv", "text/csv")

st.subheader("Reach and timing")
left, right = st.columns(2)
with left:
    fig = px.scatter(
        view, x="Published At", y="Views", color="Channel Title", log_y=True,
        hover_name="Video Title", hover_data={"Channel Title": False, "Likes": True},
        title="Views by publish date (log scale)",
    )
    fig.update_traces(marker=dict(size=11, line=dict(width=0)))
    fig.update_layout(showlegend=view["Channel Title"].nunique() <= 8)
    st.plotly_chart(ui.style_fig(fig))
with right:
    top = view.groupby("Channel Title", as_index=False)["Views"].sum().nlargest(10, "Views").sort_values("Views")
    fig = px.bar(top, x="Views", y="Channel Title", orientation="h", title="Top channels by total views")
    fig.update_traces(marker_color="#4f8ef7")
    fig.update_yaxes(title=None)
    st.plotly_chart(ui.style_fig(fig))

ui.callout(
    "NOTE",
    items=[
        "This report displays raw data returned by the YouTube API.",
        "AI-powered filtering and misinformation detection (Green/Yellow/Red status) is performed in the <b>AUTOMATE</b> section.",
    ],
    tone="ok",
)
ui.callout(
    "About these numbers",
    items=[
        "Views, likes and comments are snapshots from the moment of fetching.",
        "Hidden subscriber counts and disabled comments show as empty, not zero.",
    ],
)
