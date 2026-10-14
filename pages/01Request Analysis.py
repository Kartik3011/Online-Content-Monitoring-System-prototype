import html
from datetime import date, datetime, timedelta

import plotly.express as px
import requests
import streamlit as st

from module import state, ui, yextractor

ui.setup_page("Request Analysis", "🔎")

# Nominatim's usage policy asks for a User-Agent that identifies your app. Put your own contact here.
NOMINATIM_UA = "NetShield/2.0 (Streamlit app; contact: replace-with-your-email@example.com)"


@st.cache_data(ttl=86400, show_spinner=False)
def geocode(city: str):
    """City name to (lat, lon). Cached for a day so repeat searches don't hit the free service."""
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"city": city, "format": "json", "limit": 1},
            headers={"User-Agent": NOMINATIM_UA},
            timeout=6,
        )
        response.raise_for_status()
        data = response.json()
        return (float(data[0]["lat"]), float(data[0]["lon"])) if data else (None, None)
    except Exception:  # noqa: BLE001
        return None, None


ui.page_header(
    "Request Analysis",
    "Choose a topic and a period. NetShield fetches matching YouTube videos together with their channel statistics.",
)

today = date.today()
with st.form("fetch_form"):
    c1, c2 = st.columns(2)
    keyword = c1.text_input("Keyword or hashtag", "Air pollution and AQI news")
    city = c2.text_input("City (optional)", "New Delhi", help="Limits results to videos geotagged near this city. Leave empty to search everywhere.")

    c3, c4, c5, c6 = st.columns(4)
    start = c3.date_input("From", today - timedelta(days=30))
    end = c4.date_input("To", today)
    radius = c5.selectbox("Search radius", ["10km", "25km", "50km", "100km", "250km", "500km"], index=2)
    limit = c6.number_input("Maximum videos", min_value=1, max_value=50, value=8)

    b1, b2, _ = st.columns([1.1, 1.3, 2])
    review = b1.form_submit_button("Fetch Data & Go to Content Report")
    run = b2.form_submit_button("Complete Analysis & Switch to Automate", type="primary")
st.caption("Heads up: each fetch costs roughly 100 YouTube quota units. You get 10,000 a day, so there's room for plenty of searches.")

if review or run:
    if not keyword.strip():
        st.error("Enter a keyword or hashtag.")
        st.stop()
    if start > end:
        st.error("The start date must be on or before the end date.")
        st.stop()

    try:
        with st.status("Fetching videos from YouTube...", expanded=True) as status:
            lat = lon = None
            if city.strip():
                lat, lon = geocode(city.strip())
                if lat is None:
                    st.write(f"Could not find coordinates for '{city}'. Searching without a location filter.")
                else:
                    st.write(f"Location set to {city} ({lat:.2f}, {lon:.2f}), radius {radius}.")
            df, notes = yextractor.fetch_videos(
                keyword.strip(), lat, lon, radius, int(limit),
                datetime.combine(start, datetime.min.time()), datetime.combine(end, datetime.max.time()),
            )
            status.update(label=f"Fetched {len(df)} videos", state="complete", expanded=False)
    except yextractor.YouTubeError as exc:
        st.error(str(exc))
        st.stop()

    if df.empty:
        st.warning("No videos matched. Try a broader keyword, a longer date range, or clear the city.")
        for note in notes:
            st.caption(note)
        st.stop()

    state.set_videos(df, {"keyword": keyword.strip(), "city": city.strip(), "notes": notes,
                          "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    if run:
        st.switch_page("pages/06Automate.py")

    st.success(f"Loaded {len(df)} videos from {df['Channel ID'].nunique()} channels.")
    for note in notes:
        ui.callout(text=note, tone="warn")

    shown = df[["Video Title", "Channel Title", "Published At", "Views", "Video URL"]]
    st.dataframe(
        shown, hide_index=True, width="stretch",
        column_config={"Video URL": st.column_config.LinkColumn("Link", display_text="Watch")},
    )

    preview = df.nlargest(min(10, len(df)), "Views").sort_values("Views")
    fig = px.bar(preview, x="Views", y="Video Title", orientation="h", title="Views for this batch")
    fig.update_traces(marker_color="#4f8ef7")
    fig.update_yaxes(title=None, tickfont=dict(size=11))
    st.plotly_chart(ui.style_fig(fig, 280))

    st.page_link("pages/02Social Content Report.py", label="Open the social content report")
    st.page_link("pages/06Automate.py", label="Run the AI analysis")

elif state.get_videos() is not None:
    params, current = state.get_params(), state.get_videos()
    ui.callout(
        "Current dataset",
        text=f"{len(current)} videos for <b>{html.escape(str(params.get('keyword', '')))}</b>, fetched {params.get('fetched_at', '')}. "
             "Fetching again replaces this dataset and clears its analysis.",
        tone="info",
    )

ui.callout(
    "Note on Re-Searching",
    items=[
        "Fetching a new keyword or hashtag replaces the current dataset for this browser session, and clears "
        "any analysis that was run on the old one.",
        "Unlike the original disk-based version, data here lives in your own session, so there's no shared "
        "cache to clear between searches and no need to refresh the page.",
    ],
    tone="ok",
)
