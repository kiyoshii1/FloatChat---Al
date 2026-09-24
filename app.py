"""
FloatChat — AI-Powered Conversational Interface for ARGO Ocean Data
======================================================================
SIH25040 · Ministry of Earth Sciences (MoES) · INCOIS

Run:
    streamlit run app.py

First time setup:
    pip install -r requirements.txt
    python data/generate_argo_data.py     # builds argo_data.db
    streamlit run app.py
"""

import uuid
import streamlit as st
import pandas as pd
from dataclasses import asdict

from backend import database as db
from backend.nlp_engine import parse_query, describe_filter, QueryFilter, CANONICAL_VARIABLE_LABEL
from backend import visualizer as viz
from backend.llm_client import llm_available, summarize_with_llm, template_summary

st.set_page_config(
    page_title="FloatChat — Ocean Intelligence Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme — dark ocean-intelligence dashboard
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(0,119,182,0.20) 0%, transparent 45%),
        radial-gradient(circle at 85% 15%, rgba(72,202,228,0.12) 0%, transparent 40%),
        linear-gradient(180deg, #061323 0%, #071A2E 40%, #05101D 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050E1B 0%, #071626 100%);
    border-right: 1px solid rgba(72,202,228,0.15);
}
section[data-testid="stSidebar"] * { color: #CFEFFA !important; }

/* Hero header */
.fc-hero {
    padding: 1.6rem 2rem;
    border-radius: 18px;
    background: linear-gradient(120deg, rgba(0,119,182,0.35), rgba(2,62,138,0.25) 60%, rgba(6,19,35,0.1));
    border: 1px solid rgba(72,202,228,0.25);
    margin-bottom: 1.4rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.25);
}
.fc-hero h1 {
    font-size: 2.1rem; font-weight: 800; margin: 0;
    background: linear-gradient(90deg, #90E0EF, #48CAE4 45%, #CAF0F8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}
.fc-hero p { color: #9FD8E8; margin: 0.35rem 0 0 0; font-size: 0.98rem; }
.fc-badges { margin-top: 0.7rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.fc-badge {
    display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px;
    background: rgba(72,202,228,0.12); border: 1px solid rgba(72,202,228,0.35);
    color: #ADE8F4; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.3px;
}

/* KPI cards */
.fc-kpi {
    background: linear-gradient(160deg, rgba(20,44,74,0.75), rgba(9,24,43,0.75));
    border: 1px solid rgba(72,202,228,0.18);
    border-radius: 14px;
    padding: 1rem 1.1rem;
    height: 100%;
}
.fc-kpi .label { color: #7FB8CC; font-size: 0.76rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
.fc-kpi .value { color: #EAF9FF; font-size: 1.65rem; font-weight: 800; margin-top: 0.15rem; font-family: 'JetBrains Mono', monospace; }
.fc-kpi .sub { color: #5C93A8; font-size: 0.74rem; margin-top: 0.2rem; }

.fc-panel {
    background: rgba(12,28,48,0.55);
    border: 1px solid rgba(72,202,228,0.14);
    border-radius: 16px;
    padding: 1.1rem 1.2rem 0.6rem 1.2rem;
    margin-bottom: 1rem;
}
.fc-panel h4 { color: #CDEFFA; font-size: 0.95rem; margin: 0 0 0.6rem 0; font-weight: 700; }

/* Tabs */
button[data-baseweb="tab"] {
    color: #7FB8CC !important; font-weight: 600; font-size: 0.95rem;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #EAF9FF !important; border-bottom: 3px solid #48CAE4 !important;
}

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    background: rgba(14,32,54,0.55);
    border: 1px solid rgba(72,202,228,0.12);
    border-radius: 14px;
}

/* Metrics native */
div[data-testid="stMetric"] {
    background: rgba(14,32,54,0.55);
    border: 1px solid rgba(72,202,228,0.15);
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
}
div[data-testid="stMetricLabel"] { color: #7FB8CC !important; }
div[data-testid="stMetricValue"] { color: #EAF9FF !important; }

/* Dataframes */
div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

hr { border-color: rgba(72,202,228,0.15); }
</style>
""", unsafe_allow_html=True)


def kpi_card(label, value, sub=""):
    st.markdown(f"""
    <div class="fc-kpi">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def panel_start(title):
    st.markdown(f'<div class="fc-panel"><h4>{title}</h4>', unsafe_allow_html=True)


def panel_end():
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "context_filter" not in st.session_state:
    st.session_state.context_filter = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------------------------------------------------------------------------
# Load stats once (used by hero, sidebar, dashboard)
# ---------------------------------------------------------------------------
try:
    stats = db.db_stats()
    db_ok = True
except FileNotFoundError:
    db_ok = False

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🌊 FloatChat")
    st.caption("Ocean Intelligence Platform")
    st.caption("SIH25040 · MoES / INCOIS")
    st.divider()

    if not db_ok:
        st.error("No database found.")
        st.code("python data/generate_argo_data.py", language="bash")
        st.stop()

    st.success("Database connected")
    st.metric("Floats", stats["floats"])
    c1, c2 = st.columns(2)
    c1.metric("Profiles", stats["profiles"])
    c2.metric("Measurements", f"{stats['measurements']:,}")
    st.caption(f"📅 {stats['date_min']} → {stats['date_max']}")
    st.caption("🌍 " + ", ".join(stats["regions"]))

    st.divider()
    st.caption("🤖 LLM layer: " + ("Claude-enhanced ✅" if llm_available() else "Offline / rule-based mode"))
    if not llm_available():
        st.caption("Set ANTHROPIC_API_KEY to enable richer natural-language answers.")

    st.divider()
    st.markdown("**Try asking:**")
    examples = [
        "Show me salinity profiles near the Bay of Bengal for 2024",
        "Compare salinity profiles in the Arabian Sea between summer and winter",
        "Show float trajectory for float 2900153",
        "Temperature vs salinity diagram for Equatorial Indian Ocean",
        "Salinity trend over time in Bay of Bengal",
        "Show BGC oxygen profiles in the Andaman Sea",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}", use_container_width=True):
            st.session_state.pending_query = ex
            st.session_state.active_tab = "chat"

    st.divider()
    if st.button("🔄 Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.context_filter = None
        st.session_state.last_result = None
        st.rerun()

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="fc-hero">
    <h1>🌊 FloatChat — Ocean Intelligence Platform</h1>
    <p>AI-powered conversational access to ARGO float data — profiles, trajectories, spatial patterns, and trends, explained.</p>
    <div class="fc-badges">
        <span class="fc-badge">🛰️ {stats['floats']} floats</span>
        <span class="fc-badge">📈 {stats['profiles']:,} profiles</span>
        <span class="fc-badge">🌍 {len(stats['regions'])} regions</span>
        <span class="fc-badge">🤖 {'Claude-enhanced' if llm_available() else 'Offline mode'}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Core: run one chat query end-to-end (unchanged logic, reused across tabs)
# ---------------------------------------------------------------------------
def run_query(user_text: str):
    filt = parse_query(user_text, context=st.session_state.context_filter)
    st.session_state.context_filter = filt

    profiles, measurements = db.get_profiles_with_measurements(filt)

    result_stats = {}
    fig = None
    var_label = CANONICAL_VARIABLE_LABEL.get(filt.variable, filt.variable)

    if profiles.empty:
        result_stats = {"n_profiles": 0}
    else:
        result_stats["n_profiles"] = len(profiles)
        result_stats["n_floats"] = profiles["float_id"].nunique()
        result_stats["date_range"] = (profiles["date"].min(), profiles["date"].max())
        result_stats["delayed_pct"] = 100.0 * (profiles["data_mode"] == "D").mean()
        result_stats["var_label"] = var_label

        if not measurements.empty and filt.variable in measurements.columns:
            result_stats["mean_value"] = float(measurements[filt.variable].mean())
            result_stats["min_value"] = float(measurements[filt.variable].min())
            result_stats["max_value"] = float(measurements[filt.variable].max())

        title = f"{var_label.title()} — {filt.region or (('Float ' + str(filt.float_id)) if filt.float_id else 'All regions')}"

        if filt.plot_type == "trajectory":
            fig = viz.plot_trajectory(profiles, title=f"Float trajectory — {filt.region or filt.float_id}")
        elif filt.plot_type == "ts_diagram":
            fig = viz.plot_ts_diagram(measurements, title=f"T-S Diagram — {filt.region or 'selected floats'}")
        elif filt.plot_type == "timeseries":
            fig = viz.plot_timeseries(measurements, filt.variable, title=f"{title} — trend over time")
        elif filt.plot_type == "compare_profile" and filt.compare_season_months:
            measurements["_season_group"] = measurements["date"].apply(
                lambda d: filt.season_label if pd.to_datetime(d).month in filt.season_months
                else (filt.compare_season_label if pd.to_datetime(d).month in filt.compare_season_months else None)
            )
            group_labels = {filt.season_label: filt.season_label.title(),
                             filt.compare_season_label: filt.compare_season_label.title()}
            fig = viz.plot_compare_profile(measurements, filt.variable, "_season_group", group_labels,
                                            title=f"{title} — {filt.season_label} vs {filt.compare_season_label}")
        else:
            fig = viz.plot_profile(measurements, filt.variable, title=title)

    filt_desc = describe_filter(filt)

    if llm_available():
        summary = summarize_with_llm(user_text, filt_desc, result_stats) or template_summary(user_text, filt_desc, result_stats)
    else:
        summary = template_summary(user_text, filt_desc, result_stats)

    return {
        "filter": filt, "filter_desc": filt_desc, "profiles": profiles,
        "measurements": measurements, "stats": result_stats, "fig": fig, "summary": summary,
    }


def render_result(result, key=None):
    st.markdown(result["summary"])
    st.caption("📍 Filter applied: " + result["filter_desc"])

    if result["fig"] is not None:
        chart_key = f"chat_fig_{key}" if key is not None else f"chat_fig_{uuid.uuid4().hex}"
        st.plotly_chart(result["fig"], use_container_width=True, key=chart_key)

    if not result["profiles"].empty:
        with st.expander(f"📄 Provenance & data ({len(result['profiles'])} profiles) — export below"):
            show_cols = ["profile_id", "float_id", "wmo_id", "cycle_number", "date",
                         "latitude", "longitude", "region", "data_mode"]
            show_cols = [c for c in show_cols if c in result["profiles"].columns]
            st.dataframe(result["profiles"][show_cols], use_container_width=True, height=220)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Export profiles (CSV)", result["profiles"].to_csv(index=False),
                    file_name="floatchat_profiles.csv", mime="text/csv", use_container_width=True,
                )
            with col2:
                if not result["measurements"].empty:
                    st.download_button(
                        "⬇️ Export measurements (CSV)", result["measurements"].to_csv(index=False),
                        file_name="floatchat_measurements.csv", mime="text/csv", use_container_width=True,
                    )

# ---------------------------------------------------------------------------
# Main navigation
# ---------------------------------------------------------------------------
tab_dashboard, tab_chat, tab_explorer, tab_data = st.tabs(
    ["📊 Dashboard", "🤖 AI Chat", "🗺️ Float Explorer", "📁 Data Explorer"]
)

# ============================= DASHBOARD TAB ===============================
with tab_dashboard:
    overview = db.float_overview()
    region_df = db.region_counts()
    mode_df = db.data_mode_counts()
    trend_df = db.monthly_profile_counts()

    bgc_pct = 100.0 * overview["is_bgc"].mean() if not overview.empty else 0
    span_days = (pd.to_datetime(stats["date_max"]) - pd.to_datetime(stats["date_min"])).days
    span_years = round(span_days / 365, 1)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1: kpi_card("Active Floats", stats["floats"], f"{len(stats['regions'])} regions")
    with k2: kpi_card("Total Profiles", f"{stats['profiles']:,}", "cycle observations")
    with k3: kpi_card("Measurements", f"{stats['measurements']:,}", "depth-level readings")
    with k4: kpi_card("BGC Floats", f"{bgc_pct:.0f}%", "biogeochemical-capable")
    with k5: kpi_card("Data Span", f"{span_years} yrs", f"{stats['date_min']} → {stats['date_max']}")

    st.write("")
    col_map, col_side = st.columns([2, 1])
    with col_map:
        panel_start("🗺️ Live Float Positions")
        st.plotly_chart(viz.plot_overview_map(overview), use_container_width=True, key="dashboard_overview_map")
        panel_end()
    with col_side:
        panel_start("🌍 Profiles by Region")
        st.plotly_chart(viz.plot_region_bar(region_df), use_container_width=True, key="dashboard_region_bar")
        panel_end()

    col_trend, col_mode = st.columns([2, 1])
    with col_trend:
        panel_start("📈 Profile Coverage Over Time")
        st.plotly_chart(viz.plot_monthly_trend(trend_df), use_container_width=True, key="dashboard_monthly_trend")
        panel_end()
    with col_mode:
        panel_start("🔧 Data Mode Split")
        st.plotly_chart(viz.plot_data_mode_donut(mode_df), use_container_width=True, key="dashboard_data_mode_donut")
        panel_end()

# ============================= AI CHAT TAB ==================================
with tab_chat:
    st.caption("Ask about ARGO ocean data in plain English — get real profiles, trajectories, and diagrams instantly.")

    for msg_idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                render_result(msg["result"], key=f"history_{msg_idx}")
            else:
                st.markdown(msg["content"])

    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input("Ask about Argo ocean data… e.g. 'Show salinity profiles in the Bay of Bengal for 2024'")
    query_to_run = pending or user_input

    if query_to_run:
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)

        with st.chat_message("assistant"):
            with st.spinner("Querying Argo database…"):
                result = run_query(query_to_run)
            render_result(result, key=f"new_{len(st.session_state.messages)}")

        st.session_state.messages.append({"role": "assistant", "result": result})
        st.session_state.last_result = result

# ============================= FLOAT EXPLORER TAB ===========================
with tab_explorer:
    overview = db.float_overview()
    st.caption("Browse every float in the fleet — filter by region or platform type, then inspect its position and stats.")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        region_pick = st.multiselect("Region", sorted(overview["region"].unique()), default=None)
    with fcol2:
        platform_pick = st.multiselect("Platform type", sorted(overview["platform_type"].dropna().unique()), default=None)
    with fcol3:
        bgc_only = st.checkbox("BGC floats only")

    filtered = overview.copy()
    if region_pick:
        filtered = filtered[filtered["region"].isin(region_pick)]
    if platform_pick:
        filtered = filtered[filtered["platform_type"].isin(platform_pick)]
    if bgc_only:
        filtered = filtered[filtered["is_bgc"] == 1]

    panel_start(f"🛰️ {len(filtered)} floats matching filters")
    st.plotly_chart(viz.plot_overview_map(filtered), use_container_width=True, key="explorer_overview_map")
    panel_end()

    st.dataframe(
        filtered[["wmo_id", "region", "platform_type", "is_bgc", "launch_date", "n_profiles", "last_date"]]
        .rename(columns={"wmo_id": "WMO ID", "region": "Region", "platform_type": "Platform",
                          "is_bgc": "BGC", "launch_date": "Launched", "n_profiles": "Profiles",
                          "last_date": "Last profile"}),
        use_container_width=True, height=320,
    )

# ============================= DATA EXPLORER TAB ============================
with tab_data:
    st.caption("Query the raw profile/measurement tables directly — useful for validation, export, or deeper analysis.")

    dcol1, dcol2, dcol3 = st.columns(3)
    with dcol1:
        d_region = st.selectbox("Region", ["All"] + sorted(stats["regions"]))
    with dcol2:
        d_start = st.text_input("Start date (YYYY-MM-DD)", value=stats["date_min"])
    with dcol3:
        d_end = st.text_input("End date (YYYY-MM-DD)", value=stats["date_max"])

    d_filt = QueryFilter(
        region=None if d_region == "All" else d_region,
        date_start=d_start or None, date_end=d_end or None,
    )
    d_profiles = db.get_profiles(d_filt)

    st.write(f"**{len(d_profiles)} profiles** match this filter.")
    st.dataframe(d_profiles, use_container_width=True, height=380)

    if not d_profiles.empty:
        st.download_button(
            "⬇️ Export filtered profiles (CSV)", d_profiles.to_csv(index=False),
            file_name="floatchat_data_explorer.csv", mime="text/csv",
        )