"""
FloatChat - Visualization Layer
=================================
Auto-generates the right oceanographic plot from a QueryFilter + the
data returned by the database layer. This is the "intent-aware
auto-visualization" innovation angle called out in the PS analysis:
depth-profile vs trajectory-map vs T-S diagram vs time series, picked
automatically from query intent rather than a generic bar/line chart.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

VAR_LABELS = {
    "temperature_c": "Temperature (°C)",
    "salinity_psu": "Salinity (PSU)",
    "dissolved_oxygen_umol_kg": "Dissolved Oxygen (µmol/kg)",
}


def plot_profile(measurements: pd.DataFrame, variable: str, title: str):
    """Depth vs variable, one line per profile, colored by date."""
    if measurements.empty:
        return _empty_fig("No profiles matched this query.")

    fig = go.Figure()
    profile_ids = measurements["profile_id"].unique()
    dates = pd.to_datetime(measurements.groupby("profile_id")["date"].first())

    for pid in profile_ids[:40]:  # cap lines for readability
        sub = measurements[measurements["profile_id"] == pid].sort_values("depth_m")
        fig.add_trace(go.Scatter(
            x=sub[variable], y=-sub["depth_m"],
            mode="lines+markers", name=f"Float {sub['wmo_id'].iloc[0]} cyc",
            line=dict(width=1.4),
            marker=dict(size=3),
            opacity=0.55,
            showlegend=False,
            hovertemplate=f"%{{x}} @ %{{y}} m<br>Float {sub['wmo_id'].iloc[0]}<br>%{{customdata}}<extra></extra>",
            customdata=sub["date"],
        ))

    # mean profile overlay
    mean_prof = measurements.groupby("depth_m")[variable].mean().reset_index().sort_values("depth_m")
    fig.add_trace(go.Scatter(
        x=mean_prof[variable], y=-mean_prof["depth_m"],
        mode="lines", name="Mean profile",
        line=dict(width=3, color="black", dash="solid"),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=VAR_LABELS.get(variable, variable),
        yaxis_title="Depth (m)",
        template="plotly_dark",
        height=550,
        legend=dict(orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5"),
    )
    return fig


def plot_compare_profile(measurements: pd.DataFrame, variable: str,
                          group_col: str, group_labels: dict, title: str):
    """Two mean profiles overlaid, e.g. summer vs winter, for comparison."""
    if measurements.empty:
        return _empty_fig("No profiles matched this comparison.")

    fig = go.Figure()
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for i, (grp_val, label) in enumerate(group_labels.items()):
        sub = measurements[measurements[group_col] == grp_val]
        if sub.empty:
            continue
        mean_prof = sub.groupby("depth_m")[variable].mean().reset_index().sort_values("depth_m")
        std_prof = sub.groupby("depth_m")[variable].std().reset_index().sort_values("depth_m")
        fig.add_trace(go.Scatter(
            x=mean_prof[variable], y=-mean_prof["depth_m"],
            mode="lines+markers", name=label,
            line=dict(width=3, color=colors[i % len(colors)]),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=VAR_LABELS.get(variable, variable),
        yaxis_title="Depth (m)",
        template="plotly_dark",
        height=550,
        legend=dict(orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5"),
    )
    return fig


def plot_trajectory(profiles: pd.DataFrame, title: str):
    """Float trajectory map (lat/lon path colored by float)."""
    if profiles.empty:
        return _empty_fig("No float trajectories matched this query.")

    fig = px.line_map(
        profiles.sort_values(["float_id", "date"]),
        lat="latitude", lon="longitude", color=profiles["float_id"].astype(str),
        hover_data=["date", "cycle_number", "data_mode"],
        title=title,
    )
    fig.update_traces(mode="lines+markers", marker=dict(size=5))
    fig.update_layout(
        map_style="carto-darkmatter",
        map_zoom=3,
        map_center={"lat": profiles["latitude"].mean(), "lon": profiles["longitude"].mean()},
        height=580,
        margin=dict(l=0, r=0, t=40, b=0),
        legend_title="Float ID",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5"),
    )
    return fig


def plot_ts_diagram(measurements: pd.DataFrame, title: str):
    """Temperature-Salinity diagram, colored by depth."""
    if measurements.empty:
        return _empty_fig("No data available for a T-S diagram.")

    fig = px.scatter(
        measurements, x="salinity_psu", y="temperature_c",
        color="depth_m", color_continuous_scale="Viridis_r",
        hover_data=["wmo_id", "date"],
        title=title,
    )
    fig.update_layout(
        xaxis_title="Salinity (PSU)",
        yaxis_title="Temperature (°C)",
        template="plotly_dark",
        height=550,
        coloraxis_colorbar_title="Depth (m)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5"),
    )
    return fig


def plot_timeseries(measurements: pd.DataFrame, variable: str, title: str, surface_only=True):
    """Surface-layer variable trend over time, aggregated by profile date."""
    if measurements.empty:
        return _empty_fig("No data available for a time series.")

    df = measurements[measurements["depth_m"] <= 10] if surface_only else measurements
    ts = df.groupby("date")[variable].mean().reset_index().sort_values("date")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(ts["date"]), y=ts[variable],
        mode="lines+markers", line=dict(width=2, color="#1f77b4"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=VAR_LABELS.get(variable, variable) + (" (surface, ≤10 m)" if surface_only else ""),
        template="plotly_dark",
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5"),
    )
    return fig


REGION_COLORS = {
    "Arabian Sea": "#00B4D8",
    "Bay of Bengal": "#48CAE4",
    "Equatorial Indian Ocean": "#90E0EF",
    "Andaman Sea": "#0077B6",
    "Southern Indian Ocean": "#023E8A",
}


def plot_overview_map(float_overview: pd.DataFrame, dark=True):
    """All-float snapshot map for the dashboard: one marker per float at its
    latest known position, colored by region, sized by profile count."""
    if float_overview.empty or float_overview["latitude"].isna().all():
        return _empty_fig("No float position data available.", dark=dark)

    df = float_overview.dropna(subset=["latitude", "longitude"])
    fig = px.scatter_map(
        df, lat="latitude", lon="longitude",
        color="region", size="n_profiles",
        size_max=22, zoom=2.4,
        hover_name="wmo_id",
        hover_data={"region": True, "platform_type": True, "is_bgc": True,
                    "n_profiles": True, "latitude": False, "longitude": False},
        color_discrete_map=REGION_COLORS,
    )
    fig.update_layout(
        map_style="carto-darkmatter" if dark else "carto-positron",
        height=460,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E6F1F5" if dark else "#0B1F3A")),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5" if dark else "#0B1F3A"),
    )
    return fig


def plot_region_bar(region_df: pd.DataFrame, dark=True):
    if region_df.empty:
        return _empty_fig("No regional data available.", dark=dark)
    fig = px.bar(
        region_df.sort_values("profiles"), x="profiles", y="region", orientation="h",
        color="region", color_discrete_map=REGION_COLORS, text="profiles",
    )
    fig.update_traces(textposition="outside", showlegend=False)
    fig.update_layout(
        height=300, margin=dict(l=0, r=10, t=10, b=0),
        xaxis_title="Profiles", yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5" if dark else "#0B1F3A"),
    )
    return fig


def plot_data_mode_donut(mode_df: pd.DataFrame, dark=True):
    if mode_df.empty:
        return _empty_fig("No data-mode information available.", dark=dark)
    label_map = {"R": "Real-time", "D": "Delayed-mode"}
    mode_df = mode_df.copy()
    mode_df["label"] = mode_df["data_mode"].map(label_map).fillna(mode_df["data_mode"])
    fig = px.pie(
        mode_df, names="label", values="n", hole=0.55,
        color="label", color_discrete_map={"Real-time": "#48CAE4", "Delayed-mode": "#023E8A"},
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(
        height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5" if dark else "#0B1F3A"),
    )
    return fig


def plot_monthly_trend(ts_df: pd.DataFrame, dark=True):
    if ts_df.empty:
        return _empty_fig("No time coverage data available.", dark=dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["month"], y=ts_df["profiles"], mode="lines", fill="tozeroy",
        line=dict(width=2, color="#00B4D8"), fillcolor="rgba(0,180,216,0.18)",
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=10, t=10, b=0),
        xaxis_title="", yaxis_title="Profiles / month",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E6F1F5" if dark else "#0B1F3A"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
    )
    return fig


def _empty_fig(message, dark=True):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False,
                        font=dict(size=16, color="#E6F1F5" if dark else "#0B1F3A"))
    fig.update_layout(
        height=350, xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig