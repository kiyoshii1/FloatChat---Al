"""
Quick end-to-end smoke test (no Streamlit needed):
    python test_pipeline.py
Runs a handful of the PS's own example queries through the full
parse -> query -> visualize pipeline and prints a report.
"""

from backend.nlp_engine import parse_query, describe_filter
from backend import database as db
from backend import visualizer as viz
from backend.llm_client import template_summary

QUERIES = [
    "Show me salinity profiles near the Bay of Bengal for 2024",
    "Compare salinity profiles in the Arabian Sea between summer and winter",
    "Show float trajectory for float 2900153",
    "Temperature vs salinity diagram for Equatorial Indian Ocean",
    "Salinity trend over time in Bay of Bengal",
    "Show BGC oxygen profiles in the Andaman Sea",
]

def main():
    print("=" * 70)
    print("FloatChat pipeline smoke test")
    print("=" * 70)
    stats = db.db_stats()
    print(f"DB: {stats['floats']} floats, {stats['profiles']} profiles, "
          f"{stats['measurements']} measurements\n")

    context = None
    for q in QUERIES:
        filt = parse_query(q, context=None)  # each example is independent here
        profiles, measurements = db.get_profiles_with_measurements(filt)

        print(f"Q: {q}")
        print(f"  -> parsed filter : {describe_filter(filt)}")
        print(f"  -> plot type     : {filt.plot_type}")
        print(f"  -> matched       : {len(profiles)} profiles, {len(measurements)} measurement rows")

        s = {"n_profiles": len(profiles),
             "n_floats": profiles['float_id'].nunique() if not profiles.empty else 0,
             "var_label": filt.variable}
        print(f"  -> summary       : {template_summary(q, describe_filter(filt), s)}")

        # sanity: actually build the figure object, make sure it doesn't throw
        if filt.plot_type == "trajectory":
            fig = viz.plot_trajectory(profiles, "test")
        elif filt.plot_type == "ts_diagram":
            fig = viz.plot_ts_diagram(measurements, "test")
        elif filt.plot_type == "timeseries":
            fig = viz.plot_timeseries(measurements, filt.variable, "test")
        else:
            fig = viz.plot_profile(measurements, filt.variable, "test")
        assert fig is not None
        print(f"  -> figure built OK ({len(fig.data)} traces)\n")

    print("All smoke tests passed ✅")


if __name__ == "__main__":
    main()
