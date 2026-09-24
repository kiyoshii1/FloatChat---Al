"""
FloatChat - Database Access Layer
==================================
Thin wrapper around the SQLite store built by data/generate_argo_data.py.
All query functions take a `QueryFilter` (see nlp_engine.py) and return
pandas DataFrames, so the visualizer layer stays storage-agnostic.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "argo_data.db")
DB_PATH = os.path.abspath(DB_PATH)


def _connect():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"No database found at {DB_PATH}. Run `python data/generate_argo_data.py` first."
        )
    return sqlite3.connect(DB_PATH)


def list_regions():
    conn = _connect()
    df = pd.read_sql("SELECT DISTINCT region FROM floats ORDER BY region", conn)
    conn.close()
    return df["region"].tolist()


def list_floats(region=None):
    conn = _connect()
    q = "SELECT * FROM floats"
    params = []
    if region:
        q += " WHERE region = ?"
        params.append(region)
    df = pd.read_sql(q, conn, params=params)
    conn.close()
    return df


def get_profiles(filt):
    """
    filt: QueryFilter dataclass (region, float_id, date_start, date_end,
          season_months, lat_range, lon_range)
    Returns profile-level metadata matching the filter.
    """
    conn = _connect()
    clauses = []
    params = []

    if filt.float_id:
        clauses.append("p.float_id = ?")
        params.append(filt.float_id)
    if filt.region:
        clauses.append("p.region = ?")
        params.append(filt.region)
    if filt.date_start:
        clauses.append("p.date >= ?")
        params.append(filt.date_start)
    if filt.date_end:
        clauses.append("p.date <= ?")
        params.append(filt.date_end)
    if filt.lat_range:
        clauses.append("p.latitude BETWEEN ? AND ?")
        params.extend(filt.lat_range)
    if filt.lon_range:
        clauses.append("p.longitude BETWEEN ? AND ?")
        params.extend(filt.lon_range)
    if filt.bgc_only:
        clauses.append("p.has_bgc = 1")
    if filt.data_mode:
        clauses.append("p.data_mode = ?")
        params.append(filt.data_mode)

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    q = f"""
        SELECT p.*, f.platform_type, f.wmo_id
        FROM profiles p
        JOIN floats f ON f.float_id = p.float_id
        {where}
        ORDER BY p.date
    """
    df = pd.read_sql(q, conn, params=params)
    conn.close()

    if filt.season_months and not df.empty:
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df = df[df["month"].isin(filt.season_months)]

    return df


def get_measurements(profile_ids):
    if len(profile_ids) == 0:
        return pd.DataFrame()
    conn = _connect()
    placeholders = ",".join("?" for _ in profile_ids)
    q = f"SELECT * FROM measurements WHERE profile_id IN ({placeholders})"
    df = pd.read_sql(q, conn, params=list(profile_ids))
    conn.close()
    return df


def get_profiles_with_measurements(filt, max_profiles=60):
    """Convenience: get filtered profiles + their measurement rows joined."""
    profiles = get_profiles(filt)
    if profiles.empty:
        return profiles, pd.DataFrame()
    # cap for demo performance / plot readability
    sampled = profiles.head(max_profiles)
    meas = get_measurements(sampled["profile_id"].tolist())
    merged = meas.merge(
        sampled[["profile_id", "float_id", "date", "latitude", "longitude",
                 "region", "data_mode", "wmo_id"]],
        on="profile_id", how="left"
    )
    return profiles, merged


def float_overview():
    """
    One row per float: latest known position + basic metadata + profile count.
    Powers the dashboard overview map and the float registry table.
    """
    conn = _connect()
    q = """
        SELECT f.float_id, f.wmo_id, f.region, f.platform_type, f.is_bgc,
               f.launch_date, f.project_name,
               COUNT(p.profile_id) AS n_profiles,
               MAX(p.date) AS last_date
        FROM floats f
        LEFT JOIN profiles p ON p.float_id = f.float_id
        GROUP BY f.float_id
    """
    floats = pd.read_sql(q, conn)

    # latest lat/lon per float (last profile by date)
    latest_pos = pd.read_sql("""
        SELECT p.float_id, p.latitude, p.longitude, p.date
        FROM profiles p
        INNER JOIN (
            SELECT float_id, MAX(date) AS max_date FROM profiles GROUP BY float_id
        ) m ON p.float_id = m.float_id AND p.date = m.max_date
    """, conn)
    conn.close()

    latest_pos = latest_pos.drop_duplicates(subset="float_id")
    merged = floats.merge(latest_pos[["float_id", "latitude", "longitude"]], on="float_id", how="left")
    return merged


def region_counts():
    conn = _connect()
    df = pd.read_sql("""
        SELECT region, COUNT(DISTINCT float_id) AS floats, COUNT(*) AS profiles
        FROM profiles GROUP BY region ORDER BY profiles DESC
    """, conn)
    conn.close()
    return df


def data_mode_counts():
    conn = _connect()
    df = pd.read_sql("SELECT data_mode, COUNT(*) AS n FROM profiles GROUP BY data_mode", conn)
    conn.close()
    return df


def monthly_profile_counts():
    conn = _connect()
    df = pd.read_sql("SELECT date FROM profiles", conn)
    conn.close()
    if df.empty:
        return df
    df["month"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    ts = df.groupby("month").size().reset_index(name="profiles")
    return ts


def db_stats():
    conn = _connect()
    stats = {
        "floats": pd.read_sql("SELECT COUNT(*) c FROM floats", conn).iloc[0, 0],
        "profiles": pd.read_sql("SELECT COUNT(*) c FROM profiles", conn).iloc[0, 0],
        "measurements": pd.read_sql("SELECT COUNT(*) c FROM measurements", conn).iloc[0, 0],
        "date_min": pd.read_sql("SELECT MIN(date) d FROM profiles", conn).iloc[0, 0],
        "date_max": pd.read_sql("SELECT MAX(date) d FROM profiles", conn).iloc[0, 0],
        "regions": list_regions(),
    }
    conn.close()
    return stats
