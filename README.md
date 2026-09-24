# 🌊 FloatChat — AI-Powered Conversational Interface for ARGO Ocean Data

**SIH25040** · Ministry of Earth Sciences (MoES) · INCOIS · Software · Miscellaneous

A working prototype: ask an oceanography question in plain English, get real
Argo-style profiles, trajectories, and diagrams back — no NetCDF, no SQL, no
scripting.

```
"Show me salinity profiles near the Bay of Bengal for 2024"
"Compare salinity profiles in the Arabian Sea between summer and winter"
"Show float trajectory for float 2900153"
"Temperature vs salinity diagram for Equatorial Indian Ocean"
"Salinity trend over time in Bay of Bengal"
"Show BGC oxygen profiles in the Andaman Sea"
```

---

## Quick start

```bash
pip install -r requirements.txt
python data/generate_argo_data.py     # builds argo_data.db (one-time)
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

Optional — run the pipeline without a browser, useful for CI / quick checks:
```bash
python test_pipeline.py
```

### Enable the LLM layer (optional)
The app works fully offline out of the box. To let Claude generate the
natural-language answer text (instead of the deterministic template),
export an API key before launching:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
streamlit run app.py
```
The LLM is only ever given **already-retrieved, grounded numbers** to phrase
into a sentence — it never invents data or queries the database directly.
This avoids the "LLM answers from its own knowledge" hallucination failure
mode.

---

## Architecture

```
User question
     │
     ▼
┌─────────────────────┐
│  nlp_engine.py       │  regex/keyword parser → QueryFilter
│  (+ conversational    │  (region, variable, dates, season,
│   memory / context)   │   float_id, plot_type, BGC flag...)
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  database.py         │  SQLite: floats / profiles / measurements
│                       │  (pre-processed — no live NetCDF parsing)
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  visualizer.py        │  intent-aware auto-viz:
│                       │  profile · trajectory · T-S diagram ·
│                       │  time series · season comparison
└─────────┬────────────┘
          ▼
┌─────────────────────┐
│  llm_client.py        │  optional: Claude phrases the grounded
│  (graceful fallback)  │  stats into a natural sentence
└─────────┬────────────┘
          ▼
   app.py (Streamlit chat UI)
   + CSV export + provenance panel
```

### Why SQLite + pre-processing (not live NetCDF parsing)?
The problem statement explicitly calls for combining an LLM with "modern
structured databases" — parsing raw NetCDF per query is slow and was flagged
as a red flag evaluators notice. `data/generate_argo_data.py` mimics the
real ingestion step (`argopy`/`xarray` → relational store) by generating a
statistically realistic Indian-Ocean Argo dataset once, up front. Swap that
module for a real GDAC/`argopy` ingestion script and the rest of the stack
(schema, query engine, viz, chat UI) is unchanged.

### Data model
| Table | Grain | Key columns |
|---|---|---|
| `floats` | 1 row / float | `float_id`, `wmo_id`, `region`, `platform_type`, `is_bgc` |
| `profiles` | 1 row / float / cycle | `profile_id`, `float_id`, `date`, `latitude`, `longitude`, `data_mode` (R=real-time, D=delayed-mode/QC'd) |
| `measurements` | 1 row / depth level | `pressure_dbar`, `depth_m`, `temperature_c`, `salinity_psu`, `dissolved_oxygen_umol_kg`, `temp_qc`, `sal_qc` |

### Supported query intents (auto-detected)
| Intent | Trigger words | Plot |
|---|---|---|
| Depth profile | default | Line plot, depth vs variable, mean overlay |
| Trajectory | "trajectory", "path", "track", "map" | Lat/lon map, one line per float |
| T-S diagram | "temperature ... salinity ... diagram", "T-S" | Scatter, colored by depth |
| Time series | "trend", "over time", "time series" | Surface-layer value vs date |
| Seasonal comparison | "compare ... summer/winter/monsoon" | Two mean profiles overlaid |

### Conversational memory
Each turn's `QueryFilter` carries forward into the next turn (`st.session_state.context_filter`),
so follow-ups like *"now filter for 2023 only"* narrow the previous query
instead of starting over. Click **Reset conversation** in the sidebar to clear it.

---

## Project layout
```
floatchat/
├── app.py                       # Streamlit chat UI (entry point)
├── test_pipeline.py             # offline smoke test, no browser needed
├── requirements.txt
├── data/
│   └── generate_argo_data.py    # synthetic Indian-Ocean Argo dataset → argo_data.db
├── backend/
│   ├── database.py              # SQLite query layer
│   ├── nlp_engine.py            # NL → QueryFilter parser + conversational memory
│   ├── visualizer.py            # Plotly: profile/trajectory/T-S/timeseries/compare
│   └── llm_client.py            # optional Claude-generated summaries (graceful fallback)
└── argo_data.db                 # generated (git-ignore this in a real repo)
```

## Extending toward production
- **Real data**: replace `data/generate_argo_data.py` with an `argopy`/`xarray`
  ingestion job against the Ifremer/NOAA GDAC mirrors or ERDDAP, writing into
  the same three-table schema.
- **Scale**: swap SQLite for PostgreSQL + PostGIS (geospatial queries) or
  DuckDB/Parquet for a larger archive.
- **Smarter NL**: route low-confidence parses (ambiguous region/variable) to
  the LLM for disambiguation before falling back to a clarifying question —
  the PS explicitly flags "no handling of ambiguous queries" as a red flag.
- **Multilingual**: add a translation pass before `nlp_engine.parse_query`
  for Hindi/regional-language support (INCOIS public-outreach alignment).
- **QC transparency**: the schema already carries `temp_qc`/`sal_qc`/`data_mode`
  — surface these more prominently in the UI (e.g. a QC toggle) for
  oceanographer users.
