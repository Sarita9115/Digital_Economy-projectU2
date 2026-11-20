"""Yucatán Hurricane Early Warning & Response System (Yucatán EWRS) – Streamlit MVP.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import warnings
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from models.risk_model import (
    LANDFALL_TRACKS,
    RISK_COLORS,
    Scenario,
    load_model,
    load_municipalities,
    predict_municipality_risks,
    scenario_to_dict,
)
from services.alert_generator import (
    CHANNELS,
    LANGUAGES,
    MAYA_DISCLAIMER,
    SMS_MAX_CHARS,
    STATUS_LIVE,
    AlertResult,
    generate_alerts,
    get_llm_config,
    llm_available,
)
from services import triage_engine as triage

st.set_page_config(
    page_title="Yucatán EWRS",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_ORDER = ["Low", "Moderate", "High", "Critical"]
STATUS_COLORS = {"Operational": "#2E9E6A", "Degraded": "#E3B23C", "Offline": "#C8423B"}
BAND_COLORS = {
    "P1 – Immediate": "#C8423B",
    "P2 – Urgent": "#E07B39",
    "P3 – Priority": "#E3B23C",
    "P4 – Routine": "#2E9E6A",
}

# --------------------------------------------------------------------------- #
# Styling (neutral; works in both light and dark Streamlit themes)
# --------------------------------------------------------------------------- #
st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
.ewrs-title {font-size: 1.65rem; font-weight: 700; margin-bottom: 0;}
.ewrs-sub {opacity: .7; margin-top: .1rem; font-size: .95rem;}
.kpi {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .85rem 1rem;
      background: rgba(128,128,128,.06); height: 100%;}
.kpi .label {font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7;}
.kpi .value {font-size: 1.7rem; font-weight: 700; line-height: 1.25;}
.kpi .hint {font-size: .8rem; opacity: .75;}
.badge {display: inline-block; padding: .18rem .6rem; border-radius: 999px; font-size: .78rem;
        font-weight: 600; color: #fff; margin-right: .35rem; white-space: nowrap;}
.chip {display: inline-block; padding: .15rem .55rem; border-radius: 6px; font-size: .8rem;
       border: 1px solid rgba(128,128,128,.35); margin: 0 .3rem .3rem 0;}
.msg {white-space: pre-wrap; font-size: .9rem; line-height: 1.45; border-radius: 8px;
      padding: .75rem .85rem; background: rgba(128,128,128,.07); border: 1px solid rgba(128,128,128,.2);}
.meta {font-size: .76rem; opacity: .7; margin-top: .3rem;}
.comp {border-left: 5px solid; border-radius: 6px; padding: .55rem .8rem; margin-bottom: .5rem;
       background: rgba(128,128,128,.06);}
.comp .name {font-weight: 600;}
.comp .note {font-size: .82rem; opacity: .8;}
</style>
""",
    unsafe_allow_html=True,
)


def badge(text: str, color: str) -> str:
    """Return an HTML pill badge."""
    return f'<span class="badge" style="background:{color}">{html.escape(text)}</span>'


def kpi(col: Any, label: str, value: str, hint: str = "") -> None:
    """Render a KPI card in a column."""
    col.markdown(
        f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
        f'<div class="hint">{hint}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading risk model…")
def get_model() -> dict[str, Any]:
    """Load (or train on first run) the RandomForest artifact."""
    return load_model()


@st.cache_data(show_spinner=False)
def get_predictions(category: int, rainfall: float, surge: float, zone: str) -> pd.DataFrame:
    """Cached risk predictions for a scenario."""
    return predict_municipality_risks(Scenario(category, rainfall, surge, zone), get_model())


@st.cache_data(show_spinner=False)
def get_metadata() -> dict[str, Any]:
    """Municipality dataset metadata (sources, disclaimer)."""
    return load_municipalities()[1]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### 🌀 Hurricane Scenario Simulator")
    category = st.slider("Saffir-Simpson category", 1, 5, 4)
    rainfall = st.slider("Rainfall (mm / 24 h)", 50, 500, 300, step=10)
    surge = st.slider("Storm surge (m)", 0.0, 6.0, 3.0, step=0.1)
    zone = st.selectbox("Landfall zone", list(LANDFALL_TRACKS.keys()), index=0)

    st.divider()
    st.markdown("### ⚙️ Operations")
    has_key = llm_available()
    offline_mode = st.toggle(
        "Offline / Demo Mode",
        value=not has_key,
        help="Forces pre-approved alert templates. Guarantees the demo works without internet.",
    )
    outage = st.toggle(
        "Simulate Power & Telecom Outage",
        value=False,
        help="Shows how each component degrades when power and commercial telecom fail.",
    )
    map_renderer = st.selectbox(
        "Map renderer",
        ["Auto", "Street map (needs internet)", "Offline vector map"],
        help="Auto uses OpenStreetMap tiles when online and switches to an offline vector map otherwise.",
    )

    cfg = get_llm_config()
    online = not offline_mode and not outage
    if online and has_key:
        st.caption(f"🟢 LLM: **{cfg['provider'].title()}** · `{cfg['model']}`")
    elif not has_key:
        st.caption("⚪ No API key in `.env` → offline templates")
    else:
        st.caption("⚪ LLM disabled → offline templates")

    art = get_model()
    with st.expander("Model card"):
        st.markdown(
            f"**RandomForestRegressor** · {art['metrics']['n_train']:,} train / "
            f"{art['metrics']['n_test']:,} test synthetic rows  \n"
            f"R² = **{art['metrics']['r2']:.2f}** · MAE = **{art['metrics']['mae']:.1f}** pts"
        )
        imp = pd.Series(art["feature_importances"]).sort_values()
        st.bar_chart(imp, horizontal=True, height=260)

scenario = Scenario(category, float(rainfall), float(surge), zone)
scenario_dict = scenario_to_dict(scenario)
preds = get_predictions(category, float(rainfall), float(surge), zone)
metadata = get_metadata()

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="ewrs-title">Yucatán Hurricane Early Warning & Response System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ewrs-sub">AI-assisted hyperlocal risk, multilingual alerts and human-in-the-loop response · '
    "Prototype for PROCIVY</div>",
    unsafe_allow_html=True,
)
mode_badge = badge("Online", "#3B6EA8") if online else badge("Offline / Demo", "#6B7280")
outage_badge = badge("Power & Telecom Outage", "#C8423B") if outage else ""
st.markdown(
    f'<div style="margin-top:.5rem">{mode_badge}{outage_badge}'
    f'<span class="chip">Category {category}</span><span class="chip">{rainfall} mm/24h</span>'
    f'<span class="chip">Surge {surge:.1f} m</span><span class="chip">Landfall: {zone}</span></div>',
    unsafe_allow_html=True,
)

tab_map, tab_alerts, tab_triage, tab_check = st.tabs(
    ["🗺️ Hyperlocal Risk Map", "📣 Multilingual Alert Studio", "🚑 Post-Impact Triage", "⚖️ Critical Check"]
)


# --------------------------------------------------------------------------- #
# Map helpers
# --------------------------------------------------------------------------- #
def _hover(df: pd.DataFrame) -> pd.DataFrame:
    """Add formatted hover columns."""
    d = df.copy()
    d["Exposed"] = d["exposed_population"].map("{:,}".format)
    d["Population"] = d["population"].map("{:,}".format)
    d["size"] = 8 + d["risk_score"] / 3.5
    return d


def street_map(df: pd.DataFrame, zone: str) -> go.Figure:
    """OpenStreetMap tile map (scatter_mapbox, or scatter_map on Plotly >= 6)."""
    d = _hover(df)
    track = LANDFALL_TRACKS[zone]
    common = dict(
        lat="lat", lon="lon", color="risk_level", size="size", size_max=26,
        color_discrete_map=RISK_COLORS, category_orders={"risk_level": RISK_ORDER},
        hover_name="name", hover_data={"risk_score": True, "Exposed": True, "Population": True,
                                       "size": False, "lat": False, "lon": False, "risk_level": False},
        zoom=6.4, center={"lat": 20.95, "lon": -89.0}, height=520,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if hasattr(px, "scatter_mapbox"):
            fig = px.scatter_mapbox(d, mapbox_style="open-street-map", **common)
            fig.add_trace(go.Scattermapbox(lat=[p[0] for p in track], lon=[p[1] for p in track], mode="lines",
                                           line=dict(width=3, color="#7C3AED"), name="Storm track"))
        else:
            fig = px.scatter_map(d, map_style="open-street-map", **common)
            fig.add_trace(go.Scattermap(lat=[p[0] for p in track], lon=[p[1] for p in track], mode="lines",
                                        line=dict(width=3, color="#7C3AED"), name="Storm track"))
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(title="Risk level", orientation="h", y=-0.02))
    return fig


def vector_map(df: pd.DataFrame, zone: str) -> go.Figure:
    """Offline-safe scatter_geo map (no tile server required)."""
    d = _hover(df)
    track = LANDFALL_TRACKS[zone]
    fig = px.scatter_geo(
        d, lat="lat", lon="lon", color="risk_level", size="size", size_max=26,
        color_discrete_map=RISK_COLORS, category_orders={"risk_level": RISK_ORDER},
        hover_name="name", hover_data={"risk_score": True, "Exposed": True, "size": False,
                                       "lat": False, "lon": False, "risk_level": False},
        text="name", height=520,
    )
    fig.update_traces(textposition="top center", textfont=dict(size=11))
    fig.add_trace(go.Scattergeo(lat=[p[0] for p in track], lon=[p[1] for p in track], mode="lines",
                                line=dict(width=3, color="#7C3AED"), name="Storm track"))
    fig.update_geos(
        resolution=50, showland=True, landcolor="rgba(160,160,150,0.25)", showocean=True,
        oceancolor="rgba(70,130,180,0.18)", showcoastlines=True, coastlinecolor="rgba(120,120,120,0.8)",
        showcountries=True, countrycolor="rgba(120,120,120,0.5)", showsubunits=True,
        subunitcolor="rgba(120,120,120,0.6)", lataxis_range=[19.4, 22.4], lonaxis_range=[-91.4, -86.6],
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      legend=dict(title="Risk level", orientation="h", y=-0.02))
    return fig


def render_map(df: pd.DataFrame, zone: str, renderer: str, offline: bool) -> None:
    """Render the risk map; never crash – fall back to the vector map."""
    use_street = renderer.startswith("Street") or (renderer == "Auto" and not offline)
    if use_street:
        try:
            st.plotly_chart(street_map(df, zone), width="stretch", config={"displayModeBar": False})
            st.caption("Basemap © OpenStreetMap contributors · switch to *Offline vector map* if tiles do not load.")
            return
        except Exception as exc:  # pragma: no cover - defensive
            st.info(f"Street map unavailable ({type(exc).__name__}); showing offline vector map.")
    st.plotly_chart(vector_map(df, zone), width="stretch", config={"displayModeBar": False})
    st.caption("Offline vector map (no tile server required).")


# --------------------------------------------------------------------------- #
# Tab 1 – Hyperlocal Risk Map
# --------------------------------------------------------------------------- #
with tab_map:
    high = preds[preds["risk_level"].isin(["High", "Critical"])]
    n_crit = int((preds["risk_level"] == "Critical").sum())
    exposed = int(preds["exposed_population"].sum())
    demand = int(preds["shelter_demand"].sum())
    capacity = int(preds["shelter_capacity"].sum())
    util = demand / capacity if capacity else 0.0
    over = preds[preds["shelter_demand"] > preds["shelter_capacity"]]["name"].tolist()
    util_state = "Over capacity" if util > 1 else ("Strained" if util > 0.7 else "Within capacity")

    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "High / Critical municipalities", f"{len(high)} / {len(preds)}", f"{n_crit} critical")
    kpi(c2, "Estimated exposed population", f"{exposed:,}", "Residents facing life-threatening hazards")
    kpi(c3, "Shelter capacity status", f"{util:.0%}", f"{util_state} · {demand:,} / {capacity:,} beds"
        + (f" · ⚠️ {', '.join(over)}" if over else ""))
    top = preds.iloc[0]
    kpi(c4, "Highest-risk municipality", html.escape(top["name"]),
        f"Score {top['risk_score']:.0f}/100 · {top['risk_level']}")

    st.write("")
    left, right = st.columns([3, 2], gap="large")
    with left:
        render_map(preds, zone, map_renderer, offline=offline_mode or outage)
    with right:
        st.markdown("**Risk ranking**")
        bar = px.bar(
            preds.sort_values("risk_score"), x="risk_score", y="name", orientation="h", color="risk_level",
            color_discrete_map=RISK_COLORS, category_orders={"risk_level": RISK_ORDER},
            labels={"risk_score": "Risk score (0–100)", "name": ""}, height=470, range_x=[0, 100],
        )
        for x in (25, 50, 75):
            bar.add_vline(x=x, line_dash="dot", line_color="rgba(128,128,128,.5)")
        bar.update_layout(margin=dict(l=0, r=10, t=10, b=0), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})

    with st.expander("🔍 Top 3 ML risk drivers per municipality", expanded=False):
        table = pd.DataFrame(
            {
                "Municipality": preds["name"],
                "Risk score": preds["risk_score"],
                "Level": preds["risk_level"],
                "Distance to track (km)": preds["distance_to_track_km"].astype(int),
                "Exposed pop.": preds["exposed_population"],
                "Shelter demand / capacity": [
                    f"{d:,} / {c:,}" for d, c in zip(preds["shelter_demand"], preds["shelter_capacity"])
                ],
                **{
                    f"Driver {i + 1}": [
                        f"{f[i]['label']}  (+{f[i]['contribution']:.1f})" if len(f) > i else "" for f in preds["top_factors"]
                    ]
                    for i in range(3)
                },
            }
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={
                "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                "Exposed pop.": st.column_config.NumberColumn(format="%d"),
            },
        )
        st.caption(
            "Drivers come from a local occlusion analysis: each feature is reset to its training average and the "
            "drop in predicted risk (points, shown in brackets) measures its contribution."
        )

    st.caption(
        f"ℹ️ **Data note:** {metadata.get('disclaimer', '')} Real-world sources: "
        + "; ".join(metadata.get("reference_sources", []))
        + ". Risk scores come from a model trained on synthetic, physically plausible scenarios."
    )


# --------------------------------------------------------------------------- #
# Tab 2 – Multilingual Alert Studio
# --------------------------------------------------------------------------- #
with tab_alerts:
    left, right = st.columns([1, 3], gap="large")
    with left:
        names = preds["name"].tolist()
        muni = st.selectbox("Municipality", names, index=0, help="Sorted by current risk score.")
        profile = preds[preds["name"] == muni].iloc[0].to_dict()
        st.markdown(
            f"{badge(profile['risk_level'], RISK_COLORS[profile['risk_level']])}"
            f"<b>{profile['risk_score']:.0f}</b>/100",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"- Exposed population: **{profile['exposed_population']:,}**\n"
            f"- Maya-speaking: **{profile['maya_speaking_pct']:.0f}%**\n"
            f"- Distance to track: **{profile['distance_to_track_km']:.0f} km**\n"
            f"- Main driver: {profile['top_factors'][0]['label']}"
        )
        generate = st.button("📣 Generate Alert", type="primary", width="stretch")
        if outage and not offline_mode:
            st.caption("Telecom outage simulated → cloud LLM unreachable, offline templates will be used.")

    request_key = (muni, category, rainfall, surge, zone, online)
    if generate:
        with st.spinner("Generating multilingual alerts…"):
            st.session_state["alert_result"] = generate_alerts(profile, scenario_dict, online=online)
            st.session_state["alert_key"] = request_key

    with right:
        result: AlertResult | None = st.session_state.get("alert_result")
        if result is None:
            st.info("Select a municipality and press **Generate Alert** to draft Cell Broadcast, WhatsApp and radio messages in Spanish, Yucatec Maya and English.")
        else:
            if st.session_state.get("alert_key") != request_key:
                st.warning("Scenario or municipality changed since these alerts were generated – press **Generate Alert** to refresh.")
            src_color = "#2E9E6A" if result.status == STATUS_LIVE else "#6B7280"
            st.markdown(
                f"{badge(result.status, src_color)}{badge(result.risk_level, RISK_COLORS[result.risk_level])}"
                f"<b>{html.escape(result.municipality)}</b> · <span style='opacity:.7'>{html.escape(result.provider)}</span>",
                unsafe_allow_html=True,
            )
            if result.reason:
                st.caption(f"Fallback reason: {result.reason}")
            for w in result.warnings:
                st.caption(f"⚠️ {w}")

            lang_tabs = st.tabs([f"{name}" for name in LANGUAGES.values()])
            for lang_tab, lang in zip(lang_tabs, LANGUAGES):
                with lang_tab:
                    cols = st.columns([1, 1.25, 1.25], gap="medium")
                    for col, (ch, ch_name) in zip(cols, CHANNELS.items()):
                        text = result.alerts[lang][ch]
                        if ch == "sms":
                            meta = f"{len(text)}/{SMS_MAX_CHARS} characters"
                        elif ch == "radio":
                            words = len(text.split())
                            meta = f"{words} words · ≈{words / 2.5:.0f} s read aloud"
                        else:
                            meta = f"{len(text.split())} words"
                        col.markdown(f"**{ch_name}**")
                        col.markdown(f'<div class="msg">{html.escape(text)}</div><div class="meta">{meta}</div>',
                                     unsafe_allow_html=True)
                    if lang == "yua":
                        st.warning(f"⚠️ {MAYA_DISCLAIMER}")

    st.caption(f"⚠️ {MAYA_DISCLAIMER}")


# --------------------------------------------------------------------------- #
# Tab 3 – Post-Impact Triage (Human-in-the-Loop)
# --------------------------------------------------------------------------- #
with tab_triage:
    incidents, brigades = triage.load_incident_data()
    area_risk = dict(zip(preds["name"], preds["risk_score"]))
    queue = triage.suggest_allocation(triage.score_incidents(incidents, area_risk), brigades)
    triage.init_decisions(queue["id"].tolist(), st.session_state)
    decisions = st.session_state["triage_decisions"]
    options = triage.brigade_options(brigades)

    statuses = [decisions[i]["status"] for i in queue["id"]]
    c1, c2, c3, c4, c5 = st.columns(5)
    kpi(c1, "Open incidents", str(len(queue)))
    kpi(c2, "P1 – Immediate", str(int((queue["priority_band"] == "P1 – Immediate").sum())))
    kpi(c3, "Pending review", str(statuses.count(triage.STATUS_PENDING)))
    kpi(c4, "Approved", str(statuses.count(triage.STATUS_APPROVED)))
    kpi(c5, "Overridden", str(statuses.count(triage.STATUS_OVERRIDDEN)))
    st.write("")

    if outage:
        st.warning("📡 Outage mode: digital incident intake (apps, WhatsApp, telemetry) is down. New reports arrive via "
                   "VHF/HF radio and satellite phones and are typed in by operators – the queue may be incomplete.")

    a1, a2, _ = st.columns([1.3, 1, 3])

    def _approve_all() -> None:
        for row in queue.itertuples():
            if st.session_state["triage_decisions"][row.id]["status"] == triage.STATUS_PENDING:
                triage.record_decision(st.session_state, row.id, triage.STATUS_APPROVED, row.ai_suggestion, "Bulk approval")

    a1.button("✅ Approve all pending", on_click=_approve_all, width="stretch")
    a2.button("↺ Reset decisions", on_click=triage.reset_decisions, args=(st.session_state,), width="stretch")
    st.caption("AI suggests a brigade per incident (highest priority first, limited units). The PROCIVY operator "
               "approves or overrides every dispatch – no unit moves without human sign-off.")

    for row in queue.itertuples():
        dec = decisions[row.id]
        status_color = {"Approved": "#2E9E6A", "Overridden": "#3B6EA8"}.get(dec["status"], "#6B7280")
        with st.container(border=True):
            h1, h2 = st.columns([3, 2], gap="large")
            with h1:
                st.markdown(
                    f"{badge(row.priority_band, BAND_COLORS[row.priority_band])}"
                    f"<b>{row.priority_score:.0f}</b>/100 &nbsp;·&nbsp; <code>{row.id}</code> &nbsp;·&nbsp; "
                    f"{html.escape(row.municipality)} &nbsp;·&nbsp; <span style='opacity:.7'>{row.reported_at} · {html.escape(row.source)}</span>",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**{row.title}**  \n{row.description}")
                st.caption(
                    f"Severity {row.severity}/5 · Vulnerable people {row.vulnerable_people:,} of {row.people_affected:,} · "
                    f"Infrastructure criticality {row.infrastructure_criticality}/5 · Access: {row.access.replace('_', ' ')}"
                )
            with h2:
                st.markdown(f"🤖 **AI suggestion:** {row.ai_suggestion}")
                st.caption(row.ai_rationale)
                st.markdown(
                    f"{badge(dec['status'], status_color)}"
                    + (f"→ <b>{html.escape(dec['assigned'])}</b> <span style='opacity:.6'>({dec['time']})</span>"
                       if dec["assigned"] else ""),
                    unsafe_allow_html=True,
                )
                b1, b2 = st.columns([1, 2])
                b1.button(
                    "Approve", key=f"approve_{row.id}", type="primary", width="stretch",
                    disabled=dec["status"] != triage.STATUS_PENDING,
                    on_click=triage.record_decision,
                    args=(st.session_state, row.id, triage.STATUS_APPROVED, row.ai_suggestion, "AI suggestion accepted"),
                )
                with b2.popover("Override", width="stretch"):
                    default_idx = options.index(row.ai_suggestion) if row.ai_suggestion in options else 0
                    choice = st.selectbox("Assign instead", options, index=default_idx, key=f"ov_sel_{row.id}")
                    note = st.text_input("Reason (audit log)", key=f"ov_note_{row.id}",
                                         placeholder="e.g. local knowledge: road via Kinchil is open")
                    st.button(
                        "Confirm override", key=f"ov_btn_{row.id}",
                        on_click=triage.record_decision,
                        args=(st.session_state, row.id, triage.STATUS_OVERRIDDEN, choice, note or "Operator override"),
                    )

    with st.expander("📝 Dispatch audit log", expanded=False):
        audit = st.session_state.get("triage_audit", [])
        if audit:
            st.dataframe(pd.DataFrame(audit), hide_index=True, width="stretch")
        else:
            st.caption("No decisions recorded yet.")
        st.caption("Brigade roster: " + " · ".join(f"{b['type']} ×{b['units']}" for b in brigades))


# --------------------------------------------------------------------------- #
# Tab 4 – Critical Check (Baseline vs. AI)
# --------------------------------------------------------------------------- #
BASELINE_COMPONENTS: list[dict[str, str]] = [
    {"name": "Cell Broadcast (CBS) alerts", "normal": "Operational", "outage": "Degraded",
     "note": "No app or data plan needed; works while towers run on backup batteries (≈4–8 h)."},
    {"name": "Community radio (AM/FM, generator-backed)", "normal": "Operational", "outage": "Operational",
     "note": "Battery radios reach households in Spanish and Maya; stations run on generators."},
    {"name": "Static GIS & printed evacuation maps", "normal": "Operational", "outage": "Operational",
     "note": "Pre-printed risk atlas and routes posted at shelters and municipal halls."},
    {"name": "Drainage & pozos de absorción maintenance", "normal": "Operational", "outage": "Operational",
     "note": "Pre-season physical work – no runtime dependency on power or networks."},
    {"name": "Sirens, perifoneo & VHF/HF radio", "normal": "Operational", "outage": "Operational",
     "note": "Loudspeaker vehicles, brigades and radio nets independent of commercial telecom."},
]
AI_COMPONENTS: list[dict[str, str]] = [
    {"name": "Hyperlocal ML risk model", "normal": "Operational", "outage": "Degraded",
     "note": "Runs locally on a laptop, but live CONAGUA/SMN feeds stop → last known scenario only."},
    {"name": "LLM multilingual alert drafting", "normal": "Operational", "outage": "Degraded",
     "note": "Cloud API unreachable → automatic switch to pre-approved offline templates."},
    {"name": "WhatsApp community distribution", "normal": "Operational", "outage": "Offline",
     "note": "Requires mobile data / internet – unavailable until telecom is restored."},
    {"name": "Interactive street-map dashboard", "normal": "Operational", "outage": "Degraded",
     "note": "Map tiles unavailable → offline vector map; needs a powered device."},
    {"name": "Real-time digital incident intake", "normal": "Operational", "outage": "Offline",
     "note": "Apps, social media and IoT telemetry go dark; reports only by radio / satellite phone."},
    {"name": "AI triage & brigade suggestions", "normal": "Operational", "outage": "Degraded",
     "note": "Still computes locally but on delayed, incomplete data – human operator decides."},
]
STATUS_VALUE = {"Operational": 100, "Degraded": 50, "Offline": 0}

with tab_check:
    condition = "outage" if outage else "normal"
    if outage:
        st.error("⚡ **Power & telecom outage simulated.** Highlighted below: which AI components degrade and which "
                 "low-tech protocols keep working.")
    else:
        st.success("All systems nominal. Toggle **Simulate Power & Telecom Outage** in the sidebar to stress-test the architecture.")

    def render_components(title: str, comps: list[dict[str, str]]) -> None:
        st.markdown(f"#### {title}")
        for c in comps:
            status = c[condition]
            color = STATUS_COLORS[status]
            changed = outage and status != c["normal"]
            weight = "border-width:0 0 0 5px;" + ("box-shadow:0 0 0 2px " + color + "55;" if changed else "")
            st.markdown(
                f'<div class="comp" style="border-color:{color};{weight}">'
                f'<span class="name">{html.escape(c["name"])}</span> {badge(status, color)}'
                f'<div class="note">{html.escape(c["note"])}</div></div>',
                unsafe_allow_html=True,
            )

    col_b, col_ai = st.columns(2, gap="large")
    with col_b:
        render_components("🛡️ Baseline – simple, non-AI protocols", BASELINE_COMPONENTS)
    with col_ai:
        render_components("🤖 AI augmentation layer", AI_COMPONENTS)

    def capability(comps: list[dict[str, str]], cond: str) -> float:
        return sum(STATUS_VALUE[c[cond]] for c in comps) / len(comps)

    cap = pd.DataFrame(
        [
            {"Layer": layer, "Condition": cond_label, "Capability retained (%)": capability(comps, cond)}
            for layer, comps in (("Baseline protocols", BASELINE_COMPONENTS), ("AI layer", AI_COMPONENTS))
            for cond, cond_label in (("normal", "Normal operations"), ("outage", "Power & telecom outage"))
        ]
    )
    g1, g2 = st.columns([2, 3], gap="large")
    with g1:
        fig = px.bar(
            cap, x="Layer", y="Capability retained (%)", color="Condition", barmode="group",
            color_discrete_map={"Normal operations": "#3B6EA8", "Power & telecom outage": "#C8423B"},
            range_y=[0, 105], height=340, text_auto=".0f",
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.2, title=""),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis_title="")
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Operational = 100 · Degraded = 50 · Offline = 0 (average across components).")
    with g2:
        st.markdown("#### What each layer adds")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Warning granularity", "State/regional bulletins", "Municipality-level risk scores & drivers"),
                    ("Language reach", "Mostly Spanish", "Spanish + Yucatec Maya + English drafts in seconds"),
                    ("Message drafting time", "Manual, 30–60 min", "< 1 min (human approval still required)"),
                    ("Response prioritisation", "Ad-hoc, phone-based", "Scored queue + suggested brigades (HITL)"),
                    ("Infrastructure dependence", "Very low", "High – power, data, cloud APIs"),
                    ("Cost & maintenance", "Low, well understood", "Higher – data pipelines, model upkeep"),
                    ("Trust & explainability", "Familiar to communities", "Needs transparency, validation, training"),
                ],
                columns=["Criterion", "Baseline", "AI layer"],
            ),
            hide_index=True,
            width="stretch",
        )

    st.info(
        "**Design principle:** AI is an *augmentation layer*, not a replacement. Simple protocols – Cell Broadcast, "
        "community radio, printed maps and drainage maintenance – remain the backbone; the AI layer degrades "
        "gracefully to pre-approved templates and human decision-making when power and telecom fail."
    )
