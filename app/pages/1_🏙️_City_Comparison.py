import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="City Comparison · SkyCast AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Shared design system CSS ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg-top: #03131f;
            --bg-bottom: #0d2535;
            --panel: rgba(7, 27, 40, 0.82);
            --panel-strong: rgba(10, 36, 54, 0.96);
            --line: rgba(131, 211, 255, 0.18);
            --text: #e9f6ff;
            --muted: #8fb5c8;
            --accent: #68d7ff;
            --accent-2: #0ea5e9;
            --warm: #ffb36b;
            --success: #6ee7b7;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(104, 215, 255, 0.22), transparent 28%),
                radial-gradient(circle at 85% 10%, rgba(255, 179, 107, 0.12), transparent 22%),
                linear-gradient(180deg, var(--bg-top) 0%, var(--bg-bottom) 100%);
            color: var(--text);
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
            position: relative;
            z-index: 1;
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        [data-testid="stDecoration"] { display: none; }
        [data-testid="stHeader"] { background: transparent; }

        /* ══ Sidebar – identical to main app ══ */
        [data-testid="stSidebar"] {
            background: linear-gradient(
                180deg, #03131f 0%, #071b28 60%, #0d2535 100%
            ) !important;
            border-right: 1px solid rgba(104, 215, 255, 0.10) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.35);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0 !important;
            background: transparent !important;
        }
        /* Hide Streamlit's auto-generated duplicate page nav */
        [data-testid="stSidebarNav"]      { display: none !important; }
        [data-testid="stSidebarNavItems"] { display: none !important; }

        /* Sidebar typography tokens */
        .sidebar-logo {
            display: flex;
            align-items: center;
            gap: 0.6rem;
            padding: 0.8rem 1rem 1rem 1rem;
            border-bottom: 1px solid rgba(104, 215, 255, 0.10);
            margin-bottom: 0.8rem;
        }
        .sidebar-logo-icon { font-size: 1.5rem; }
        .sidebar-logo-text {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 700;
            color: #e9f6ff;
            letter-spacing: -0.02em;
        }
        .sidebar-logo-sub {
            font-size: 0.7rem;
            color: #8fb5c8;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .nav-section-label {
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: rgba(143, 181, 200, 0.5);
            padding: 0 1rem;
            margin: 0.5rem 0 0.3rem;
        }
        .nav-divider {
            height: 1px;
            background: rgba(104, 215, 255, 0.10);
            margin: 0.8rem 1rem;
        }

        /* ── Page header ── */
        .page-header {
            padding: 1.5rem 1.6rem 1.4rem;
            border-radius: 24px;
            background:
                linear-gradient(135deg, rgba(104, 215, 255, 0.10), rgba(14, 165, 233, 0.06)),
                rgba(4, 18, 29, 0.88);
            border: 1px solid rgba(104, 215, 255, 0.15);
            box-shadow: 0 16px 40px rgba(0,0,0,0.22);
            backdrop-filter: blur(16px);
            margin-bottom: 1.5rem;
            animation: riseIn 0.7s ease-out both;
        }

        .page-eyebrow {
            display: inline-flex;
            gap: 0.4rem;
            align-items: center;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(104, 215, 255, 0.2);
            background: rgba(104, 215, 255, 0.07);
            color: var(--accent);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.13em;
            margin-bottom: 0.6rem;
        }

        .page-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(1.7rem, 3vw, 2.4rem);
            line-height: 1.05;
            margin: 0 0 0.4rem 0;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.6;
            max-width: 68ch;
            margin: 0;
        }

        /* ── Compare badges ── */
        .compare-header {
            display: flex;
            gap: 1rem;
            align-items: center;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }

        .compare-badge {
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .compare-badge-a {
            background: rgba(104, 215, 255, 0.12);
            border: 1px solid rgba(104, 215, 255, 0.3);
            color: #68d7ff;
        }

        .compare-badge-b {
            background: rgba(255, 179, 107, 0.12);
            border: 1px solid rgba(255, 179, 107, 0.3);
            color: #ffb36b;
        }

        /* ── Stat summary row ── */
        .stat-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 1rem 0;
        }

        .stat-card {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.07);
            transition: transform 0.22s ease, border-color 0.22s ease;
            animation: riseIn 0.9s ease-out both;
        }

        .stat-card:hover {
            transform: translateY(-4px);
            border-color: rgba(104, 215, 255, 0.2);
        }

        .stat-label {
            color: var(--muted);
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .stat-value-a {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            color: #68d7ff;
            margin-top: 0.3rem;
        }

        .stat-value-b {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            color: #ffb36b;
            margin-top: 0.15rem;
        }

        .stat-delta {
            font-size: 0.78rem;
            color: #8fb5c8;
            margin-top: 0.15rem;
        }

        /* ── Section styles ── */
        .section-label {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            margin-bottom: 0.5rem;
        }

        .section-title {
            font-size: 1.8rem;
            margin-bottom: 0.4rem;
            font-family: 'Space Grotesk', sans-serif;
        }

        .section-copy {
            color: var(--muted);
            margin-bottom: 1.2rem;
            max-width: 72ch;
            line-height: 1.7;
        }

        .spacer-sm { height: 0.35rem; }

        /* ── Animations ── */
        @keyframes riseIn {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        div.stButton > button:first-child {
            width: 100%;
            min-height: 3rem;
            border-radius: 14px;
            border: 1px solid rgba(104, 215, 255, 0.2);
            background: linear-gradient(135deg, #16384a 0%, #1b5d74 100%);
            color: #e9f6ff;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 0.95rem;
            transition: transform 0.18s ease, box-shadow 0.18s ease;
        }

        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 28px rgba(27, 93, 116, 0.32);
            color: #e9f6ff;
        }

        @media (max-width: 700px) {
            .stat-row { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <style>
        .nav-item {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            padding: 0.6rem 1rem;
            border-radius: 12px;
            margin: 0.2rem 0.4rem;
            font-size: 0.88rem;
            font-family: 'IBM Plex Sans', sans-serif;
            color: rgba(143,181,200,0.85);
            text-decoration: none !important;
            transition: background 0.18s ease, color 0.18s ease;
        }
        .nav-item:hover {
            background: rgba(104,215,255,0.09);
            color: #e9f6ff;
            text-decoration: none !important;
        }
        .nav-item.active {
            background: rgba(104,215,255,0.13);
            color: #68d7ff;
            font-weight: 600;
        }
        .nav-item-icon { font-size: 1.05rem; width: 1.3rem; text-align: center; flex-shrink: 0; }
        </style>

        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🌦️</div>
            <div>
                <div class="sidebar-logo-text">SkyCast AI</div>
                <div class="sidebar-logo-sub">Weather Intelligence</div>
            </div>
        </div>

        <div class="nav-section-label">Navigation</div>

        <a class="nav-item" href="/" target="_self">
            <span class="nav-item-icon">🏠</span> Dashboard
        </a>
        <a class="nav-item active" href="/City_Comparison" target="_self">
            <span class="nav-item-icon">🏙️</span> City Comparison
        </a>
        <a class="nav-item" href="/Model_Analytics" target="_self">
            <span class="nav-item-icon">🤖</span> Model Analytics
        </a>

        <div class="nav-divider"></div>
        """,
        unsafe_allow_html=True,
    )

# ── Collapsed mini icon-rail ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .skycast-mini-rail {
        position: fixed; left: 0; top: 50%; transform: translateY(-50%);
        z-index: 9999; display: none; flex-direction: column; align-items: center;
        gap: 0.5rem; padding: 0.75rem 0.45rem;
        background: linear-gradient(180deg, #03131f 0%, #0d2535 100%);
        border-right: 1px solid rgba(104,215,255,0.14);
        border-radius: 0 14px 14px 0; box-shadow: 4px 0 20px rgba(0,0,0,0.4);
    }
    .skycast-mini-rail a {
        display: flex; align-items: center; justify-content: center;
        width: 2.4rem; height: 2.4rem; border-radius: 10px; font-size: 1.15rem;
        text-decoration: none; color: rgba(143,181,200,0.8);
        transition: background 0.18s ease, color 0.18s ease, transform 0.15s ease;
    }
    .skycast-mini-rail a:hover { background: rgba(104,215,255,0.12); color: #68d7ff; transform: scale(1.1); }
    .skycast-mini-rail a.active { background: rgba(104,215,255,0.16); color: #68d7ff; }
    .mini-rail-divider { width: 1.6rem; height: 1px; background: rgba(104,215,255,0.15); }
    </style>
    <div class="skycast-mini-rail" id="skyCastMiniRail">
        <a href="/" target="_self" title="Dashboard">🏠</a>
        <a href="/City_Comparison" target="_self" title="City Comparison" class="active">🏙️</a>
        <a href="/Model_Analytics" target="_self" title="Model Analytics">🤖</a>
    </div>
    <script>
    (function() {
        function syncRail() {
            var rail = document.getElementById('skyCastMiniRail');
            }
        }, 200);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ── Resource paths ────────────────────────────────────────────────────────────
_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODEL_PATH  = os.path.join(_ROOT, "models", "lstm_model.h5")
_SCALER_PATH = os.path.join(_ROOT, "models", "scaler.pkl")
_DATA_PATH   = os.path.join(_ROOT, "data", "indian_cities_weather.csv")

FEATURES = ["max_temp", "min_temp", "avg_temp", "humidity", "rainfall", "wind_speed", "pressure", "cloud_cover"]


@st.cache_data
def load_data():
    df = pd.read_csv(_DATA_PATH)
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "Date": "date", "City": "city",
            "Temperature_Max (°C)": "max_temp", "Temperature_Min (°C)": "min_temp",
            "Temperature_Avg (°C)": "avg_temp", "Humidity (%)": "humidity",
            "Rainfall (mm)": "rainfall", "Wind_Speed (km/h)": "wind_speed",
            "Pressure (hPa)": "pressure", "Cloud_Cover (%)": "cloud_cover",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


df = load_data()
cities = sorted(df["city"].unique())

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <div class="page-eyebrow">🏙️ Comparative Analysis</div>
        <h1 class="page-title">City-to-City Comparison</h1>
        <p class="page-subtitle">
            Compare temperature, rainfall, humidity, wind speed and pressure
            patterns across any two Indian cities using 60 days of historical data.
            Instantly see which city runs warmer, wetter, or windier.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── City selectors ────────────────────────────────────────────────────────────
sel_col1, sel_col2 = st.columns(2, gap="large")
with sel_col1:
    city_a = st.selectbox("City A", cities, index=cities.index("Hyderabad") if "Hyderabad" in cities else 0, key="cmp_a")
with sel_col2:
    default_b = (cities.index("Mumbai") if "Mumbai" in cities else 1)
    city_b = st.selectbox("City B", cities, index=default_b, key="cmp_b")

# ── Number of days slider ─────────────────────────────────────────────────────
days = st.slider("History window (days)", min_value=14, max_value=365, value=60, step=7, key="cmp_days")

if city_a == city_b:
    st.info("⚠️ Please select two **different** cities to compare.")
    st.stop()

# ── Slice data ────────────────────────────────────────────────────────────────
df_a = df[df["city"] == city_a].sort_values("date").tail(days).reset_index(drop=True)
df_b = df[df["city"] == city_b].sort_values("date").tail(days).reset_index(drop=True)

# ── Summary stat cards ─────────────────────────────────────────────────────────
delta_temp  = df_a["avg_temp"].mean()  - df_b["avg_temp"].mean()
delta_rain  = df_a["rainfall"].mean()  - df_b["rainfall"].mean()
delta_hum   = df_a["humidity"].mean()  - df_b["humidity"].mean()
delta_wind  = df_a["wind_speed"].mean() - df_b["wind_speed"].mean()

def _arrow(val):
    return "▲" if val > 0 else "▼"

def _badge(val, unit):
    sign = "+" if val > 0 else ""
    return f"{sign}{val:.1f} {unit}"

st.markdown(
    f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-label">Avg Temperature</div>
            <div class="stat-value-a">{df_a['avg_temp'].mean():.1f}°C</div>
            <div class="stat-value-b">{df_b['avg_temp'].mean():.1f}°C</div>
            <div class="stat-delta">{_arrow(delta_temp)} {abs(delta_temp):.1f}°C difference</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Daily Rainfall</div>
            <div class="stat-value-a">{df_a['rainfall'].mean():.1f} mm</div>
            <div class="stat-value-b">{df_b['rainfall'].mean():.1f} mm</div>
            <div class="stat-delta">{_arrow(delta_rain)} {abs(delta_rain):.1f} mm difference</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Humidity</div>
            <div class="stat-value-a">{df_a['humidity'].mean():.0f}%</div>
            <div class="stat-value-b">{df_b['humidity'].mean():.0f}%</div>
            <div class="stat-delta">{_arrow(delta_hum)} {abs(delta_hum):.0f}% difference</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Avg Wind Speed</div>
            <div class="stat-value-a">{df_a['wind_speed'].mean():.1f} km/h</div>
            <div class="stat-value-b">{df_b['wind_speed'].mean():.1f} km/h</div>
            <div class="stat-delta">{_arrow(delta_wind)} {abs(delta_wind):.1f} km/h difference</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Summary sentence ──────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="compare-header">
        <span class="compare-badge compare-badge-a">{city_a}</span>
        <span style="color:#8fb5c8;font-size:0.88rem;">
            is <strong style="color:#e9f6ff">{abs(delta_temp):.1f}°C {'warmer' if delta_temp > 0 else 'cooler'}</strong> and
            <strong style="color:#e9f6ff">{abs(delta_rain):.1f} mm {'wetter' if delta_rain > 0 else 'drier'}</strong>/day than
        </span>
        <span class="compare-badge compare-badge-b">{city_b}</span>
        <span style="color:#8fb5c8;font-size:0.84rem;">over the last {days} days</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_temp, tab_rain, tab_humidity, tab_wind, tab_pressure = st.tabs(
    ["🌡️ Temperature", "🌧️ Rainfall", "💧 Humidity", "💨 Wind Speed", "🔵 Pressure"]
)

_layout_defaults = dict(
    height=400,
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#d7edf8", family="IBM Plex Sans"),
    margin=dict(l=10, r=10, t=50, b=60),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="top", y=-0.15, x=0),
    title_font=dict(size=16, family="Space Grotesk"),
)

def apply_axes(fig):
    fig.update_xaxes(showgrid=False, zeroline=False, title="")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(143,181,200,0.12)", zeroline=False, title="")
    return fig


with tab_temp:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_a["date"], y=df_a["max_temp"], name=f"{city_a} Max",
        line=dict(color="#68d7ff", width=2), opacity=0.75,
        hovertemplate=f"{city_a} Max: %{{y:.1f}}°C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_a["date"], y=df_a["avg_temp"], name=f"{city_a} Avg",
        line=dict(color="#68d7ff", width=3),
        hovertemplate=f"{city_a} Avg: %{{y:.1f}}°C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_b["date"], y=df_b["max_temp"], name=f"{city_b} Max",
        line=dict(color="#ffb36b", width=2), opacity=0.75,
        hovertemplate=f"{city_b} Max: %{{y:.1f}}°C<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=df_b["date"], y=df_b["avg_temp"], name=f"{city_b} Avg",
        line=dict(color="#ffb36b", width=3),
        hovertemplate=f"{city_b} Avg: %{{y:.1f}}°C<extra></extra>",
    ))
    fig.update_layout(title="Temperature Comparison (Max & Average) — °C", **_layout_defaults)
    st.plotly_chart(apply_axes(fig), use_container_width=True)

    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

    # Min temp band
    band_fig = go.Figure()
    band_fig.add_trace(go.Scatter(
        x=df_a["date"], y=df_a["min_temp"], name=f"{city_a} Min",
        mode="lines", fill="tozeroy",
        line=dict(color="#68d7ff", width=2),
        fillcolor="rgba(104,215,255,0.08)",
        hovertemplate=f"{city_a} Min: %{{y:.1f}}°C<extra></extra>",
    ))
    band_fig.add_trace(go.Scatter(
        x=df_b["date"], y=df_b["min_temp"], name=f"{city_b} Min",
        mode="lines", fill="tozeroy",
        line=dict(color="#ffb36b", width=2),
        fillcolor="rgba(255,179,107,0.08)",
        hovertemplate=f"{city_b} Min: %{{y:.1f}}°C<extra></extra>",
    ))
    band_fig.update_layout(title="Minimum Temperature — °C", **_layout_defaults)
    st.plotly_chart(apply_axes(band_fig), use_container_width=True)


with tab_rain:
    rain_fig = go.Figure()
    rain_fig.add_trace(go.Bar(
        x=df_a["date"], y=df_a["rainfall"],
        name=city_a, marker_color="rgba(104,215,255,0.70)",
        hovertemplate=f"{city_a}: %{{y:.1f}} mm<extra></extra>",
    ))
    rain_fig.add_trace(go.Bar(
        x=df_b["date"], y=df_b["rainfall"],
        name=city_b, marker_color="rgba(255,179,107,0.70)",
        hovertemplate=f"{city_b}: %{{y:.1f}} mm<extra></extra>",
    ))
    rain_fig.update_layout(barmode="overlay", title="Daily Rainfall (mm)", **_layout_defaults)
    st.plotly_chart(apply_axes(rain_fig), use_container_width=True)

    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

    # Cumulative rainfall
    cum_fig = go.Figure()
    cum_fig.add_trace(go.Scatter(
        x=df_a["date"], y=df_a["rainfall"].cumsum(),
        name=city_a, line=dict(color="#68d7ff", width=2.5),
        fill="tozeroy", fillcolor="rgba(104,215,255,0.07)",
        hovertemplate=f"{city_a} cumulative: %{{y:.1f}} mm<extra></extra>",
    ))
    cum_fig.add_trace(go.Scatter(
        x=df_b["date"], y=df_b["rainfall"].cumsum(),
        name=city_b, line=dict(color="#ffb36b", width=2.5),
        fill="tozeroy", fillcolor="rgba(255,179,107,0.07)",
        hovertemplate=f"{city_b} cumulative: %{{y:.1f}} mm<extra></extra>",
    ))
    cum_fig.update_layout(title="Cumulative Rainfall (mm)", **_layout_defaults)
    st.plotly_chart(apply_axes(cum_fig), use_container_width=True)


with tab_humidity:
    hum_fig = go.Figure()
    hum_fig.add_trace(go.Scatter(
        x=df_a["date"], y=df_a["humidity"], name=city_a,
        line=dict(color="#68d7ff", width=2.5), fill="tozeroy",
        fillcolor="rgba(104,215,255,0.08)",
        hovertemplate=f"{city_a}: %{{y:.0f}}%<extra></extra>",
    ))
    hum_fig.add_trace(go.Scatter(
        x=df_b["date"], y=df_b["humidity"], name=city_b,
        line=dict(color="#ffb36b", width=2.5),
        hovertemplate=f"{city_b}: %{{y:.0f}}%<extra></extra>",
    ))
    hum_fig.update_layout(title="Relative Humidity (%)", **_layout_defaults)
    st.plotly_chart(apply_axes(hum_fig), use_container_width=True)


with tab_wind:
    wind_fig = go.Figure()
    wind_fig.add_trace(go.Scatter(
        x=df_a["date"], y=df_a["wind_speed"], name=city_a,
        line=dict(color="#68d7ff", width=2.5),
        hovertemplate=f"{city_a}: %{{y:.1f}} km/h<extra></extra>",
    ))
    wind_fig.add_trace(go.Scatter(
        x=df_b["date"], y=df_b["wind_speed"], name=city_b,
        line=dict(color="#ffb36b", width=2.5),
        hovertemplate=f"{city_b}: %{{y:.1f}} km/h<extra></extra>",
    ))
    wind_fig.update_layout(title="Wind Speed (km/h)", **_layout_defaults)
    st.plotly_chart(apply_axes(wind_fig), use_container_width=True)


with tab_pressure:
    pres_fig = go.Figure()
    pres_fig.add_trace(go.Scatter(
        x=df_a["date"], y=df_a["pressure"], name=city_a,
        line=dict(color="#68d7ff", width=2.5),
        hovertemplate=f"{city_a}: %{{y:.0f}} hPa<extra></extra>",
    ))
    pres_fig.add_trace(go.Scatter(
        x=df_b["date"], y=df_b["pressure"], name=city_b,
        line=dict(color="#ffb36b", width=2.5),
        hovertemplate=f"{city_b}: %{{y:.0f}} hPa<extra></extra>",
    ))
    pres_fig.update_layout(title="Atmospheric Pressure (hPa)", **_layout_defaults)
    st.plotly_chart(apply_axes(pres_fig), use_container_width=True)

# ── Correlation analysis ───────────────────────────────────────────────────────
st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-label">Feature Correlation</div>'
    '<div class="section-title" style="font-size:1.5rem;">Scatter Analysis</div>'
    '<div class="section-copy">Explore the relationship between any two weather variables for each city.</div>',
    unsafe_allow_html=True,
)

scatter_col1, scatter_col2 = st.columns(2, gap="large")
with scatter_col1:
    x_feat = st.selectbox("X axis", FEATURES, index=2, key="sx")
with scatter_col2:
    y_feat = st.selectbox("Y axis", FEATURES, index=4, key="sy")

scatter_fig = go.Figure()
scatter_fig.add_trace(go.Scatter(
    x=df_a[x_feat], y=df_a[y_feat], mode="markers",
    name=city_a, marker=dict(color="rgba(104,215,255,0.65)", size=6),
    hovertemplate=f"{city_a} — {x_feat}: %{{x:.1f}}, {y_feat}: %{{y:.1f}}<extra></extra>",
))
scatter_fig.add_trace(go.Scatter(
    x=df_b[x_feat], y=df_b[y_feat], mode="markers",
    name=city_b, marker=dict(color="rgba(255,179,107,0.65)", size=6),
    hovertemplate=f"{city_b} — {x_feat}: %{{x:.1f}}, {y_feat}: %{{y:.1f}}<extra></extra>",
))
scatter_fig.update_layout(
    title=f"{x_feat.replace('_',' ').title()} vs {y_feat.replace('_',' ').title()}",
    **_layout_defaults,
)
scatter_fig.update_xaxes(showgrid=True, gridcolor="rgba(143,181,200,0.10)", zeroline=False, title=x_feat.replace("_", " ").title())
scatter_fig.update_yaxes(showgrid=True, gridcolor="rgba(143,181,200,0.10)", zeroline=False, title=y_feat.replace("_", " ").title())
st.plotly_chart(scatter_fig, use_container_width=True)
