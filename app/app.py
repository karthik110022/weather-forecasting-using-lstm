import os
import sys
import warnings
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from streamlit_js_eval import get_geolocation

# Ensure the root directory is in the Python path so Streamlit can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="SkyCast AI",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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

        .stApp::before,
        .stApp::after {
            content: "";
            position: fixed;
            inset: auto;
            border-radius: 999px;
            pointer-events: none;
            filter: blur(8px);
            opacity: 0.5;
            z-index: 0;
            animation: driftGlow 18s ease-in-out infinite;
        }

        .stApp::before {
            top: 8%;
            left: 4%;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(104, 215, 255, 0.22), transparent 68%);
        }

        .stApp::after {
            right: 6%;
            bottom: 12%;
            width: 280px;
            height: 280px;
            background: radial-gradient(circle, rgba(255, 179, 107, 0.14), transparent 70%);
            animation-delay: -6s;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1320px;
            position: relative;
            z-index: 1;
        }

        .block-container > div[data-testid="stVerticalBlock"] {
            gap: 1.15rem;
        }

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
        }

        h1, h2, h3, h4 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--text);
            letter-spacing: -0.02em;
        }

        p, div, span {
            box-sizing: border-box;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stToolbar"] {
            right: 1rem;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(104, 215, 255, 0.14), rgba(14, 165, 233, 0.08)),
                rgba(4, 20, 30, 0.85);
            border: 1px solid var(--line);
            border-radius: 28px;
            padding: 1.1rem 1.2rem 0.95rem 1.2rem;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(18px);
            animation: riseIn 0.8s ease-out both;
            margin-bottom: 0.35rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) minmax(320px, 0.9fr);
            gap: 1rem;
            align-items: stretch;
            position: relative;
            z-index: 1;
        }

        .hero-copy-wrap {
            position: relative;
            z-index: 1;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            inset: auto -5% -35% 35%;
            height: 260px;
            background: radial-gradient(circle, rgba(104, 215, 255, 0.22), transparent 65%);
            pointer-events: none;
            animation: pulseGlow 7s ease-in-out infinite;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent 20%, rgba(255, 255, 255, 0.08) 42%, transparent 56%);
            transform: translateX(-140%);
            animation: sweep 9s linear infinite;
        }

        .eyebrow {
            display: inline-flex;
            gap: 0.45rem;
            align-items: center;
            padding: 0.45rem 0.8rem;
            border-radius: 999px;
            border: 1px solid rgba(104, 215, 255, 0.2);
            background: rgba(104, 215, 255, 0.08);
            color: var(--accent);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            margin-bottom: 0.7rem;
        }

        .hero-title {
            font-size: clamp(1.85rem, 3vw, 2.45rem);
            line-height: 1.02;
            margin: 0;
            max-width: none;
            white-space: nowrap;
        }

        .hero-copy {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.42;
            max-width: 34ch;
            margin-top: 0.35rem;
            margin-bottom: 0;
        }

        .hero-aside {
            position: relative;
            min-height: 100%;
            padding: 0.9rem;
            border-radius: 20px;
            background:
                linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.03)),
                rgba(4, 18, 29, 0.72);
            border: 1px solid rgba(104, 215, 255, 0.14);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
            overflow: hidden;
        }

        .hero-aside::before {
            content: "";
            position: absolute;
            inset: 14% 12% auto auto;
            width: 170px;
            height: 170px;
            border-radius: 50%;
            border: 1px dashed rgba(104, 215, 255, 0.18);
            opacity: 0.6;
        }

        .hero-aside::after {
            content: "";
            position: absolute;
            inset: 19% 17% auto auto;
            width: 110px;
            height: 110px;
            border-radius: 50%;
            border: 1px solid rgba(104, 215, 255, 0.14);
            box-shadow: 0 0 28px rgba(104, 215, 255, 0.08);
        }

        .hero-aside-top {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: flex-start;
            position: relative;
            z-index: 1;
        }

        .hero-aside-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            margin: 0;
        }

        .hero-aside-copy {
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 0.2rem;
            margin-bottom: 0;
            max-width: 28ch;
        }

        .signal-pill {
            padding: 0.45rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(110, 231, 183, 0.18);
            background: rgba(110, 231, 183, 0.08);
            color: var(--success);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            white-space: nowrap;
        }

        .hero-action-row {
            position: relative;
            z-index: 1;
            margin-top: 0.55rem;
            display: flex;
            gap: 0.6rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .hero-action-row .stButton {
            flex: 0 0 auto;
            min-width: 190px;
        }

        .hero-temp {
            position: relative;
            z-index: 1;
            margin-top: 0.55rem;
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2.4rem, 4vw, 3.6rem);
            line-height: 0.92;
        }

        .hero-temp-sub {
            color: #cde3ef;
            font-size: 0.88rem;
            margin-top: 0.25rem;
            line-height: 1.5;
            max-width: 42ch;
        }

        .hero-stats {
            position: relative;
            z-index: 1;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.55rem;
        }

        .hero-stat {
            padding: 0.75rem 0.85rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .hero-stat-label {
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-stat-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.02rem;
            margin-top: 0.25rem;
        }

        .hero-radar {
            position: relative;
            z-index: 1;
            margin-top: 0.55rem;
            padding-top: 0.55rem;
            border-top: 1px solid rgba(255, 255, 255, 0.07);
        }

        .hero-radar-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
        }

        .hero-radar-item {
            padding: 0.65rem 0.55rem;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.035);
            text-align: center;
        }

        .hero-radar-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 0.95rem;
        }

        .hero-radar-label {
            color: var(--muted);
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-top: 0.2rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.25rem;
            box-shadow: 0 14px 38px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(16px);
            transition: transform 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease;
            animation: riseIn 0.8s ease-out both;
        }

        .panel > *:first-child {
            margin-top: 0;
        }

        .panel > *:last-child {
            margin-bottom: 0;
        }

        .panel:hover {
            transform: translateY(-4px);
            border-color: rgba(104, 215, 255, 0.22);
            box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28);
        }

        .panel-strong {
            background: var(--panel-strong);
        }

        .panel-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1rem;
            margin-bottom: 0.2rem;
        }

        .panel-kicker {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }

        .status-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .status-card {
            padding: 1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.06);
            transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
            animation: riseIn 0.9s ease-out both;
        }

        .status-card:hover {
            transform: translateY(-5px);
            border-color: rgba(104, 215, 255, 0.22);
            background: rgba(255, 255, 255, 0.05);
        }

        .status-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .status-value {
            margin-top: 0.35rem;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.65rem;
        }

        .status-note {
            color: #b9d6e6;
            font-size: 0.88rem;
            margin-top: 0.35rem;
        }

        .forecast-hero {
            padding: 1.35rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at top right, rgba(110, 231, 183, 0.18), transparent 32%),
                linear-gradient(145deg, rgba(14, 165, 233, 0.18), rgba(255, 179, 107, 0.08)),
                rgba(5, 22, 33, 0.95);
            border: 1px solid rgba(110, 231, 183, 0.22);
            min-height: 100%;
            position: relative;
            overflow: hidden;
            animation: riseIn 1s ease-out both;
        }

        .forecast-hero::after {
            content: "";
            position: absolute;
            inset: auto -15% -25% auto;
            width: 220px;
            height: 220px;
            background: radial-gradient(circle, rgba(110, 231, 183, 0.16), transparent 68%);
            animation: pulseGlow 8s ease-in-out infinite;
        }

        .forecast-tag {
            color: var(--success);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }

        .forecast-main {
            font-family: 'Space Grotesk', sans-serif;
            font-size: clamp(2.2rem, 4vw, 3.6rem);
            line-height: 0.95;
            margin: 0.5rem 0 0.75rem 0;
        }

        .forecast-copy {
            color: #d7edf8;
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 1rem;
        }

        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.8rem;
        }

        .mini-card {
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.045);
            border: 1px solid rgba(255, 255, 255, 0.06);
            transition: transform 0.22s ease, border-color 0.22s ease;
        }

        .mini-card:hover {
            transform: translateY(-4px);
            border-color: rgba(255, 255, 255, 0.14);
        }

        .mini-label {
            color: var(--muted);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .mini-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            margin-top: 0.25rem;
        }

        .section-label {
            color: var(--accent);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            margin-bottom: 0.5rem;
        }

        .section-title {
            font-size: 2rem;
            margin-bottom: 0.4rem;
        }

        .section-copy {
            color: var(--muted);
            margin-bottom: 1.2rem;
            max-width: 72ch;
            line-height: 1.7;
        }

        .section-stack {
            margin-top: 0.35rem;
            margin-bottom: 0.2rem;
        }

        .insight-card {
            padding: 1rem 1.1rem;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.035);
            border: 1px solid rgba(255, 255, 255, 0.06);
            min-height: 100%;
            transition: transform 0.22s ease, border-color 0.22s ease, background 0.22s ease;
            animation: riseIn 1s ease-out both;
        }

        .insight-card:hover {
            transform: translateY(-5px);
            border-color: rgba(104, 215, 255, 0.22);
            background: rgba(255, 255, 255, 0.05);
        }

        .insight-title {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .insight-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.85rem;
            margin-top: 0.45rem;
        }

        .insight-copy {
            color: #cde3ef;
            font-size: 0.92rem;
            line-height: 1.6;
            margin-top: 0.45rem;
        }

        .stSelectbox label,
        .stCheckbox label {
            color: var(--muted) !important;
            font-size: 0.88rem !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        [data-baseweb="select"] > div,
        .stCheckbox > label,
        .st-emotion-cache-16txtl3,
        .st-emotion-cache-1r6slb0 {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
        }

        div.stButton > button:first-child {
            width: 100%;
            min-height: 3.2rem;
            border-radius: 18px;
            border: 1px solid rgba(104, 215, 255, 0.2);
            background: linear-gradient(135deg, #16384a 0%, #1b5d74 100%);
            color: #e9f6ff;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1rem;
            box-shadow: 0 10px 24px rgba(27, 93, 116, 0.26);
            transition: transform 0.18s ease, box-shadow 0.18s ease;
            position: relative;
            overflow: hidden;
        }

        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 16px 30px rgba(27, 93, 116, 0.34);
            color: #e9f6ff;
        }

        div.stButton > button:first-child::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent 20%, rgba(255, 255, 255, 0.35) 45%, transparent 62%);
            transform: translateX(-150%);
            animation: sweep 5.5s linear infinite;
        }

        [data-testid="stMetric"] {
            background: transparent;
            border: none;
            padding: 0;
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted) !important;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.76rem !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.8rem !important;
        }

        [data-testid="stMetricDelta"] {
            color: #cde3ef !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 14px;
            padding: 0.4rem 0.9rem;
            color: var(--muted);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(104, 215, 255, 0.12);
            color: var(--text);
        }

        .footnote {
            color: var(--muted);
            font-size: 0.88rem;
            line-height: 1.6;
            margin-top: 0.75rem;
            margin-bottom: 0;
        }

        .spacer-xs {
            height: 0.15rem;
        }

        .spacer-sm {
            height: 0.35rem;
        }

        .delay-1 { animation-delay: 0.08s; }
        .delay-2 { animation-delay: 0.16s; }
        .delay-3 { animation-delay: 0.24s; }
        .delay-4 { animation-delay: 0.32s; }
        .delay-5 { animation-delay: 0.4s; }

        @keyframes riseIn {
            from {
                opacity: 0;
                transform: translateY(18px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes driftGlow {
            0%, 100% {
                transform: translate3d(0, 0, 0) scale(1);
            }
            50% {
                transform: translate3d(18px, -14px, 0) scale(1.08);
            }
        }

        @keyframes pulseGlow {
            0%, 100% {
                opacity: 0.4;
                transform: scale(1);
            }
            50% {
                opacity: 0.75;
                transform: scale(1.08);
            }
        }

        @keyframes sweep {
            0% {
                transform: translateX(-150%);
            }
            100% {
                transform: translateX(180%);
            }
        }

        @media (max-width: 1100px) {
            .hero-grid {
                grid-template-columns: minmax(0, 1fr);
            }

            .status-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 640px) {
            .hero-shell {
                padding: 1.4rem;
            }

            .status-strip,
            .mini-grid,
            .hero-stats,
            .hero-radar-grid {
                grid-template-columns: minmax(0, 1fr);
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/lstm_model.h5", compile=False)


@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("data/indian_cities_weather.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(
        columns={
            "Date": "date",
            "City": "city",
            "Temperature_Max (°C)": "max_temp",
            "Temperature_Min (°C)": "min_temp",
            "Temperature_Avg (°C)": "avg_temp",
            "Humidity (%)": "humidity",
            "Rainfall (mm)": "rainfall",
            "Wind_Speed (km/h)": "wind_speed",
            "Pressure (hPa)": "pressure",
            "Cloud_Cover (%)": "cloud_cover",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    return df


def infer_conditions(avg_temp, rainfall, humidity, cloud_cover):
    if rainfall >= 12:
        return "Monsoon Pulse", "Heavy atmospheric moisture with strong precipitation signals."
    if rainfall >= 3:
        return "Passing Showers", "A wetter setup with moderate rain potential in the next cycle."
    if cloud_cover >= 70:
        return "Cloud Layer", "Dense cloud presence is likely to keep the skyline muted."
    if avg_temp >= 32 and humidity <= 45:
        return "Dry Heat", "Warm daytime conditions with lighter moisture in the air."
    if humidity >= 75:
        return "Humid Air", "Sticky air mass holding in warmth and moisture."
    return "Stable Window", "Balanced conditions with no strong weather disruption indicated."


def build_series_chart(data, y, title, color, fill=False):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["date"],
            y=data[y],
            mode="lines",
            line=dict(color=color, width=3),
            fill="tozeroy" if fill else None,
            fillcolor="rgba(104, 215, 255, 0.10)" if fill else None,
            hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d7edf8", family="IBM Plex Sans"),
        margin=dict(l=10, r=10, t=42, b=10),
        hovermode="x unified",
        title_font=dict(size=18, family="Space Grotesk"),
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        title="",
        tickfont=dict(color="#8fb5c8"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(143, 181, 200, 0.12)",
        zeroline=False,
        title="",
        tickfont=dict(color="#8fb5c8"),
    )
    return fig


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    )
    return 2 * earth_radius_km * asin(sqrt(a))


def find_nearest_supported_city(latitude, longitude):
    from src.api import CITY_COORDINATES

    ranked = sorted(
        (
            (
                city_name,
                haversine_km(latitude, longitude, coords["lat"], coords["lon"]),
            )
            for city_name, coords in CITY_COORDINATES.items()
        ),
        key=lambda item: item[1],
    )
    return ranked[0]


model = load_model()
scaler = load_scaler()
df = load_data()

cities = sorted(df["city"].unique())
features = [
    "max_temp",
    "min_temp",
    "avg_temp",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure",
    "cloud_cover",
]
SEQ_LEN = 60

default_city = "Hyderabad" if "Hyderabad" in cities else cities[0]
st.session_state.setdefault("target_city", default_city)
st.session_state.setdefault("location_mode", "browser")
st.session_state.setdefault("prediction_city", default_city)
st.session_state.setdefault("prediction_mode", "preset")
st.session_state.setdefault("generate_forecast", False)
st.session_state.setdefault("request_browser_location", False)
st.session_state.setdefault("auto_location_attempted", False)
st.session_state.setdefault("geo_request_key", 0)

if (
    not st.session_state.get("auto_location_attempted")
    and not st.session_state.get("browser_location")
):
    st.session_state["auto_location_attempted"] = True
    st.session_state["request_browser_location"] = True
    st.session_state["location_mode"] = "browser"

if st.session_state.get("request_browser_location"):
    geo_result = get_geolocation(component_key=f"live_geo_{st.session_state['geo_request_key']}")
    if geo_result:
        st.session_state["request_browser_location"] = False
        coords = geo_result.get("coords")
        geo_error = geo_result.get("error")

        if coords:
            browser_lat = float(coords["latitude"])
            browser_lon = float(coords["longitude"])
            nearest_city, nearest_distance_km = find_nearest_supported_city(browser_lat, browser_lon)
            from src.api import reverse_geocode_location

            try:
                resolved_browser_label = reverse_geocode_location(browser_lat, browser_lon)
            except Exception:
                resolved_browser_label = f"Current location ({browser_lat:.2f}, {browser_lon:.2f})"

            st.session_state["browser_location"] = {
                "label": resolved_browser_label,
                "latitude": browser_lat,
                "longitude": browser_lon,
                "nearest_city": nearest_city,
                "nearest_distance_km": nearest_distance_km,
            }
            st.session_state["location_mode"] = "browser"
            st.rerun()
        elif geo_error:
            st.session_state["geo_error_message"] = geo_error.get("message", "Location permission denied.")

from src.api import get_live_weather_data

selected_mode = st.session_state.get("location_mode", "preset")
prediction_mode = st.session_state.get("prediction_mode", "preset")
use_live_data = st.session_state.get("use_live_weather", True)
preset_city = st.session_state.get("target_city", default_city)
prediction_city = st.session_state.get("prediction_city", default_city)
browser_location = st.session_state.get("browser_location")

live_label = preset_city
live_coords = None
live_fallback_city = preset_city
live_location_status = "Using fallback city"

if browser_location:
    live_label = browser_location["label"]
    live_coords = (browser_location["latitude"], browser_location["longitude"])
    live_fallback_city = browser_location["nearest_city"]
    live_location_status = (
        f"Live location detected from browser coordinates "
        f"({browser_location['latitude']:.4f}, {browser_location['longitude']:.4f}) "
        f"• nearest supported city: {browser_location['nearest_city']}"
    )
else:
    selected_mode = "preset"
    live_label = preset_city
    live_location_status = f"Live location unavailable • using fallback city: {preset_city}"

prediction_label = prediction_city
prediction_coords = None
prediction_fallback_city = prediction_city

if prediction_mode == "browser" and browser_location:
    prediction_label = browser_location["label"]
    prediction_coords = (browser_location["latitude"], browser_location["longitude"])
    prediction_fallback_city = browser_location["nearest_city"]
else:
    prediction_mode = "preset"
    prediction_label = prediction_city

live_city_df = df[df["city"] == live_fallback_city].sort_values("date")
prediction_city_df = df[df["city"] == prediction_fallback_city].sort_values("date")

if use_live_data:
    try:
        with st.spinner("Syncing live weather history..."):
            if live_coords:
                live_latest = get_live_weather_data(
                    days=SEQ_LEN,
                    latitude=live_coords[0],
                    longitude=live_coords[1],
                )
            else:
                live_latest = get_live_weather_data(live_label, days=SEQ_LEN)
        data_mode = "Live Open-Meteo"
        success_load = True
        load_issue = None
    except Exception as exc:
        live_latest = live_city_df[["date"] + features].tail(SEQ_LEN)
        data_mode = "Dataset fallback"
        success_load = True
        load_issue = str(exc)
else:
    live_latest = live_city_df[["date"] + features].tail(SEQ_LEN)
    data_mode = "Local dataset"
    success_load = True
    load_issue = None

if use_live_data:
    try:
        if prediction_coords:
            latest = get_live_weather_data(
                days=SEQ_LEN,
                latitude=prediction_coords[0],
                longitude=prediction_coords[1],
            )
        else:
            latest = get_live_weather_data(prediction_label, days=SEQ_LEN)
    except Exception:
        latest = prediction_city_df[["date"] + features].tail(SEQ_LEN)
else:
    latest = prediction_city_df[["date"] + features].tail(SEQ_LEN)

live_today = datetime.now().strftime("%B %d, %Y")
latest_data_date = live_latest["date"].iloc[-1].strftime("%B %d, %Y")

live_row = live_latest.iloc[-1]
avg_temp_now = float(live_row["avg_temp"])
rainfall_now = float(live_row["rainfall"])
humidity_now = float(live_row["humidity"])
wind_now = float(live_row["wind_speed"])
pressure_now = float(live_row["pressure"])
cloud_now = float(live_row["cloud_cover"])
condition_name, condition_copy = infer_conditions(avg_temp_now, rainfall_now, humidity_now, cloud_now)

forecast_ready = False
prediction_current_avg = float(latest.iloc[-1]["avg_temp"])
avg_temp_pred = prediction_current_avg + (latest["avg_temp"].iloc[-1] - latest["avg_temp"].iloc[-7]) / 7
rainfall_pred = max(latest["rainfall"].tail(7).mean(), 0.0)

if st.session_state.get("generate_forecast", False) and success_load:
    with st.spinner("Running neural sequence analysis..."):
        latest_scaled = scaler.transform(latest[features])
        X_input = np.expand_dims(latest_scaled, axis=0)
        pred_tensor = model(X_input, training=False)
        pred = pred_tensor.numpy()

        dummy = np.zeros((1, len(features)))
        dummy[0, 2] = pred[0][0]
        dummy[0, 4] = pred[0][1]
        real_pred = scaler.inverse_transform(pd.DataFrame(dummy, columns=features))

        avg_temp_pred = float(real_pred[0, 2])
        rainfall_pred = max(float(real_pred[0, 4]), 0.0)
        forecast_ready = True

temp_delta_pred = avg_temp_pred - prediction_current_avg
recent_temp_mean = latest["avg_temp"].tail(7).mean()
recent_rain_mean = latest["rainfall"].tail(7).mean()
recent_humidity_mean = latest["humidity"].tail(7).mean()
recent_wind_mean = latest["wind_speed"].tail(7).mean()
trend_delta = latest["avg_temp"].iloc[-1] - latest["avg_temp"].iloc[-7]
temp_range = latest["max_temp"].max() - latest["min_temp"].min()
rainy_days = int((latest["rainfall"] > 0).sum())

st.markdown(
    f"""
    <section class="hero-shell">
        <div class="hero-grid">
            <div class="hero-copy-wrap">
                <div class="eyebrow">Geographic Forecast Console</div>
                <h1 class="hero-title">Tomorrow's Weather Forecast</h1>
                <p class="hero-copy">
                    This project uses live Open-Meteo weather history and an LSTM model to estimate tomorrow's temperature for the selected location. It reads recent atmospheric patterns such as temperature, rainfall, humidity, wind, pressure, and cloud cover to give a simple next-day forecast view. The dashboard is designed to help users quickly compare today's observed weather with the model's prediction for the next day.
                </p>
            </div>
            <div class="hero-aside delay-2">
                <div class="hero-aside-top">
                    <div>
                        <p class="hero-aside-title">{live_label} weather briefing</p>
                        <div class="hero-aside-copy">Live weather snapshot for now, with tomorrow's prediction shown separately for the selected city.</div>
                    </div>
                    <div class="signal-pill">{data_mode}</div>
                </div>
                <div class="hero-temp">{avg_temp_now:.1f}°C</div>
                <div class="hero-temp-sub">Today: {live_today} • Live data date: {latest_data_date}</div>
                <div class="hero-stats">
                    <div class="hero-stat">
                        <div class="hero-stat-label">Condition</div>
                        <div class="hero-stat-value">{condition_name}</div>
                    </div>
                    <div class="hero-stat">
                        <div class="hero-stat-label">Today's Rainfall</div>
                        <div class="hero-stat-value">{rainfall_now:.1f} mm</div>
                    </div>
                    <div class="hero-stat">
                        <div class="hero-stat-label">Today's Humidity</div>
                        <div class="hero-stat-value">{humidity_now:.0f}%</div>
                    </div>
                </div>
                <div class="hero-radar">
                    <div class="hero-radar-grid">
                        <div class="hero-radar-item">
                            <div class="hero-radar-value">{wind_now:.1f} km/h</div>
                            <div class="hero-radar-label">Wind</div>
                        </div>
                        <div class="hero-radar-item">
                            <div class="hero-radar-value">{pressure_now:.0f} hPa</div>
                            <div class="hero-radar-label">Pressure</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="hero-action-row">', unsafe_allow_html=True)
use_live_location_btn = st.button("Use Live Location", use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

if use_live_location_btn:
    st.session_state["location_mode"] = "browser"
    st.session_state["request_browser_location"] = True
    st.session_state["geo_request_key"] += 1
    st.rerun()

st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

control_col, prediction_col = st.columns([1, 1], gap="large")

with control_col:
    st.markdown('<div class="panel panel-strong delay-1">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Mission Control</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="panel-kicker">Choose a location from the dropdown and get the predicted weather for tomorrow.</div>',
        unsafe_allow_html=True,
    )

    location_options = []
    location_labels = {}

    if browser_location:
        browser_label = f"Current Location (near {browser_location['nearest_city']})"
        location_options.append(browser_label)
        location_labels[browser_label] = ("browser", browser_label)

    for city_name in cities:
        location_options.append(city_name)
        location_labels[city_name] = ("preset", city_name)

    default_option = browser_label if browser_location and prediction_mode == "browser" else prediction_city
    if default_option not in location_options:
        default_option = location_options[0]

    selected_location_option = st.selectbox(
        "Choose Location",
        location_options,
        index=location_options.index(default_option),
    )

    generate_btn = st.button("Get Weather Prediction", type="primary", use_container_width=True)

    st.checkbox(
        "Use live Open-Meteo history",
        value=use_live_data,
        help="Enabled: pulls the latest 60-day history from Open-Meteo. Disabled: uses the dataset bundled with this project.",
        key="use_live_weather",
    )

    if generate_btn:
        option_mode, option_value = location_labels[selected_location_option]
        st.session_state["prediction_mode"] = option_mode
        if option_mode == "preset":
            st.session_state["prediction_city"] = option_value
        st.session_state["generate_forecast"] = True
        st.rerun()

    st.caption(f"Active location: {live_label}")
    st.caption(live_location_status)
    st.caption(f"Prediction target: {prediction_label}")
    st.caption(f"Live date: {live_today} • latest data used: {latest_data_date}")

    if st.session_state.get("geo_error_message"):
        st.caption(f"Location issue: {st.session_state['geo_error_message']}")
        st.session_state["geo_error_message"] = None
    if load_issue:
        st.caption(f"Live fetch was unavailable. Using saved history instead. Details: {load_issue}")

    st.markdown("</div>", unsafe_allow_html=True)

with prediction_col:
    st.markdown(
        f"""
        <div class="prediction-card delay-2">
            <div class="prediction-label">Predicted Weather</div>
            <div class="prediction-value">{avg_temp_pred:.1f}°C</div>
            <div class="prediction-copy">
                Forecast for <strong>{prediction_label}</strong> with expected rain of <strong>{rainfall_pred:.1f} mm</strong>.
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-label">Predicted City</div>
                    <div class="hero-stat-value">{prediction_label}</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Tomorrow Temp</div>
                    <div class="hero-stat-value">{avg_temp_pred:.1f}°C</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Tomorrow Rain</div>
                    <div class="hero-stat-value">{rainfall_pred:.1f} mm</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Temp Shift</div>
                    <div class="hero-stat-value">{temp_delta_pred:+.1f}°C</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="status-strip">
        <div class="status-card delay-1">
            <div class="status-label">7-Day Trend</div>
            <div class="status-value">{trend_delta:+.1f}°C</div>
            <div class="status-note">{trend_delta:+.1f}°C versus 7 days ago</div>
        </div>
        <div class="status-card delay-2">
            <div class="status-label">Rain Frequency</div>
            <div class="status-value">{rainy_days}/60</div>
            <div class="status-note">{recent_rain_mean:.1f} mm daily average over the last 7 days</div>
        </div>
        <div class="status-card delay-3">
            <div class="status-label">Temperature Range</div>
            <div class="status-value">{temp_range:.1f}°C</div>
            <div class="status-note">Observed between the coolest low and warmest high in this window</div>
        </div>
        <div class="status-card delay-4">
            <div class="status-label">Atmospheric Read</div>
            <div class="status-value">{condition_name}</div>
            <div class="status-note">{condition_copy}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-stack"><div class="section-label">Climate Playback</div><div class="section-title">See how the weather moved before the forecast.</div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">Instead of a single flat chart, the history section now breaks out temperature, moisture, and atmospheric pressure so the model context reads more like an operational weather console.</div>',
    unsafe_allow_html=True,
)

insight_cols = st.columns(3, gap="large")
with insight_cols[0]:
    st.markdown(
        f"""
        <div class="insight-card delay-1">
            <div class="insight-title">Warmest day</div>
            <div class="insight-value">{latest['max_temp'].max():.1f}°C</div>
            <div class="insight-copy">Peak daytime temperature inside the loaded 60-day history for {prediction_label}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with insight_cols[1]:
    st.markdown(
        f"""
        <div class="insight-card delay-2">
            <div class="insight-title">Wettest day</div>
            <div class="insight-value">{latest['rainfall'].max():.1f} mm</div>
            <div class="insight-copy">Highest single-day precipitation event recorded in the active sequence.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with insight_cols[2]:
    st.markdown(
        f"""
        <div class="insight-card delay-3">
            <div class="insight-title">Avg pressure band</div>
            <div class="insight-value">{latest['pressure'].mean():.0f} hPa</div>
            <div class="insight-copy">Mean surface pressure across the same timeline used for forecasting.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_temp, tab_rain, tab_atmos = st.tabs(["Temperature Field", "Rainfall Pulse", "Atmospheric Signals"])

with tab_temp:
    temp_fig = go.Figure()
    temp_fig.add_trace(
        go.Scatter(
            x=latest["date"],
            y=latest["max_temp"],
            mode="lines",
            line=dict(color="#ffb36b", width=2.5),
            name="Max temp",
            hovertemplate="%{x|%b %d, %Y}<br>Max: %{y:.1f}°C<extra></extra>",
        )
    )
    temp_fig.add_trace(
        go.Scatter(
            x=latest["date"],
            y=latest["min_temp"],
            mode="lines",
            line=dict(color="#68d7ff", width=2.5),
            name="Min temp",
            hovertemplate="%{x|%b %d, %Y}<br>Min: %{y:.1f}°C<extra></extra>",
        )
    )
    temp_fig.add_trace(
        go.Scatter(
            x=latest["date"],
            y=latest["avg_temp"],
            mode="lines",
            line=dict(color="#f6fbff", width=3),
            name="Avg temp",
            hovertemplate="%{x|%b %d, %Y}<br>Average: %{y:.1f}°C<extra></extra>",
        )
    )
    temp_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d7edf8", family="IBM Plex Sans"),
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    temp_fig.update_xaxes(showgrid=False, zeroline=False, title="")
    temp_fig.update_yaxes(showgrid=True, gridcolor="rgba(143, 181, 200, 0.12)", zeroline=False, title="")
    st.plotly_chart(temp_fig, use_container_width=True)

with tab_rain:
    rain_fig = px.bar(
        latest,
        x="date",
        y="rainfall",
        color="rainfall",
        color_continuous_scale=["#123b4c", "#0ea5e9", "#6ee7b7"],
        labels={"date": "Date", "rainfall": "Rainfall (mm)"},
    )
    rain_fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#d7edf8", family="IBM Plex Sans"),
        margin=dict(l=10, r=10, t=20, b=10),
        coloraxis_showscale=False,
    )
    rain_fig.update_xaxes(showgrid=False, zeroline=False, title="")
    rain_fig.update_yaxes(showgrid=True, gridcolor="rgba(143, 181, 200, 0.12)", zeroline=False, title="")
    st.plotly_chart(rain_fig, use_container_width=True)

with tab_atmos:
    atmos_left, atmos_right = st.columns(2, gap="large")
    with atmos_left:
        humidity_fig = build_series_chart(
            latest,
            y="humidity",
            title="Humidity Flow",
            color="#68d7ff",
            fill=True,
        )
        st.plotly_chart(humidity_fig, use_container_width=True)
    with atmos_right:
        pressure_fig = build_series_chart(
            latest,
            y="pressure",
            title="Pressure Drift",
            color="#ffb36b",
        )
        st.plotly_chart(pressure_fig, use_container_width=True)

st.markdown('<div class="panel delay-4">', unsafe_allow_html=True)
st.markdown('<div class="panel-title">Forecast Interpretation</div>', unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="panel-kicker">
        The model is forecasting <strong>{avg_temp_pred:.1f}°C</strong> and
        <strong>{rainfall_pred:.1f} mm</strong> next. That outlook is shaped by a recent sequence
        where humidity averaged <strong>{recent_humidity_mean:.0f}%</strong>, wind averaged
        <strong>{recent_wind_mean:.1f} km/h</strong>, and the active data source was
        <strong>{data_mode}</strong>.
    </div>
    <div class="footnote">
        If the live Open-Meteo request is unavailable, the app automatically falls back to the bundled dataset and still runs the same model pipeline.
    </div>
    """,
    unsafe_allow_html=True,
)
metric_left, metric_right, metric_third = st.columns(3, gap="large")
with metric_left:
    st.metric(
        "7-day average temperature",
        f"{recent_temp_mean:.1f}°C",
        f"{recent_temp_mean - latest['avg_temp'].head(7).mean():+.1f}°C vs first week",
    )
with metric_right:
    st.metric(
        "7-day average rainfall",
        f"{recent_rain_mean:.1f} mm",
        f"{latest['rainfall'].tail(7).sum():.1f} mm accumulated",
    )
with metric_third:
    st.metric(
        "Sequence length",
        f"{len(latest)} days",
        latest['date'].iloc[-1].strftime("Ends %b %d, %Y"),
    )
st.markdown("</div>", unsafe_allow_html=True)
