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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

        .weather-icon-wrap {
            position: absolute;
            inset: 12% 8% auto auto;
            width: 90px;
            height: 90px;
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0.85;
        }

        .weather-icon-wrap svg {
            width: 100%;
            height: 100%;
            filter: drop-shadow(0 0 20px rgba(104, 215, 255, 0.25));
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

        .mission-control {
            background: linear-gradient(145deg, rgba(14, 80, 120, 0.9), rgba(7, 27, 40, 0.95));
            border: 2px solid rgba(104, 215, 255, 0.3);
            border-radius: 28px;
            padding: 1.5rem;
            box-shadow:
                0 20px 50px rgba(0, 0, 0, 0.35),
                0 0 40px rgba(104, 215, 255, 0.08),
                inset 0 1px 0 rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            position: relative;
            overflow: hidden;
        }

        .mission-control::before {
            content: "";
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(104, 215, 255, 0.06) 0%, transparent 50%);
            animation: pulseGlow 6s ease-in-out infinite;
            pointer-events: none;
        }

        .mission-control-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.3rem;
            position: relative;
            z-index: 1;
        }

        .mission-control-kicker {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.5;
            margin-bottom: 1.2rem;
            position: relative;
            z-index: 1;
        }

        .mission-control .stSelectbox,
        .mission-control .stCheckbox {
            position: relative;
            z-index: 1;
        }

        .mission-control [data-baseweb="select"] {
            border-radius: 14px !important;
            border: 1px solid rgba(104, 215, 255, 0.2) !important;
            background: rgba(255, 255, 255, 0.05) !important;
            transition: all 0.3s ease !important;
        }

        .mission-control [data-baseweb="select"]:hover {
            border-color: rgba(104, 215, 255, 0.4) !important;
            background: rgba(255, 255, 255, 0.08) !important;
            box-shadow: 0 0 20px rgba(104, 215, 255, 0.15) !important;
        }

        .mission-control [data-baseweb="select"]:focus {
            border-color: rgba(104, 215, 255, 0.6) !important;
            box-shadow: 0 0 25px rgba(104, 215, 255, 0.25) !important;
        }

        .mission-control [data-baseweb="select"] > div {
            background: transparent !important;
            border-radius: 14px !important;
        }

        .mission-control .stSelectbox label {
            color: var(--accent) !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.1em !important;
            margin-bottom: 0.4rem !important;
        }

        .mission-control [data-baseweb="popover"] {
            background: rgba(7, 27, 40, 0.98) !important;
            border: 1px solid rgba(104, 215, 255, 0.2) !important;
            border-radius: 14px !important;
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4), 0 0 30px rgba(104, 215, 255, 0.1) !important;
            backdrop-filter: blur(20px) !important;
        }

        .mission-control [data-baseweb="option"] {
            background: transparent !important;
            border-radius: 10px !important;
            padding: 0.6rem 1rem !important;
            transition: all 0.2s ease !important;
        }

        .mission-control [data-baseweb="option"]:hover {
            background: rgba(104, 215, 255, 0.15) !important;
        }

        .mission-control [data-baseweb="option"]:focus {
            background: rgba(104, 215, 255, 0.2) !important;
        }

        .mission-control [aria-selected="true"] {
            background: rgba(14, 165, 233, 0.3) !important;
            color: var(--text) !important;
        }

        .mission-control .stCheckbox {
            margin-top: 0.8rem;
        }

        .mission-control .stCheckbox > label {
            color: var(--muted) !important;
            font-size: 0.85rem !important;
        }

        .mission-control .stButton > button {
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            border: 1px solid rgba(104, 215, 255, 0.4);
            font-size: 1.05rem;
            font-weight: 700;
            padding: 0.6rem 1.5rem;
            box-shadow: 0 8px 24px rgba(14, 165, 233, 0.35);
        }

        .mission-control .stButton > button:hover {
            background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
            box-shadow: 0 12px 32px rgba(14, 165, 233, 0.45);
            transform: translateY(-2px);
        }

        .mission-control-info {
            margin-top: 1rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(104, 215, 255, 0.12);
            position: relative;
            z-index: 1;
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


@st.cache_data(ttl=120, show_spinner=False)
def check_internet():
    """Fast connectivity check — just tests if we can reach Open-Meteo."""
    import socket
    try:
        socket.create_connection(("api.open-meteo.com", 443), timeout=2).close()
        return True
    except Exception:
        return False


is_online = check_internet()


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


@st.cache_data
def evaluate_saved_model():
    dataset = load_data().sort_values(["city", "date"]).reset_index(drop=True)
    scaled_arr = scaler.transform(dataset[features])
    scaled_df = pd.DataFrame(scaled_arr, columns=features)
    scaled_df["city"] = dataset["city"].values

    X_eval, y_eval = [], []
    for _, city_group in scaled_df.groupby("city"):
        city_values = city_group[features].values
        for i in range(len(city_values) - SEQ_LEN):
            X_eval.append(city_values[i : i + SEQ_LEN])
            y_eval.append(city_values[i + SEQ_LEN][[2, 4]])

    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)
    split = int(len(X_eval) * 0.7)
    X_test = X_eval[split:]
    y_test = y_eval[split:]

    pred_scaled = model(X_test, training=False).numpy()
    y_true_full = np.zeros((len(y_test), len(features)))
    y_pred_full = np.zeros((len(pred_scaled), len(features)))
    y_true_full[:, 2] = y_test[:, 0]
    y_true_full[:, 4] = y_test[:, 1]
    y_pred_full[:, 2] = pred_scaled[:, 0]
    y_pred_full[:, 4] = pred_scaled[:, 1]

    y_true_real = scaler.inverse_transform(pd.DataFrame(y_true_full, columns=features))
    y_pred_real = scaler.inverse_transform(pd.DataFrame(y_pred_full, columns=features))

    true_temp = y_true_real[:, 2]
    pred_temp = y_pred_real[:, 2]
    true_rain = y_true_real[:, 4]
    pred_rain = y_pred_real[:, 4]

    return {
        "samples": int(len(X_test)),
        "temp_rmse": float(np.sqrt(mean_squared_error(true_temp, pred_temp))),
        "temp_mae": float(mean_absolute_error(true_temp, pred_temp)),
        "temp_r2": float(r2_score(true_temp, pred_temp)),
        "rain_rmse": float(np.sqrt(mean_squared_error(true_rain, pred_rain))),
        "rain_mae": float(mean_absolute_error(true_rain, pred_rain)),
        "rain_r2": float(r2_score(true_rain, pred_rain)),
    }


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


def calculate_feels_like(temp_c, humidity, wind_speed):
    """Calculate apparent temperature (feels like) using simplified heat index and wind chill."""
    if temp_c >= 20 and wind_speed < 4.8:
        # Heat index for warm temperatures
        T = temp_c
        RH = humidity
        HI = (
            -8.78469475556
            + 1.61139411 * T
            + 2.33854883889 * RH
            - 0.14611605 * T * RH
            - 0.012308094 * T ** 2
            - 0.0164248277778 * RH ** 2
            + 0.002211732 * T ** 2 * RH
            + 0.00072546 * T * RH ** 2
            - 0.000003582 * T ** 2 * RH ** 2
        )
        return max(HI, temp_c)
    elif temp_c < 10 and wind_speed > 4.8:
        # Wind chill for cold temperatures
        V = wind_speed ** 0.16
        WC = 13.12 + 0.6215 * temp_c - 11.37 * V + 0.3965 * temp_c * V
        return min(WC, temp_c)
    return temp_c


def get_weather_icon_svg(cloud_cover, rainfall, avg_temp, humidity):
    """Return an SVG weather icon based on conditions."""
    # Determine weather type
    is_rainy = rainfall >= 2
    is_cloudy = cloud_cover >= 60
    is_hot = avg_temp >= 32

    if is_rainy:
        # Rain icon
        return '''<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="rainCloud" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#8fb5c8"/>
                    <stop offset="100%" stop-color="#5a8fa8"/>
                </linearGradient>
            </defs>
            <path d="M75 45c0-11-9-20-20-20-8.5 0-15.8 5.3-18.7 12.8C35.2 36.3 33 35 30.5 35c-4.1 0-7.5 3.4-7.5 7.5 0 0.4 0 0.7 0.1 1.1C18.2 44.7 15 48.7 15 53.5 15 58.8 19.2 63 24.5 63h50.5c5.2 0 9.5-4.2 9.5-9.5 0-5-3.9-9.2-8.9-9.5 0-0.3-0.1-0.7-0.1-1-0.5-0.1-1-0.1-1.5-0.1-8.8 0-16 7.2-16 16 0 0.2 0 0.5 0 0.7" fill="url(#rainCloud)"/>
            <line x1="30" y1="70" x2="25" y2="82" stroke="#68d7ff" stroke-width="2.5" stroke-linecap="round"/>
            <line x1="45" y1="68" x2="40" y2="80" stroke="#68d7ff" stroke-width="2.5" stroke-linecap="round"/>
            <line x1="60" y1="70" x2="55" y2="82" stroke="#68d7ff" stroke-width="2.5" stroke-linecap="round"/>
            <line x1="38" y1="78" x2="33" y2="90" stroke="#68d7ff" stroke-width="2.5" stroke-linecap="round"/>
            <line x1="53" y1="76" x2="48" y2="88" stroke="#68d7ff" stroke-width="2.5" stroke-linecap="round"/>
            <line x1="68" y1="78" x2="63" y2="90" stroke="#68d7ff" stroke-width="2.5" stroke-linecap="round"/>
        </svg>'''
    elif is_cloudy:
        # Cloudy icon
        return '''<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="cloudGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stop-color="#b8d4e3"/>
                    <stop offset="100%" stop-color="#8fb5c8"/>
                </linearGradient>
            </defs>
            <path d="M78 42c0-9.4-7.6-17-17-17-6.8 0-12.6 4-15.4 9.7C44.3 33.1 42.4 32 40.2 32c-3.9 0-7 3.1-7 7 0 0.4 0 0.8 0.1 1.1C29.6 40.9 27 44 27 48c0 4.4 3.6 8 8 8h47c4.4 0 8-3.6 8-8 0-4.4-3.6-8-8-8h-1c0-5.5-4.5-10-10-10-1.4 0-2.7 0.3-3.9 0.8C68.8 26.2 73 21.8 78 21c-1.8-1.2-4-2-6.4-2-7.2 0-13 5.8-13 13 0 0.5 0 1 0.1 1.5" fill="url(#cloudGrad)"/>
            <circle cx="55" cy="38" r="12" fill="#ffb36b" opacity="0.9"/>
            <circle cx="62" cy="33" r="6" fill="#ffd700" opacity="0.8"/>
        </svg>'''
    elif is_hot:
        # Sunny hot icon
        return '''<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="sunGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#ffd700"/>
                    <stop offset="100%" stop-color="#ffb36b"/>
                </linearGradient>
            </defs>
            <circle cx="50" cy="50" r="22" fill="url(#sunGrad)"/>
            <g stroke="#ffb36b" stroke-width="3" stroke-linecap="round">
                <line x1="50" y1="12" x2="50" y2="20"/>
                <line x1="50" y1="80" x2="50" y2="88"/>
                <line x1="12" y1="50" x2="20" y2="50"/>
                <line x1="80" y1="50" x2="88" y2="50"/>
                <line x1="23" y1="23" x2="29" y2="29"/>
                <line x1="71" y1="71" x2="77" y2="77"/>
                <line x1="23" y1="77" x2="29" y2="71"/>
                <line x1="71" y1="29" x2="77" y2="23"/>
            </g>
        </svg>'''
    else:
        # Clear/sunny icon
        return '''<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="clearGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stop-color="#68d7ff"/>
                    <stop offset="100%" stop-color="#0ea5e9"/>
                </linearGradient>
            </defs>
            <circle cx="50" cy="50" r="20" fill="url(#clearGrad)"/>
            <g stroke="#68d7ff" stroke-width="2.5" stroke-linecap="round" opacity="0.8">
                <line x1="50" y1="14" x2="50" y2="22"/>
                <line x1="50" y1="78" x2="50" y2="86"/>
                <line x1="14" y1="50" x2="22" y2="50"/>
                <line x1="78" y1="50" x2="86" y2="50"/>
                <line x1="25" y1="25" x2="31" y2="31"/>
                <line x1="69" y1="69" x2="75" y2="75"/>
                <line x1="25" y1="75" x2="31" y2="69"/>
                <line x1="69" y1="31" x2="75" y2="25"/>
            </g>
        </svg>'''


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

model = load_model()
scaler = load_scaler()
df = load_data()
evaluation_metrics = evaluate_saved_model()

cities = sorted(df["city"].unique())

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
    is_online
    and not st.session_state.get("auto_location_attempted")
    and not st.session_state.get("browser_location")
):
    st.session_state["auto_location_attempted"] = True
    st.session_state["request_browser_location"] = True
    st.session_state["location_mode"] = "browser"

if is_online and st.session_state.get("request_browser_location"):
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
if not is_online:
    st.session_state["use_live_weather"] = False
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

# Fetch actual current weather from Open-Meteo Forecast API
from src.api import get_current_weather as fetch_current_weather, get_weather_description

if is_online:
    try:
        if live_coords:
            current_weather = fetch_current_weather(latitude=live_coords[0], longitude=live_coords[1])
        else:
            current_weather = fetch_current_weather(live_fallback_city)

        # Use actual current weather for the live section
        actual_current_temp = current_weather["current_temp"]
        actual_feels_like = current_weather["feels_like"]
        actual_humidity = current_weather["humidity"]
        actual_precipitation = current_weather["precipitation"]
        actual_wind = current_weather["wind_speed"]
        actual_pressure = current_weather["pressure"]
        actual_cloud = current_weather["cloud_cover"]
        actual_weather_code = current_weather["weather_code"]
        actual_weather_desc = get_weather_description(actual_weather_code)
        actual_current_time = current_weather["time"]

        live_data_mode = "Live Open-Meteo (Current)"
        use_current_for_live = True
    except Exception:
        # Fallback to archive data if forecast API fails
        actual_current_temp = avg_temp_now
        actual_feels_like = calculate_feels_like(avg_temp_now, humidity_now, wind_now)
        actual_humidity = humidity_now
        actual_precipitation = rainfall_now
        actual_wind = wind_now
        actual_pressure = pressure_now
        actual_cloud = cloud_now
        actual_weather_desc = condition_name
        actual_current_time = None
        live_data_mode = f"{data_mode} (Forecast API unavailable)"
        use_current_for_live = False
else:
    # Offline — use local dataset values
    actual_current_temp = avg_temp_now
    actual_feels_like = calculate_feels_like(avg_temp_now, humidity_now, wind_now)
    actual_humidity = humidity_now
    actual_precipitation = rainfall_now
    actual_wind = wind_now
    actual_pressure = pressure_now
    actual_cloud = cloud_now
    actual_weather_desc = condition_name
    actual_current_time = None
    live_data_mode = "Offline Mode"
    use_current_for_live = False

# Calculate additional live weather metrics for the hero-aside
feels_like_now = actual_feels_like
weather_icon_svg = get_weather_icon_svg(actual_cloud, actual_precipitation, actual_current_temp, actual_humidity)

# Build the full hero section HTML
from textwrap import dedent as _dedent

if is_online:
    _hero_aside = f"""\
<div class="hero-aside delay-2">
<div class="hero-aside-top">
<div><p class="hero-aside-title">{live_label} - Current Weather</p></div>
<div class="signal-pill">{live_data_mode}</div>
</div>
<div class="weather-icon-wrap">{weather_icon_svg}</div>
<div class="hero-temp">{actual_current_temp:.1f}°C</div>
<div class="hero-temp-sub" style="margin-bottom: 1rem;">Today: {live_today}{f" • Updated: {actual_current_time}" if actual_current_time else ""}</div>
<div class="hero-stats" style="margin-top: 0.9rem;">
<div class="hero-stat"><div class="hero-stat-label">Condition</div><div class="hero-stat-value">{actual_weather_desc}</div></div>
<div class="hero-stat"><div class="hero-stat-label">Precipitation</div><div class="hero-stat-value">{actual_precipitation:.1f} mm</div></div>
<div class="hero-stat"><div class="hero-stat-label">Humidity</div><div class="hero-stat-value">{actual_humidity:.0f}%</div></div>
</div>
<div class="hero-radar" style="margin-top: 0.9rem;">
<div class="hero-radar-grid">
<div class="hero-radar-item"><div class="hero-radar-value">{actual_wind:.1f} km/h</div><div class="hero-radar-label">Wind</div></div>
<div class="hero-radar-item"><div class="hero-radar-value">{actual_pressure:.0f} hPa</div><div class="hero-radar-label">Pressure</div></div>
</div>
</div>
</div>"""
else:
    _hero_aside = """\
<div class="hero-aside delay-2">
<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;text-align:center;padding:2rem;">
<div style="font-size:3rem;margin-bottom:0.8rem;">📡</div>
<p class="hero-aside-title" style="margin:0;">Offline Mode</p>
<p class="hero-aside-copy" style="max-width:100%;margin-top:0.6rem;">No internet connection detected. Live weather data is unavailable.<br><br>Select a city below and click <strong>Get Weather Prediction</strong> to forecast using the bundled dataset.</p>
<div class="signal-pill" style="margin-top:1rem;">Dataset Mode</div>
</div>
</div>"""

forecast_ready = False
prediction_current_avg = float(latest.iloc[-1]["avg_temp"])
avg_temp_pred = None
rainfall_pred = None

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

recent_temp_mean = latest["avg_temp"].tail(7).mean()
recent_rain_mean = latest["rainfall"].tail(7).mean()
recent_humidity_mean = latest["humidity"].tail(7).mean()
recent_wind_mean = latest["wind_speed"].tail(7).mean()
trend_delta = latest["avg_temp"].iloc[-1] - latest["avg_temp"].iloc[-7]
temp_range = latest["max_temp"].max() - latest["min_temp"].min()
rainy_days = int((latest["rainfall"] > 0).sum())

if forecast_ready:
    temp_delta_pred = avg_temp_pred - prediction_current_avg
    forecast_title = "Next-Day Model Forecast"
    forecast_value = f"{avg_temp_pred:.1f}°C"
    forecast_copy = (
        f"Forecast for <strong>{prediction_label}</strong> with expected rain of "
        f"<strong>{rainfall_pred:.1f} mm</strong> from the trained predictive model."
    )
    forecast_city_value = prediction_label
    forecast_temp_value = f"{avg_temp_pred:.1f}°C"
    forecast_rain_value = f"{rainfall_pred:.1f} mm"
    forecast_shift_value = f"{temp_delta_pred:+.1f}°C"
else:
    temp_delta_pred = 0.0
    forecast_title = "Next-Day Model Forecast"
    forecast_value = "Awaiting run"
    forecast_copy = (
        "Choose a location and click <strong>Get Weather Prediction</strong> to run the "
        "trained time-series model for tomorrow's forecast."
    )
    forecast_city_value = prediction_label
    forecast_temp_value = "Pending"
    forecast_rain_value = "Pending"
    forecast_shift_value = "Pending"

_hero_html = f"""\
<section class="hero-shell">
<div class="hero-grid">
<div class="hero-copy-wrap">
<div class="eyebrow">Time Series Analysis</div>
<h1 class="hero-title" style="white-space: normal; word-break: normal;">Weather Forecasting Using Time Series Analysis: Predictive Modeling for Climate Patterns</h1>
<p class="hero-copy" style="max-width: 100%; font-size: 1rem; line-height: 1.6; margin-top: 0.8rem;">
A deep learning model that predicts tomorrow's temperature and rainfall by analyzing the last 60 days of weather patterns. Select any of the 32 supported Indian cities or use your live location to get instant weather forecasts powered by LSTM neural networks trained on Open-Meteo historical data.<br><br>
The CNN-LSTM architecture captures both local patterns and long-term dependencies in weather sequences, providing accurate next-day predictions. Weather data includes temperature, humidity, rainfall, wind speed, pressure, and cloud cover for comprehensive climate analysis.<br><br>
Built with TensorFlow/Keras for deep learning, Streamlit for the web interface, and Open-Meteo API for real-time weather data across major Indian cities.
</p>
</div>
{_hero_aside}
</div>
</section>
"""
st.markdown(_hero_html, unsafe_allow_html=True)

if is_online:
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
    st.markdown('<div class="mission-control delay-1">', unsafe_allow_html=True)
    st.markdown('<div class="mission-control-title">Mission Control</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="mission-control-kicker">Select a city and click the button to predict tomorrow\'s weather using CNN-LSTM neural network.</div>',
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
        "Choose City",
        location_options,
        index=location_options.index(default_option),
    )

    generate_btn = st.button("Get Weather Prediction", type="primary", use_container_width=True)

    st.checkbox(
        "Use live Open-Meteo history",
        value=use_live_data,
        help="No internet connection. Using bundled dataset." if not is_online else "Enabled: pulls the latest 60-day history from Open-Meteo. Disabled: uses the dataset bundled with this project.",
        key="use_live_weather",
        disabled=not is_online,
    )

    if not is_online:
        st.caption("⚠️ No internet — using bundled dataset for predictions.")

    if generate_btn:
        option_mode, option_value = location_labels[selected_location_option]
        st.session_state["prediction_mode"] = option_mode
        if option_mode == "preset":
            st.session_state["prediction_city"] = option_value
        st.session_state["generate_forecast"] = True
        st.rerun()

    st.markdown('<div class="mission-control-info">', unsafe_allow_html=True)
    st.caption(f"Active location: {live_label}")
    st.caption(f"Prediction target: {prediction_label}")
    st.caption(f"Live date: {live_today} • latest data used: {latest_data_date}")
    if load_issue:
        st.caption(f"Live fetch was unavailable. Using saved history instead.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with prediction_col:
    st.markdown(
        f"""
        <div class="prediction-card delay-2">
            <div class="prediction-label">{forecast_title}</div>
            <div class="prediction-value">{forecast_value}</div>
            <div class="prediction-copy">
                {forecast_copy}
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-label">Forecast Target</div>
                    <div class="hero-stat-value">{forecast_city_value}</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Tomorrow Temp</div>
                    <div class="hero-stat-value">{forecast_temp_value}</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Tomorrow Rain</div>
                    <div class="hero-stat-value">{forecast_rain_value}</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-label">Temp Shift</div>
                    <div class="hero-stat-value">{forecast_shift_value}</div>
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

st.markdown('<div class="section-stack"><div class="section-label">Historical Input Data</div><div class="section-title">See how the weather moved before the forecast.</div></div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-copy">This section shows the 60-day time-series window used as model input. It breaks out temperature, moisture, rainfall, and pressure so the forecasting context is clear before tomorrow\'s prediction is generated.</div>',
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

st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)

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
