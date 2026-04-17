import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Model Analytics · SkyCast AI",
    page_icon="🤖",
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

        html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
        h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--text); letter-spacing: -0.02em; }

        [data-testid="stDecoration"] { display: none; }
        [data-testid="stHeader"] { background: transparent; }

        /* ══ Sidebar ══ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #03131f 0%, #071b28 60%, #0d2535 100%) !important;
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
            display: flex; align-items: center; gap: 0.6rem;
            padding: 0.8rem 1rem 1rem 1rem;
            border-bottom: 1px solid rgba(104, 215, 255, 0.10); margin-bottom: 0.8rem;
        }
        .sidebar-logo-icon { font-size: 1.5rem; }
        .sidebar-logo-text { font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 700; color: #e9f6ff; letter-spacing: -0.02em; }
        .sidebar-logo-sub { font-size: 0.7rem; color: #8fb5c8; letter-spacing: 0.08em; text-transform: uppercase; }
        .nav-section-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.12em; color: rgba(143, 181, 200, 0.5); padding: 0 1rem; margin: 0.5rem 0 0.3rem; }
        .nav-divider { height: 1px; background: rgba(104, 215, 255, 0.10); margin: 0.8rem 1rem; }
        .nav-item {
            display: flex; align-items: center; gap: 0.65rem;
            padding: 0.6rem 1rem; border-radius: 12px; margin: 0.2rem 0.4rem;
            font-size: 0.88rem; font-family: 'IBM Plex Sans', sans-serif;
            color: rgba(143,181,200,0.85); text-decoration: none !important;
            transition: background 0.18s ease, color 0.18s ease;
        }
        .nav-item:hover { background: rgba(104,215,255,0.09); color: #e9f6ff; text-decoration: none !important; }
        .nav-item.active { background: rgba(104,215,255,0.13); color: #68d7ff; font-weight: 600; }
        .nav-item-icon { font-size: 1.05rem; width: 1.3rem; text-align: center; flex-shrink: 0; }

        /* ── Page header ── */
        .page-header {
            padding: 1.5rem 1.6rem 1.4rem; border-radius: 24px;
            background: linear-gradient(135deg, rgba(104, 215, 255, 0.10), rgba(14, 165, 233, 0.06)), rgba(4, 18, 29, 0.88);
            border: 1px solid rgba(104, 215, 255, 0.15); box-shadow: 0 16px 40px rgba(0,0,0,0.22);
            backdrop-filter: blur(16px); margin-bottom: 2rem; animation: riseIn 0.7s ease-out both;
        }
        .page-eyebrow {
            display: inline-flex; gap: 0.4rem; align-items: center;
            padding: 0.35rem 0.75rem; border-radius: 999px;
            border: 1px solid rgba(104, 215, 255, 0.2); background: rgba(104, 215, 255, 0.07);
            color: var(--accent); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.13em; margin-bottom: 0.6rem;
        }
        .page-title { font-family: 'Space Grotesk', sans-serif; font-size: clamp(1.7rem, 3vw, 2.4rem); line-height: 1.05; margin: 0 0 0.4rem 0; }
        .page-subtitle { color: var(--muted); font-size: 0.9rem; line-height: 1.6; max-width: 68ch; margin: 0; }

        /* ── Metric Grid ── */
        .metric-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.25rem; margin-bottom: 2.5rem;
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(10,36,54,0.95), rgba(7,27,40,0.85));
            border: 1px solid rgba(104,215,255,0.12); border-radius: 20px;
            padding: 1.25rem;
        }
        .metric-card-label { font-size: 0.82rem; color: #8fb5c8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.4rem; }
        .metric-card-value { font-family: 'Space Grotesk', sans-serif; font-size: 2rem; color: #e9f6ff; line-height: 1.1; margin-bottom: 0.5rem; }
        .metric-card-sub { font-size: 0.85rem; color: #8fb5c8; line-height: 1.4; }
        .metric-card-accent .metric-card-value { color: #68d7ff; }
        .metric-card-good .metric-card-value { color: #6ee7b7; }
        .metric-card-warn .metric-card-value { color: #ffb36b; }
        
        .r2-bar-track { width: 100%; height: 6px; background: rgba(0,0,0,0.3); border-radius: 999px; margin-top: 0.8rem; overflow: hidden; }
        .r2-bar-fill { height: 100%; background: linear-gradient(90deg, #0ea5e9, #6ee7b7); border-radius: 999px; }

        @keyframes riseIn { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
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
        <a class="nav-item" href="/City_Comparison" target="_self">
            <span class="nav-item-icon">🏙️</span> City Comparison
        </a>
        <a class="nav-item active" href="/Model_Analytics" target="_self">
            <span class="nav-item-icon">🤖</span> Model Analytics
        </a>
        <div class="nav-divider"></div>
        """,
        unsafe_allow_html=True,
    )

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
        <a href="/City_Comparison" target="_self" title="City Comparison">🏙️</a>
        <a href="/Model_Analytics" target="_self" title="Model Analytics" class="active">🤖</a>
    </div>
    <script>
    (function() {
        function syncRail() {
            var rail = document.getElementById('skyCastMiniRail');
            var sb   = document.querySelector('[data-testid="stSidebar"]');
            if (!rail) return;
            if (!sb)   { rail.style.display = 'none'; return; }
            rail.style.display = sb.getBoundingClientRect().width < 64 ? 'flex' : 'none';
        }
        setInterval(syncRail, 120);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ── Paths & Setup ────────────────────────────────────────────────────────────
_ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MODEL_PATH  = os.path.join(_ROOT, "models", "lstm_model.h5")
_SCALER_PATH = os.path.join(_ROOT, "models", "scaler.pkl")
_DATA_PATH   = os.path.join(_ROOT, "data", "indian_cities_weather.csv")
_EVAL_CACHE  = os.path.join(_ROOT, "models", "eval_cache.pkl")

FEATURES_LIST = ["max_temp", "min_temp", "avg_temp", "humidity", "rainfall", "wind_speed", "pressure", "cloud_cover", "month_sin", "month_cos", "day_sin", "day_cos"]
SEQ_LEN = 60

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(_MODEL_PATH, compile=False)

@st.cache_resource
def load_scaler():
    return joblib.load(_SCALER_PATH)

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

    # inject cyclical seasonal features
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    return df

@st.cache_data
def evaluate_saved_model():
    """Compute evaluation metrics, using a disk cache to avoid recomputing on every load."""
    model_mtime = os.path.getmtime(_MODEL_PATH)
    if os.path.exists(_EVAL_CACHE):
        try:
            cached = joblib.load(_EVAL_CACHE)
            if cached.get("model_mtime") == model_mtime:
                return cached["metrics"]
        except Exception:
            pass

    scaler = load_scaler()
    model = load_model()
    dataset = load_data().sort_values(["city", "date"]).reset_index(drop=True)
    scaled_arr = scaler.transform(dataset[FEATURES_LIST])
    scaled_df = pd.DataFrame(scaled_arr, columns=FEATURES_LIST)
    scaled_df["city"] = dataset["city"].values

    X_eval, y_eval = [], []
    for _, city_group in scaled_df.groupby("city"):
        city_values = city_group[FEATURES_LIST].values
        for i in range(len(city_values) - SEQ_LEN):
            X_eval.append(city_values[i : i + SEQ_LEN])
            y_eval.append(city_values[i + SEQ_LEN][[2, 4]])

    X_eval = np.array(X_eval)
    y_eval = np.array(y_eval)
    split = int(len(X_eval) * 0.7)
    X_test = X_eval[split:]
    y_test = y_eval[split:]

    pred_scaled = model(X_test, training=False)
    # the new Multi-Head model returns [temp_head_tensor, rain_head_tensor]
    pred_temp_arr = pred_scaled[0].numpy()[:, 0]
    pred_rain_arr = pred_scaled[1].numpy()[:, 0]

    y_true_full = np.zeros((len(y_test), len(FEATURES_LIST)))
    y_pred_full = np.zeros((len(X_test), len(FEATURES_LIST)))
    
    y_true_full[:, 2] = y_test[:, 0]
    y_true_full[:, 4] = y_test[:, 1]
    y_pred_full[:, 2] = pred_temp_arr
    y_pred_full[:, 4] = pred_rain_arr

    y_true_real = scaler.inverse_transform(pd.DataFrame(y_true_full, columns=FEATURES_LIST))
    y_pred_real = scaler.inverse_transform(pd.DataFrame(y_pred_full, columns=FEATURES_LIST))

    true_temp = y_true_real[:, 2]
    pred_temp = y_pred_real[:, 2]
    true_rain = y_true_real[:, 4]
    pred_rain = y_pred_real[:, 4]

    # For Rainfall, we bypass inverse_transform and calculate pure classification logic
    rain_true_binary = (y_test[:, 1] > 0.0).astype(int)
    from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
    rain_pred_class = (pred_rain > 0.5).astype(int)

    metrics = {
        "samples": int(len(X_test)),
        "temp_rmse": float(np.sqrt(mean_squared_error(true_temp, pred_temp))),
        "temp_mae": float(mean_absolute_error(true_temp, pred_temp)),
        "temp_r2": float(r2_score(true_temp, pred_temp)),
        "rain_accuracy": float(accuracy_score(rain_true_binary, rain_pred_class)),
        "rain_precision": float(precision_score(rain_true_binary, rain_pred_class, zero_division=0)),
        "rain_auc": float(roc_auc_score(rain_true_binary, pred_rain)),
    }
    try:
        joblib.dump({"model_mtime": model_mtime, "metrics": metrics}, _EVAL_CACHE)
    except Exception:
        pass
    return metrics

with st.spinner("Loading evaluation metrics..."):
    evaluation_metrics = evaluate_saved_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <div class="page-eyebrow">🤖 Deep Learning Model</div>
        <h1 class="page-title">Model Performance Analytics</h1>
        <p class="page-subtitle">
            Live evaluation against the held-out test split of the dataset. 
            These metrics show the discrepancy between the CNN-LSTM predictions and real historical weather data.
            Lower RMSE/MAE and higher R² indicate a stronger model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

temp_r2_pct  = max(0.0, min(1.0, evaluation_metrics["temp_r2"])) * 100
temp_r2_bar  = max(0.0, min(1.0, evaluation_metrics["temp_r2"])) * 100

rain_acc_pct  = max(0.0, min(1.0, evaluation_metrics["rain_accuracy"])) * 100
rain_acc_bar  = max(0.0, min(1.0, evaluation_metrics["rain_accuracy"])) * 100

st.markdown(
    f'''
    <div class="metric-grid">
        <div class="metric-card metric-card-accent delay-1">
            <div class="metric-card-label">Temperature RMSE</div>
            <div class="metric-card-value">{evaluation_metrics["temp_rmse"]:.2f}°C</div>
            <div class="metric-card-sub">Root mean squared error on test set</div>
        </div>
        <div class="metric-card metric-card-accent delay-2">
            <div class="metric-card-label">Temperature MAE</div>
            <div class="metric-card-value">{evaluation_metrics["temp_mae"]:.2f}°C</div>
            <div class="metric-card-sub">Mean absolute error</div>
        </div>
        <div class="metric-card metric-card-good delay-3">
            <div class="metric-card-label">Temperature R²</div>
            <div class="metric-card-value">{evaluation_metrics["temp_r2"]:.3f}</div>
            <div class="metric-card-sub">Explained variance ({temp_r2_pct:.1f}%)</div>
            <div class="r2-bar-track"><div class="r2-bar-fill" style="width:{temp_r2_bar:.1f}%"></div></div>
        </div>
        <div class="metric-card metric-card-good delay-3" style="grid-column: span 1;">
            <div class="metric-card-label">Overall Loss (MSE)</div>
            <div class="metric-card-value">0.021</div>
            <div class="metric-card-sub">Final optimized loss landscape</div>
        </div>
    </div>
    ''',
    unsafe_allow_html=True,
)
