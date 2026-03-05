import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import plotly.express as px
import warnings
import sys
import os

# Ensure the root directory is in the Python path so Streamlit can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Suppress warnings that clutter the terminal
warnings.filterwarnings('ignore', category=UserWarning)

st.set_page_config(
    page_title="SkyCast AI",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR PREMIUM AESTHETICS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    h1 {
        font-weight: 600;
        color: #38bdf8 !important;
        padding-bottom: 0px !important;
        margin-bottom: 5px !important;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 600 !important;
        color: #f8fafc !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        color: #94a3b8 !important;
        font-weight: 400;
    }
    [data-testid="stMetricDelta"] svg {
        display: none;
    }
    /* Hide the caret/arrows in metrics */
    div[data-testid="stMetricDelta"] {
        justify-content: left !important;
    }
    
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 15px 25px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: transform 0.2s ease, border 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    /* Primary buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #0ea5e9, #3b82f6);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #0284c7, #2563eb);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.4);
        color: white;
    }
    
    /* Forecast box */
    .forecast-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.05));
        border-left: 4px solid #10b981;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 15px 0px;
    }
    .forecast-title {
        color: #34d399;
        font-weight: 600;
        margin-bottom: 5px;
        font-size: 1.2rem;
    }
    .forecast-text {
        color: #e2e8f0;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

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
    df = df.rename(columns={
        'Date': 'date', 'City': 'city',
        'Temperature_Max (°C)': 'max_temp', 'Temperature_Min (°C)': 'min_temp',
        'Temperature_Avg (°C)': 'avg_temp', 'Humidity (%)': 'humidity',
        'Rainfall (mm)': 'rainfall', 'Wind_Speed (km/h)': 'wind_speed',
        'Pressure (hPa)': 'pressure', 'Cloud_Cover (%)': 'cloud_cover'
    })
    df['date'] = pd.to_datetime(df['date'])
    return df

model = load_model()
scaler = load_scaler()
df = load_data()
cities = sorted(df['city'].unique())

features = [
    'max_temp','min_temp','avg_temp',
    'humidity','rainfall','wind_speed',
    'pressure','cloud_cover'
]
SEQ_LEN = 60

# --- HERO SECTION ---
st.title("🌤️ SkyCast AI")
st.markdown("<div class='subtitle'>Advanced Neural Network Weather Intelligence • Powered by LSTM</div>", unsafe_allow_html=True)

# Layout Setup
col_input, col_metrics = st.columns([1, 2.5], gap="large")

with col_input:
    st.markdown("### Location & Source")
    city = st.selectbox("Select Target City", cities)
    use_live_data = st.checkbox("📡 Connect to Live Satellite API", value=True, 
                                help="When enabled, pulls the actual past 60 days of weather from the internet to make a real forecast. If disabled, simulates a forecast from 2025.")
    
    city_df = df[df['city'] == city].sort_values("date")
    
    # Retrieve Pipeline
    if use_live_data:
        from src.api import get_live_weather_data
        try:
            with st.spinner("Connecting to Open-Meteo Satellites..."):
                latest = get_live_weather_data(city, days=SEQ_LEN)
                success_load = True
        except Exception as e:
            st.error(f"Error fetching live data: {e}")
            latest = city_df[['date'] + features].tail(SEQ_LEN)
            success_load = False
    else:
        latest = city_df[['date'] + features].tail(SEQ_LEN)
        success_load = True
        
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("✨ Generate AI Forecast", type="primary", use_container_width=True)

with col_metrics:
    st.markdown(f"### Current Context ({latest['date'].iloc[-1].strftime('%B %d, %Y')})")
    # Current Stats 
    c1, c2, c3 = st.columns(3)
    c1.metric("Current Avg Temp", f"{latest['avg_temp'].iloc[-1]:.1f} °C", "Thermometer")
    c2.metric("Humidity", f"{latest['humidity'].iloc[-1]:.0f} %", "Moisture")
    c3.metric("Wind Speed", f"{latest['wind_speed'].iloc[-1]:.1f} kmh", "Gusts")
    
    if generate_btn and success_load:
        with st.spinner("Neural Network analyzing 60-day climate sequencing..."):
            latest_for_model = latest[features]
            latest_scaled = scaler.transform(latest_for_model)
            X_input = np.expand_dims(latest_scaled, axis=0)
            
            pred_tensor = model(X_input, training=False)
            pred = pred_tensor.numpy()
            
            dummy = np.zeros((1, 8))
            dummy[0, 2] = pred[0][0]
            dummy[0, 4] = pred[0][1]
            real_pred = scaler.inverse_transform(pd.DataFrame(dummy, columns=features))
            
            avg_temp_pred = real_pred[0, 2]
            rainfall_pred = real_pred[0, 4]
            # Zero out negative rainfall logic
            if rainfall_pred < 0:
                rainfall_pred = 0.0

        st.markdown(f"""
        <div class="forecast-box">
            <div class="forecast-title">🌟 Deep Learning Forecast for Tomorrow</div>
            <div class="forecast-text">
                The network anticipates an average temperature of <b>{avg_temp_pred:.1f}°C</b> 
                with approximately <b>{rainfall_pred:.1f}mm</b> of precipitation.
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# --- DATA VISUALIZATION ---
st.markdown("### 60-Day Historic Temperature Trend")
# Enhancing Plotly Theme
fig = px.line(latest, x="date", y="avg_temp", 
              labels={"avg_temp": "Temperature (°C)", "date": "Date"},
              color_discrete_sequence=["#38bdf8"])

fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8",
    font_family="Outfit",
    margin=dict(l=0, r=0, t=20, b=0),
    hovermode="x unified"
)
fig.update_xaxes(showgrid=False, zeroline=False, title="")
fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, title="")

st.plotly_chart(fig, width="stretch")