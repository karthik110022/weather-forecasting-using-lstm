import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st

# Open-Meteo requires latitude and longitude
# Major Indian Cities with Coordinates
CITY_COORDINATES = {
    # Existing 10 Cities
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126},

    # NEW 22 Major Indian Cities
    "Pune": {"lat": 18.5204, "lon": 73.8567},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794},
    "Kochi": {"lat": 9.9312, "lon": 76.2673},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185},
    "Patna": {"lat": 25.5941, "lon": 85.1376},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
    "Indore": {"lat": 22.7196, "lon": 75.8577},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366},
    "Mysore": {"lat": 12.2958, "lon": 76.6394},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243},
    "Ranchi": {"lat": 23.3441, "lon": 85.3095},
    "Shimla": {"lat": 31.1048, "lon": 77.1734},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322},
    "Srinagar": {"lat": 34.0837, "lon": 74.7973},
    "Jammu": {"lat": 32.7266, "lon": 74.8570},
    "Leh": {"lat": 34.1526, "lon": 77.5771},
    "Panaji": {"lat": 15.4909, "lon": 73.8278},
    "Mangalore": {"lat": 12.9141, "lon": 74.8560},
    "Surat": {"lat": 21.1702, "lon": 72.8311},
}


def geocode_location(query: str, count: int = 1) -> list[dict]:
    if not query.strip():
        raise ValueError("Location query cannot be empty.")

    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": query.strip(), "count": count, "language": "en", "format": "json"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        raise ValueError(f"No location matches found for '{query}'.")
    return results


def format_location_label(location: dict) -> str:
    parts = [
        location.get("name"),
        location.get("admin1"),
        location.get("country"),
    ]
    return ", ".join(part for part in parts if part)


def reverse_geocode_location(latitude: float, longitude: float) -> str:
    response = requests.get(
        "https://nominatim.openstreetmap.org/reverse",
        params={
            "lat": latitude,
            "lon": longitude,
            "format": "jsonv2",
            "zoom": 12,
            "addressdetails": 1,
        },
        headers={"User-Agent": "SkyCastAI/1.0"},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    address = data.get("address", {})

    parts = [
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or data.get("name"),
        address.get("state_district") or address.get("state"),
    ]
    label = ", ".join(part for part in parts if part)
    return label or "Near your location"


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_weather_data(
    city: str | None = None,
    days: int = 60,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pd.DataFrame:
    """
    Fetches the last N past days of historical weather data from Open-Meteo API
    for a given city, and formats it to exactly match the model's required 8 features.
    """
    if latitude is None or longitude is None:
        if city not in CITY_COORDINATES:
            raise ValueError(f"Coordinates for city '{city}' not found.")
        lat = CITY_COORDINATES[city]["lat"]
        lon = CITY_COORDINATES[city]["lon"]
    else:
        lat = latitude
        lon = longitude
    
    # Calculate dates
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # We fetch daily max, min, mean temp, rain, and max wind speed.
    # We fetch hourly humidity, surface pressure, and cloud cover to average them out per day.
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "hourly": ["relative_humidity_2m", "surface_pressure", "cloud_cover"],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    
    # Process Daily data
    daily_data = data["daily"]
    daily_df = pd.DataFrame({
        "date": pd.to_datetime(daily_data["time"]),
        "max_temp": daily_data["temperature_2m_max"],
        "min_temp": daily_data["temperature_2m_min"],
        "avg_temp": daily_data["temperature_2m_mean"],
        "rainfall": daily_data["precipitation_sum"],
        "wind_speed": daily_data["wind_speed_10m_max"]
    })
    
    # Process Hourly data
    hourly_data = data["hourly"]
    hourly_df = pd.DataFrame({
        "time": pd.to_datetime(hourly_data["time"]),
        "humidity": hourly_data["relative_humidity_2m"],
        "pressure": hourly_data["surface_pressure"],
        "cloud_cover": hourly_data["cloud_cover"]
    })
    
    # Aggregate hourly means to daily
    hourly_df["date"] = hourly_df["time"].dt.floor("D")
    daily_means = hourly_df.groupby("date")[["humidity", "pressure", "cloud_cover"]].mean().reset_index()
    
    # Merge daily and aggregated features
    merged_df = pd.merge(daily_df, daily_means, on="date")
    
    features = [
        'date',
        'max_temp','min_temp','avg_temp',
        'humidity','rainfall','wind_speed',
        'pressure','cloud_cover'
    ]
    
    final_df = merged_df[features].copy()
    
    # Extract seasonal cyclic logic for new CNN-LSTM features
    final_df['date_parsed'] = pd.to_datetime(final_df['date'])
    final_df['month'] = final_df['date_parsed'].dt.month
    final_df['day_of_year'] = final_df['date_parsed'].dt.dayofyear
    final_df['month_sin'] = np.sin(2 * np.pi * final_df['month'] / 12.0)
    final_df['month_cos'] = np.cos(2 * np.pi * final_df['month'] / 12.0)
    final_df['day_sin'] = np.sin(2 * np.pi * final_df['day_of_year'] / 365.25)
    final_df['day_cos'] = np.cos(2 * np.pi * final_df['day_of_year'] / 365.25)
    
    final_df = final_df.drop(columns=['date_parsed', 'month', 'day_of_year'])
    
    # Return exactly N days (API might sometimes return an extra day due to timezone boundaries)
    return final_df.tail(days)

@st.cache_data(ttl=300, show_spinner=False)
def get_current_weather(
    city: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict:
    """
    Fetches current weather from Open-Meteo Forecast API.
    Returns actual current conditions (not historical).
    Cached for 5 minutes (300 seconds).
    """
    if latitude is None or longitude is None:
        if city not in CITY_COORDINATES:
            raise ValueError(f"Coordinates for city '{city}' not found.")
        lat = CITY_COORDINATES[city]["lat"]
        lon = CITY_COORDINATES[city]["lon"]
    else:
        lat = latitude
        lon = longitude

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "weather_code",
            "wind_speed_10m",
            "cloud_cover",
            "pressure_msl",
            "apparent_temperature",
        ],
        "timezone": "auto",
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()

    current = data["current"]

    return {
        "current_temp": current["temperature_2m"],
        "feels_like": current["apparent_temperature"],
        "humidity": current["relative_humidity_2m"],
        "precipitation": current["precipitation"],
        "weather_code": current["weather_code"],
        "wind_speed": current["wind_speed_10m"],
        "cloud_cover": current["cloud_cover"],
        "pressure": current["pressure_msl"],
        "timezone": data["timezone"],
        "time": current["time"],
    }


def get_weather_description(weather_code: int) -> str:
    """
    Convert WMO weather code to human-readable description.
    """
    weather_descriptions = {
        0: "Clear Sky",
        1: "Mainly Clear",
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Depositing Rime Fog",
        51: "Light Drizzle",
        53: "Moderate Drizzle",
        55: "Dense Drizzle",
        56: "Light Freezing Drizzle",
        57: "Dense Freezing Drizzle",
        61: "Slight Rain",
        63: "Moderate Rain",
        65: "Heavy Rain",
        66: "Light Freezing Rain",
        67: "Heavy Freezing Rain",
        71: "Slight Snowfall",
        73: "Moderate Snowfall",
        75: "Heavy Snowfall",
        77: "Snow Grains",
        80: "Slight Rain Showers",
        81: "Moderate Rain Showers",
        82: "Violent Rain Showers",
        85: "Slight Snow Showers",
        86: "Heavy Snow Showers",
        95: "Thunderstorm",
        96: "Thunderstorm with Slight Hail",
        99: "Thunderstorm with Heavy Hail",
    }
    return weather_descriptions.get(weather_code, "Unknown")


if __name__ == "__main__":
    df = get_live_weather_data("Hyderabad")
    print(df.shape)
    print(df.tail())
    print("\nCurrent weather:")
    current = get_current_weather("Hyderabad")
    print(current)
