import requests
import pandas as pd
from datetime import datetime, timedelta

# Open-Meteo requires latitude and longitude
CITY_COORDINATES = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Delhi": {"lat": 28.6139, "lon": 77.2090},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
    "Ahmedabad": {"lat": 23.0225, "lon": 72.5714},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Lucknow": {"lat": 26.8467, "lon": 80.9462},
    "Bhopal": {"lat": 23.2599, "lon": 77.4126}
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
    
    # Ensure correct column order for the model plus the date for plotting
    features = [
        'date',
        'max_temp','min_temp','avg_temp',
        'humidity','rainfall','wind_speed',
        'pressure','cloud_cover'
    ]
    
    final_df = merged_df[features]
    
    # Return exactly N days (API might sometimes return an extra day due to timezone boundaries)
    return final_df.tail(days)

if __name__ == "__main__":
    df = get_live_weather_data("Hyderabad")
    print(df.shape)
    print(df.tail())
