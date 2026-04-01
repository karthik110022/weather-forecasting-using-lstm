"""
Script to fetch weather data for new Indian cities from Open-Meteo API
and append to the existing dataset.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# New cities to fetch
NEW_CITIES = {
    "Pune": {"lat": 18.5204, "lon": 73.8567, "state": "Maharashtra"},
    "Nagpur": {"lat": 21.1458, "lon": 79.0882, "state": "Maharashtra"},
    "Chandigarh": {"lat": 30.7333, "lon": 76.7794, "state": "Punjab"},
    "Kochi": {"lat": 9.9312, "lon": 76.2673, "state": "Kerala"},
    "Visakhapatnam": {"lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh"},
    "Patna": {"lat": 25.5941, "lon": 85.1376, "state": "Bihar"},
    "Vadodara": {"lat": 22.3072, "lon": 73.1812, "state": "Gujarat"},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
    "Indore": {"lat": 22.7196, "lon": 75.8577, "state": "Madhya Pradesh"},
    "Guwahati": {"lat": 26.1445, "lon": 91.7362, "state": "Assam"},
    "Thiruvananthapuram": {"lat": 8.5241, "lon": 76.9366, "state": "Kerala"},
    "Mysore": {"lat": 12.2958, "lon": 76.6394, "state": "Karnataka"},
    "Jodhpur": {"lat": 26.2389, "lon": 73.0243, "state": "Rajasthan"},
    "Ranchi": {"lat": 23.3441, "lon": 85.3095, "state": "Jharkhand"},
    "Shimla": {"lat": 31.1048, "lon": 77.1734, "state": "Himachal Pradesh"},
    "Dehradun": {"lat": 30.3165, "lon": 78.0322, "state": "Uttarakhand"},
    "Srinagar": {"lat": 34.0837, "lon": 74.7973, "state": "Jammu & Kashmir"},
    "Jammu": {"lat": 32.7266, "lon": 74.8570, "state": "Jammu & Kashmir"},
    "Leh": {"lat": 34.1526, "lon": 77.5771, "state": "Ladakh"},
    "Panaji": {"lat": 15.4909, "lon": 73.8278, "state": "Goa"},
    "Mangalore": {"lat": 12.9141, "lon": 74.8560, "state": "Karnataka"},
    "Surat": {"lat": 21.1702, "lon": 72.8311, "state": "Gujarat"},
}

def fetch_city_weather(city_name: str, lat: float, lon: float, state: str) -> pd.DataFrame:
    """Fetch 2 years of weather data for a city from Open-Meteo Archive API"""

    all_data = []

    # Split into two requests (1 year each) to avoid API limits
    date_ranges = [
        ("2024-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
    ]

    for start_date, end_date in date_ranges:
        url = "https://archive-api.open-meteo.com/v1/archive"

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "wind_speed_10m_max"
            ],
            "hourly": ["relative_humidity_2m", "surface_pressure", "cloud_cover"],
            "timezone": "Asia/Kolkata"
        }

        print(f"  Fetching {city_name} ({start_date} to {end_date})...")

        try:
            response = requests.get(url, params=params, timeout=30)
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
            all_data.append(merged_df)

            # Rate limiting - wait between requests
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error fetching {city_name}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    # Combine both years
    city_df = pd.concat(all_data, ignore_index=True)
    city_df = city_df.sort_values("date").reset_index(drop=True)

    # Add city and state columns
    city_df["city"] = city_name
    city_df["state"] = state

    # Add AQI columns (not available from Open-Meteo, using placeholders)
    city_df["AQI"] = 100  # Default moderate AQI
    city_df["AQI_Category"] = "Moderate"

    # Rename columns to match dataset format
    city_df = city_df.rename(columns={
        "date": "Date",
        "city": "City",
        "state": "State",
        "max_temp": "Temperature_Max (°C)",
        "min_temp": "Temperature_Min (°C)",
        "avg_temp": "Temperature_Avg (°C)",
        "humidity": "Humidity (%)",
        "rainfall": "Rainfall (mm)",
        "wind_speed": "Wind_Speed (km/h)",
        "pressure": "Pressure (hPa)",
        "cloud_cover": "Cloud_Cover (%)"
    })

    # Reorder columns to match original dataset
    city_df = city_df[[
        "Date", "City", "State",
        "Temperature_Max (°C)", "Temperature_Min (°C)", "Temperature_Avg (°C)",
        "Humidity (%)", "Rainfall (mm)", "Wind_Speed (km/h)",
        "AQI", "AQI_Category", "Pressure (hPa)", "Cloud_Cover (%)"
    ]]

    return city_df


def main():
    print("=" * 60)
    print("Fetching Weather Data for New Indian Cities")
    print("=" * 60)
    print(f"Total cities to fetch: {len(NEW_CITIES)}")
    print()

    all_new_data = []

    for i, (city_name, info) in enumerate(NEW_CITIES.items(), 1):
        print(f"\n[{i}/{len(NEW_CITIES)}] Fetching {city_name}...")
        city_df = fetch_city_weather(
            city_name,
            info["lat"],
            info["lon"],
            info["state"]
        )

        if not city_df.empty:
            print(f"  -> Got {len(city_df)} days of data")
            all_new_data.append(city_df)
        else:
            print(f"  -> Failed to fetch data")

        # Small delay between cities
        time.sleep(1)

    if all_new_data:
        # Combine all new city data
        new_data_combined = pd.concat(all_new_data, ignore_index=True)

        # Save to a separate file first
        new_data_combined.to_csv("data/new_cities_weather.csv", index=False)
        print(f"\nSaved new city data to data/new_cities_weather.csv")

        # Read existing data
        existing_df = pd.read_csv("data/indian_cities_weather.csv")

        # Combine with existing data
        combined_df = pd.concat([existing_df, new_data_combined], ignore_index=True)

        # Sort by date and city
        combined_df["Date"] = pd.to_datetime(combined_df["Date"])
        combined_df = combined_df.sort_values(["Date", "City"]).reset_index(drop=True)
        combined_df["Date"] = combined_df["Date"].dt.strftime("%Y-%m-%d")

        # Save combined data
        combined_df.to_csv("data/indian_cities_weather.csv", index=False)

        print(f"\n{'=' * 60}")
        print("SUCCESS!")
        print(f"{'=' * 60}")
        print(f"Original cities: 10")
        print(f"New cities added: {len(NEW_CITIES)}")
        print(f"Total cities now: {len(combined_df['City'].unique())}")
        print(f"Total records: {len(combined_df)}")
        print(f"\nUpdated file: data/indian_cities_weather.csv")
    else:
        print("\nFailed to fetch any data!")


if __name__ == "__main__":
    main()
