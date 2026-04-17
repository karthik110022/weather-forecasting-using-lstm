from src.api import get_live_weather_data
try:
    print(get_live_weather_data("Hyderabad", days=5))
except Exception as e:
    print("API Error:", str(e))
