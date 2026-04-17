import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
print(joblib.load("/Users/karthikreddy/Downloads/weather-forecasting-using-lstm/models/scaler.pkl").feature_names_in_)
