import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading model and scaler...")
model = tf.keras.models.load_model('models/lstm_model.h5', compile=False)
scaler = joblib.load('models/scaler.pkl')

print("Loading data...")
df = pd.read_csv('data/indian_cities_weather.csv')

# Rename columns to match model's expected format
df.columns = df.columns.str.strip()
df = df.rename(columns={
    'Temperature_Max (°C)': 'max_temp',
    'Temperature_Min (°C)': 'min_temp',
    'Temperature_Avg (°C)': 'avg_temp',
    'Humidity (%)': 'humidity',
    'Rainfall (mm)': 'rainfall',
    'Wind_Speed (km/h)': 'wind_speed',
    'Pressure (hPa)': 'pressure',
    'Cloud_Cover (%)': 'cloud_cover'
})
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['City', 'Date']).reset_index(drop=True)

features = ['max_temp','min_temp','avg_temp','humidity','rainfall','wind_speed','pressure','cloud_cover']
SEQ_LEN = 60

print("Processing data...")
scaled_arr = scaler.transform(df[features])
scaled_df = pd.DataFrame(scaled_arr, columns=features)
scaled_df['City'] = df['City'].values

X_eval, y_eval = [], []
for _, city_group in scaled_df.groupby('City'):
    city_values = city_group[features].values
    for i in range(len(city_values) - SEQ_LEN):
        X_eval.append(city_values[i:i+SEQ_LEN])
        y_eval.append(city_values[i+SEQ_LEN][[2, 4]])

X_eval = np.array(X_eval)
y_eval = np.array(y_eval)

split = int(len(X_eval) * 0.7)
X_test, y_test = X_eval[split:], y_eval[split:]

# Take only 5 samples for quick test
np.random.seed(42)
idx = np.random.choice(len(X_test), 5, replace=False)
X_sample = X_test[idx]
y_sample = y_test[idx]

print("Making predictions...")
pred_scaled = model.predict(X_sample, verbose=1)

y_true_full = np.zeros((len(y_sample), len(features)))
y_pred_full = np.zeros((len(pred_scaled), len(features)))
y_true_full[:, 2] = y_sample[:, 0]
y_true_full[:, 4] = y_sample[:, 1]
y_pred_full[:, 2] = pred_scaled[:, 0]
y_pred_full[:, 4] = pred_scaled[:, 1]

y_true_real = scaler.inverse_transform(pd.DataFrame(y_true_full, columns=features))
y_pred_real = scaler.inverse_transform(pd.DataFrame(y_pred_full, columns=features))

true_temp = y_true_real[:, 2]
pred_temp = y_pred_real[:, 2]
true_rain = y_true_real[:, 4]
pred_rain = y_pred_real[:, 4]

print()
print("=" * 60)
print("PREDICTED vs ACTUAL TEMPERATURE (in Celsius)")
print("=" * 60)
print(f"{'Sample':<8} {'Actual':<12} {'Predicted':<12} {'Difference':<12} {'Error %':<10}")
print("-" * 60)
for i in range(5):
    diff = pred_temp[i] - true_temp[i]
    err_pct = abs(diff) / true_temp[i] * 100
    print(f"{i+1:<8} {true_temp[i]:<12.2f} {pred_temp[i]:<12.2f} {diff:<+12.2f} {err_pct:<10.1f}%")

print()
print("=" * 60)
print("SUMMARY METRICS")
print("=" * 60)
print(f"Mean Absolute Error (MAE): {mean_absolute_error(true_temp, pred_temp):.2f} C")
print(f"Root Mean Square Error (RMSE): {np.sqrt(mean_squared_error(true_temp, pred_temp)):.2f} C")
print(f"R-squared (R2): {r2_score(true_temp, pred_temp):.4f}")
print(f"Average Error: {np.mean(np.abs(pred_temp - true_temp)):.2f} C")
print()
print("=" * 60)
print("RAINFALL PREDICTIONS (in mm)")
print("=" * 60)
for i in range(5):
    diff = pred_rain[i] - true_rain[i]
    print(f"Sample {i+1}: Actual={true_rain[i]:6.2f}  Predicted={pred_rain[i]:6.2f}  Diff={diff:+6.2f}")
