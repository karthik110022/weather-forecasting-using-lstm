import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

features8 = ['max_temp', 'min_temp', 'avg_temp', 'humidity', 'rainfall', 'wind_speed', 'pressure', 'cloud_cover']
features12 = features8 + ['month_sin', 'month_cos', 'day_sin', 'day_cos']
scaler8 = MinMaxScaler().fit(pd.DataFrame(np.random.rand(10, 8), columns=features8))
scaler12 = MinMaxScaler().fit(pd.DataFrame(np.random.rand(10, 12), columns=features12))

try:
    scaler8.transform(pd.DataFrame(np.random.rand(10, 12), columns=features12))
except Exception as e:
    print("Scaler8 predicting 12:", str(e))

try:
    scaler12.transform(pd.DataFrame(np.random.rand(10, 8), columns=features8))
except Exception as e:
    print("Scaler12 predicting 8:", str(e))

