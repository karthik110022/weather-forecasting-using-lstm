import joblib, os
_ROOT = os.path.abspath(os.getcwd())
_SCALER_PATH = os.path.join(_ROOT, "models", "scaler.pkl")
scaler = joblib.load(_SCALER_PATH)
print("Features:", scaler.feature_names_in_)
