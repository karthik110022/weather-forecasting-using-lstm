import joblib, os

_ROOT = os.path.abspath(os.path.join(os.getcwd()))
_SCALER_PATH = os.path.join(_ROOT, "models", "scaler.pkl")

def evaluate_saved_model():
    scaler = joblib.load(_SCALER_PATH)
    print("Scaler features:", scaler.feature_names_in_)

evaluate_saved_model()
