import numpy as np
import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model("models/lstm_model.h5", compile=False)

print("Creating fake input...")
X_input = np.random.rand(1, 60, 8)

print("Trying model.predict...")
try:
    pred1 = model.predict(X_input, verbose=0)
    print("model.predict success:", pred1)
except Exception as e:
    print("model.predict failed:", e)

print("Trying direct model call...")
try:
    pred2 = model(X_input, training=False)
    print("Direct call success:", pred2.numpy())
except Exception as e:
    print("Direct call failed:", e)
