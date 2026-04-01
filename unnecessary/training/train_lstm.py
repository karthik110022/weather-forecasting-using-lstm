import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_preprocessing import load_and_clean, scale_features, create_sequences
from utils import set_seed
from config import *

set_seed()

def build_model(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),

        LSTM(32),
        Dropout(0.3),

        Dense(16, activation='relu'),
        Dense(2)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    return model

def train():
    df = load_and_clean("data/indian_cities_weather.csv")
    scaled, _ = scale_features(df)
    X, y = create_sequences(scaled)

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = build_model((SEQ_LEN, X.shape[2]))

    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(patience=4)
    ]

    model.fit(
        X_train, y_train,
        epochs=60,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    # Save with explicit compile settings for better compatibility
    model.save("models/lstm_model.h5", include_optimizer=False)
    print("Model saved successfully")

if __name__ == "__main__":
    train()