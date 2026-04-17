import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D, MaxPooling1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from data_preprocessing import load_and_clean, scale_features, create_sequences
from utils import set_seed
from config import *

set_seed()

def build_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN Spatial Feature Extraction
    x = Conv1D(filters=64, kernel_size=3, activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    
    # Temporal Processing (Bidirectional for past/future context mapping within the 60 days)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)

    # Shared dense representation
    shared = Dense(32, activation='relu')(x)

    # Head 1: Temperature Output (Smooth regression)
    temp_dense = Dense(16, activation='relu')(shared)
    temp_out = Dense(1, name='temp_out')(temp_dense)

    # Head 2: Rainfall Output (Classification Probabilities 0 to 1)
    rain_dense = Dense(16, activation='relu')(shared)
    rain_out = Dense(1, activation='sigmoid', name='rain_out')(rain_dense)

    model = Model(inputs=inputs, outputs=[temp_out, rain_out])

    # Compile with distinct loss functions
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'temp_out': 'mse',
            'rain_out': 'binary_crossentropy'
        },
        metrics={
            'temp_out': ['mae'],
            'rain_out': ['accuracy']
        },
        loss_weights={
            'temp_out': 1.0,
            'rain_out': 1.2
        }
    )
    return model

def train():
    print("Loading and preparing dataset...")
    df = load_and_clean("data/indian_cities_weather.csv")
    scaled, _ = scale_features(df)
    X, y = create_sequences(scaled)

    split = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Keras multi-output targets mapping
    y_train_temp = y_train[:, 0]
    y_test_temp  = y_test[:, 0]
    
    # Cast Rainfall targets to Binary Classification [0, 1] strictly
    y_train_rain_class = (y_train[:, 1] > 0).astype(int)
    y_test_rain_class  = (y_test[:, 1] > 0).astype(int)
    
    y_train_dict = {'temp_out': y_train_temp, 'rain_out': y_train_rain_class}
    y_val_dict   = {'temp_out': y_test_temp,  'rain_out': y_test_rain_class}

    model = build_model((SEQ_LEN, X.shape[2]))
    # model.summary()

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'),
        ReduceLROnPlateau(patience=5, factor=0.5, monitor='val_loss')
    ]

    print(f"Began training on {X_train.shape[0]} samples...")
    model.fit(
        X_train, y_train_dict,
        epochs=60,
        batch_size=64,
        validation_data=(X_test, y_val_dict),
        callbacks=callbacks,
        verbose=1
    )

    # Save with explicit compile settings for better compatibility
    model.save("models/lstm_model.h5", include_optimizer=False)
    print("Model saved successfully to models/lstm_model.h5")

if __name__ == "__main__":
    train()