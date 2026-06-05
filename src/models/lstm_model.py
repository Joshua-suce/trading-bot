# LSTM neural network — optional sequence model for price-direction prediction
from typing import Optional, Tuple

import numpy as np
from loguru import logger

# TensorFlow is optional — gracefully degrade if not installed
try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential, load_model

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class LSTMPredictor:
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 50,
        lstm_units: int = 64,
        dropout: float = 0.2,
        model_path: Optional[str] = None,
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model_path = model_path
        self.model: Optional[Sequential] = None
        self.is_trained = False

    # Build a 2-layer LSTM architecture with dropout for regularisation
    def _build_model(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        self.model = Sequential(
            [
                Input(shape=(self.sequence_length, self.n_features)),
                LSTM(self.lstm_units, return_sequences=True),
                Dropout(self.dropout),
                LSTM(self.lstm_units // 2, return_sequences=False),
                Dropout(self.dropout),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])
        self.model.summary()

    # Convert flat feature matrix into (samples, sequence_length, n_features) sequences
    def prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple:
        X_seq, y_seq = [], []
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length : i])
            if y is not None:
                y_seq.append(y[i])
        if y is not None:
            return np.array(X_seq), np.array(y_seq)
        return np.array(X_seq), None

    # Train the LSTM on sequential data with optional validation and early stopping
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 64,
    ):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        self._build_model()
        if self.model is None:
            raise RuntimeError("LSTM model was not initialized")
        X_seq, y_seq = self.prepare_sequences(X_train, y_train)
        callbacks = [EarlyStopping(patience=10, restore_best_weights=True)]
        validation_data = None
        if X_val is not None and y_val is not None:
            Xv_seq, yv_seq = self.prepare_sequences(X_val, y_val)
            validation_data = (Xv_seq, yv_seq)
        self.model.fit(
            X_seq,
            y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
        )
        self.is_trained = True
        logger.info("LSTM training complete")

    # Predict next-period value for a given sequence
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model not trained yet")
        X_seq, _ = self.prepare_sequences(X)
        return self.model.predict(X_seq, verbose=0).flatten()

    # Save the trained Keras model to disk
    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("No model to save")
        self.model.save(path)
        logger.info(f"LSTM model saved to {path}")

    # Load a previously trained Keras model from disk
    def load(self, path: str):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is not installed")
        self.model = load_model(path)
        self.is_trained = True
        logger.info(f"LSTM model loaded from {path}")
