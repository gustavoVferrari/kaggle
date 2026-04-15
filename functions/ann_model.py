import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim,
        hidden_units=(32, 16, 16),
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=30,
        batch_size=32,
        verbose=0
    ):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def _build_model(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(
            self.hidden_units[0],
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(self.input_dim,)
        ))

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))

        for units in self.hidden_units[1:]:
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.dropout_rate))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model

    def fit(self, X, y):
        self.model_ = self._build_model()

        self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )

        return self

    def predict(self, X):
        proba = self.model_.predict(X, verbose=0)
        return (proba > 0.5).astype(int).ravel()

    def predict_proba(self, X):
        proba = self.model_.predict(X, verbose=0)
        return np.hstack([1 - proba, proba])