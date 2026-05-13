import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for a Keras binary classifier."""

    def __init__(
        self,
        input_dim,
        hidden_units=(8,8,8),
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=30,
        batch_size=32,
        verbose=0
    ):
        """Initialize the classifier hyperparameters.

        Parameters
        ----------
        input_dim : int
            Number of input features expected by the first dense layer.
        hidden_units : int, list, or tuple, default=(8, 8, 8)
            Number of units in each hidden layer. An integer creates a single
            hidden layer.
        dropout_rate : float, default=0.1
            Dropout rate applied after each hidden layer.
        learning_rate : float, default=0.001
            Learning rate used by the Adam optimizer.
        epochs : int, default=30
            Number of training epochs passed to ``model.fit``.
        batch_size : int, default=32
            Batch size passed to ``model.fit``.
        verbose : int, default=0
            Verbosity level passed to Keras training and prediction methods.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model_ = None

    def _normalize_hidden_units(self):
        """Return hidden layer sizes as a tuple.

        Raises
        ------
        TypeError
            If ``hidden_units`` is not an integer, list, or tuple.
        """
        if isinstance(self.hidden_units, int):
            return (self.hidden_units,)

        if isinstance(self.hidden_units, (list, tuple)):
            return tuple(self.hidden_units)

        raise TypeError(
            "hidden_units must be an int, list, or tuple of integers."
        )

    def _build_model(self):
        """Build and compile the underlying Keras sequential model."""
        hidden_units = self._normalize_hidden_units()

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(
            hidden_units[0],
            activation="relu",
            kernel_initializer="he_normal",
            input_shape=(self.input_dim,)
        ))

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))

        for units in hidden_units[1:]:
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
        """Fit the neural network classifier.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like
            Binary target values.

        Returns
        -------
        KerasBinaryClassifier
            Fitted estimator instance.
        """
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
        """Predict binary class labels for the input samples.

        Parameters
        ----------
        X : array-like
            Feature matrix to predict.

        Returns
        -------
        numpy.ndarray
            One-dimensional array with predicted labels, using 0.5 as the
            probability threshold.
        """
        proba = self.model_.predict(X, verbose=0)
        return (proba > 0.5).astype(int).ravel()

    def predict_proba(self, X):
        """Predict class probabilities for the input samples.

        Parameters
        ----------
        X : array-like
            Feature matrix to predict.

        Returns
        -------
        numpy.ndarray
            Two-column array with probabilities for classes 0 and 1.
        """
        proba = self.model_.predict(X, verbose=0)
        return np.hstack([1 - proba, proba])



class KerasRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn compatible wrapper for a Keras regression model."""

    def __init__(
        self,
        input_dim,
        hidden_units=(64, 32),
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        verbose=0,
        nonlinear=True  # True = rede neural não linear | False = regressão linear
    ):
        """Initialize the regressor hyperparameters.

        Parameters
        ----------
        input_dim : int
            Number of input features expected by the model.
        hidden_units : int, list, or tuple, default=(64, 32)
            Number of units in each hidden layer when ``nonlinear`` is True.
            An integer creates a single hidden layer.
        dropout_rate : float, default=0.1
            Dropout rate applied after each hidden layer when ``nonlinear`` is
            True.
        learning_rate : float, default=0.001
            Learning rate used by the Adam optimizer.
        epochs : int, default=50
            Number of training epochs passed to ``model.fit``.
        batch_size : int, default=32
            Batch size passed to ``model.fit``.
        verbose : int, default=0
            Verbosity level passed to Keras training and prediction methods.
        nonlinear : bool, default=True
            If True, builds a multilayer perceptron. If False, builds a single
            linear output layer.
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.nonlinear = nonlinear
        self.model_ = None

    def _normalize_hidden_units(self):
        """Return hidden layer sizes as a tuple.

        Raises
        ------
        TypeError
            If ``hidden_units`` is not an integer, list, or tuple.
        """
        if isinstance(self.hidden_units, int):
            return (self.hidden_units,)

        if isinstance(self.hidden_units, (list, tuple)):
            return tuple(self.hidden_units)

        raise TypeError(
            "hidden_units must be an int, list, or tuple of integers."
        )

    def _build_model(self):
        """Build and compile the underlying Keras regression model."""
        model = tf.keras.Sequential()

        # =========================================================
        # MODELO LINEAR
        # =========================================================
        if not self.nonlinear:
            model.add(
                tf.keras.layers.Dense(
                    1,
                    activation="linear",
                    input_shape=(self.input_dim,)
                )
            )

        # =========================================================
        # MODELO NÃO LINEAR (MLP)
        # =========================================================
        else:
            hidden_units = self._normalize_hidden_units()

            model.add(
                tf.keras.layers.Dense(
                    hidden_units[0],
                    activation="relu",
                    kernel_initializer="he_normal",
                    input_shape=(self.input_dim,)
                )
            )

            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(self.dropout_rate))

            for units in hidden_units[1:]:
                model.add(
                    tf.keras.layers.Dense(
                        units,
                        activation="relu"
                    )
                )

                model.add(tf.keras.layers.BatchNormalization())
                model.add(tf.keras.layers.Dropout(self.dropout_rate))

            # saída contínua para regressão
            model.add(
                tf.keras.layers.Dense(
                    1,
                    activation="linear"
                )
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss="mse",
            metrics=["mae"]
        )

        return model

    def fit(self, X, y):
        """Fit the neural network regressor.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like
            Continuous target values.

        Returns
        -------
        KerasRegressor
            Fitted estimator instance.
        """
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
        """Predict continuous target values for the input samples.

        Parameters
        ----------
        X : array-like
            Feature matrix to predict.

        Returns
        -------
        numpy.ndarray
            One-dimensional array with predicted target values.
        """
        preds = self.model_.predict(X, verbose=0)
        return preds.ravel()
