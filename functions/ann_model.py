import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


class KerasBinaryClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper compativel com scikit-learn para classificacao binaria com Keras.

    A classe implementa a interface basica de estimadores do scikit-learn,
    permitindo usar uma rede neural Keras em pipelines, validacao cruzada e
    rotinas de busca de hiperparametros.
    """

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
        """Inicializa os hiperparametros do classificador.

        Args:
            input_dim (int): Numero de atributos de entrada esperados pela
                primeira camada densa.
            hidden_units (int | list | tuple, optional): Quantidade de unidades
                em cada camada oculta. Um inteiro cria uma unica camada oculta.
                Padrao: `(8, 8, 8)`.
            dropout_rate (float, optional): Taxa de dropout aplicada apos cada
                camada oculta. Padrao: 0.1.
            learning_rate (float, optional): Taxa de aprendizado usada pelo
                otimizador Adam. Padrao: 0.001.
            epochs (int, optional): Numero de epocas usado no `model.fit`.
                Padrao: 30.
            batch_size (int, optional): Tamanho do batch usado no `model.fit`.
                Padrao: 32.
            verbose (int, optional): Nivel de verbosidade passado aos metodos
                de treino e predicao do Keras. Padrao: 0.
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
        """Normaliza a configuracao das camadas ocultas para uma tupla.

        Returns:
            tuple: Quantidade de unidades em cada camada oculta.

        Raises:
            TypeError: Se `hidden_units` nao for inteiro, lista ou tupla.
        """
        if isinstance(self.hidden_units, int):
            return (self.hidden_units,)

        if isinstance(self.hidden_units, (list, tuple)):
            return tuple(self.hidden_units)

        raise TypeError(
            "hidden_units must be an int, list, or tuple of integers."
        )

    def _build_model(self):
        """Cria e compila o modelo sequencial Keras usado pelo classificador.

        Returns:
            tf.keras.Sequential: Modelo binario compilado com `binary_crossentropy`
            e metrica de acuracia.
        """
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
        """Treina o classificador neural.

        Args:
            X (array-like): Matriz de atributos de treino.
            y (array-like): Rotulos binarios do alvo.

        Returns:
            KerasBinaryClassifier: Instancia treinada do estimador.
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
        """Prediz rotulos binarios para as amostras informadas.

        Args:
            X (array-like): Matriz de atributos usada na predicao.

        Returns:
            np.ndarray: Vetor unidimensional com classes previstas, usando 0.5
            como limiar de probabilidade.
        """
        proba = self.model_.predict(X, verbose=0)
        return (proba > 0.5).astype(int).ravel()

    def predict_proba(self, X):
        """Prediz probabilidades das classes para as amostras informadas.

        Args:
            X (array-like): Matriz de atributos usada na predicao.

        Returns:
            np.ndarray: Matriz com duas colunas contendo as probabilidades das
            classes 0 e 1, respectivamente.
        """
        proba = self.model_.predict(X, verbose=0)
        return np.hstack([1 - proba, proba])



class KerasRegressor(BaseEstimator, RegressorMixin):
    """Wrapper compativel com scikit-learn para regressao com Keras.

    A classe permite alternar entre um modelo linear simples e uma rede MLP
    nao linear, mantendo a interface `fit` e `predict` esperada pelo
    scikit-learn.
    """

    def __init__(
        self,
        input_dim,
        hidden_units=(32, 16),
        dropout_rate=0.1,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        verbose=0,
        nonlinear=True  # True = rede neural não linear | False = regressão linear
    ):
        """Inicializa os hiperparametros do regressor.

        Args:
            input_dim (int): Numero de atributos de entrada esperados pelo
                modelo.
            hidden_units (int | list | tuple, optional): Quantidade de unidades
                em cada camada oculta quando `nonlinear=True`. Um inteiro cria
                uma unica camada oculta. Padrao: `(64, 32)`.
            dropout_rate (float, optional): Taxa de dropout aplicada apos cada
                camada oculta quando `nonlinear=True`. Padrao: 0.1.
            learning_rate (float, optional): Taxa de aprendizado usada pelo
                otimizador Adam. Padrao: 0.001.
            epochs (int, optional): Numero de epocas usado no `model.fit`.
                Padrao: 50.
            batch_size (int, optional): Tamanho do batch usado no `model.fit`.
                Padrao: 32.
            verbose (int, optional): Nivel de verbosidade passado aos metodos
                de treino e predicao do Keras. Padrao: 0.
            nonlinear (bool, optional): Se True, cria uma MLP; se False, cria
                uma unica camada de saida linear. Padrao: True.
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
        """Normaliza a configuracao das camadas ocultas para uma tupla.

        Returns:
            tuple: Quantidade de unidades em cada camada oculta.

        Raises:
            TypeError: Se `hidden_units` nao for inteiro, lista ou tupla.
        """
        if isinstance(self.hidden_units, int):
            return (self.hidden_units,)

        if isinstance(self.hidden_units, (list, tuple)):
            return tuple(self.hidden_units)

        raise TypeError(
            "hidden_units must be an int, list, or tuple of integers."
        )

    def _build_model(self):
        """Cria e compila o modelo Keras usado pelo regressor.

        Returns:
            tf.keras.Sequential: Modelo de regressao compilado com perda `mse`
            e metrica `mae`.
        """
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
        """Treina o regressor neural.

        Args:
            X (array-like): Matriz de atributos de treino.
            y (array-like): Valores continuos do alvo.

        Returns:
            KerasRegressor: Instancia treinada do estimador.
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
        """Prediz valores continuos para as amostras informadas.

        Args:
            X (array-like): Matriz de atributos usada na predicao.

        Returns:
            np.ndarray: Vetor unidimensional com os valores previstos.
        """
        preds = self.model_.predict(X, verbose=0)
        return preds.ravel()


if __name__ == "__main__":
   print("ann model carregado.")
