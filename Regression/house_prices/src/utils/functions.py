from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class MedianByYTransformer(BaseEstimator, TransformerMixin):
    """
    Substitui valores ausentes (NaN) em colunas numéricas pelas medianas
    calculadas por classe, onde a classe vem de y_train (e não de X_train).

    Parâmetros:
    -----------
    feature_cols : list[str]
        Lista de colunas numéricas nas quais aplicar a imputação.
    """
    def __init__(self, feature_cols):
        self.feature_cols = feature_cols
        self.medians_ = None

    def fit(self, X, y):
        # Verificações
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X deve ser um pandas DataFrame")
        if y is None:
            raise ValueError("y não pode ser None")
        if len(y) != len(X):
            raise ValueError("X e y devem ter o mesmo número de linhas")

        # Monta um DataFrame temporário para agrupar por y
        df_temp = X.copy()
        df_temp["_target_"] = y.values if isinstance(y, pd.Series) else y

        # Calcula as medianas por classe e coluna
        self.medians_ = (
            df_temp.groupby("_target_")[self.feature_cols].median().to_dict(orient="index")
        )

        return self

    def transform(self, X):
        X = X.copy()

        if self.medians_ is None:
            raise RuntimeError("O transformer deve ser ajustado (fit) antes de transformar.")

        # Como não temos y no transform, usamos a média global como fallback
        global_medians = pd.DataFrame(self.medians_).median(axis=1).to_dict()

        # Não sabemos as classes durante transform, então só aplicamos medianas globais
        # ou você pode armazenar as classes vistas e aplicar de acordo com um y fornecido.
        for col in self.feature_cols:
            X[col] = X[col].fillna(global_medians.get(col, np.nan))

        return X
