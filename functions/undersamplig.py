"""Funcoes para aplicar tecnicas de undersampling em dados desbalanceados.

O modulo agrupa metodos da biblioteca imbalanced-learn para reduzir a classe
majoritaria ou limpar exemplos ambigueis em problemas de classificacao. Cada
funcao recebe dados de treino e retorna os conjuntos reamostrados.
"""

from imblearn.under_sampling import (
    TomekLinks,
    OneSidedSelection,
    EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours,
    NeighbourhoodCleaningRule,
    NearMiss,
    InstanceHardnessThreshold
    )
from sklearn.linear_model import LogisticRegression


def apply_oneSidedSelection(X_train, y_train):
    """Aplica undersampling com One Sided Selection.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """

    oss = OneSidedSelection(
    sampling_strategy='auto',
    random_state=23,
    n_neighbors=1,
    n_jobs=-1)

    X_resampled, y_resampled = oss.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_nearMissV1(X_train, y_train):
    """Aplica undersampling com NearMiss versao 1.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """

    nm1 = NearMiss(
    sampling_strategy='auto',
    version=1,
    n_neighbors=3,
    n_jobs=-1)

    X_resampled, y_resampled = nm1.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_nearMissV2(X_train, y_train):
    """Aplica undersampling com NearMiss versao 2.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """

    nm2 = NearMiss(
    sampling_strategy='auto',
    version=2,
    n_neighbors=3,
    n_jobs=-1)

    X_resampled, y_resampled = nm2.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_nearMissV3(X_train, y_train):
    """Aplica undersampling com NearMiss versao 3.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """

    nm3 = NearMiss(
    sampling_strategy='auto',
    version=3,
    n_neighbors=3,
    n_jobs=-1)

    X_resampled, y_resampled = nm3.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_editedNearestNeighbours(X_train, y_train):
    """Aplica undersampling com Edited Nearest Neighbours.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """
    enn  = EditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    n_jobs=-1)

    X_resampled, y_resampled = enn.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_neighbourhoodCleaningRule(X_train, y_train):
    """Aplica undersampling com Neighbourhood Cleaning Rule.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """
    nec  = NeighbourhoodCleaningRule(
    sampling_strategy='auto',
    n_neighbors=3,
    threshold_cleaning=0.5,
    n_jobs=-1)

    X_resampled, y_resampled = nec.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_tomekLinks(X_train, y_train):
    """Aplica undersampling com Tomek Links.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """

    tl = TomekLinks(
    sampling_strategy='auto',
    n_jobs=-1)

    X_resampled, y_resampled = tl.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_repeatedEditedNearestNeighbours(X_train, y_train):
    """Aplica undersampling com Repeated Edited Nearest Neighbours.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """

    renn  = RepeatedEditedNearestNeighbours(
    sampling_strategy='auto',
    n_neighbors=3,
    kind_sel='all',
    max_iter=100,
    n_jobs=-1)

    X_resampled, y_resampled = renn.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


def apply_instanceHardnessThreshold(X_train, y_train):
    """Aplica undersampling com Instance Hardness Threshold.

    Usa regressao logistica como estimador interno para identificar exemplos
    mais dificeis e selecionar as instancias mantidas apos a reamostragem.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Rotulos de treino.

    Returns:
        tuple: `X_resampled` e `y_resampled` apos o undersampling.
    """

    clf = LogisticRegression(random_state=32, max_iter = 1000)

    iht  = InstanceHardnessThreshold(
        sampling_strategy='auto',
        cv=3,
        estimator=clf,
        n_jobs=-1)

    X_resampled, y_resampled = iht.fit_resample(X_train, y_train)

    return X_resampled, y_resampled


class UnderSampligOrchestrator:
    """Orquestra a execucao das tecnicas de undersampling disponiveis."""

    def __init__(self):
        """Inicializa o mapa de metodos de undersampling."""
        self.methods = {
            "OneSidedSelection": apply_oneSidedSelection,
            "EditedNearestNeighbours": apply_editedNearestNeighbours,
            "TomekLinks": apply_tomekLinks,
            "RepeatedEditedNearestNeighbours": apply_repeatedEditedNearestNeighbours,
            "NeighbourhoodCleaningRule": apply_neighbourhoodCleaningRule,
            "NearMissV1": apply_nearMissV1,
            "NearMissV2": apply_nearMissV2,
            "NearMissV3": apply_nearMissV3,
            "InstanceHardnessThreshold": apply_instanceHardnessThreshold
        }

        self.under_samplig_methods = list(self.methods.keys())

    def apply(self, method_name, X_train, y_train):
        """Executa o metodo de undersampling escolhido.

        Args:
            method_name (str): Nome do metodo a ser aplicado.
            X_train (array-like): Matriz de atributos de treino.
            y_train (array-like): Rotulos de treino.

        Returns:
            tuple: `X_resampled` e `y_resampled` apos o balanceamento.

        Raises:
            ValueError: Se `method_name` nao estiver registrado no
            orquestrador.
        """
        if method_name not in self.methods:
            raise ValueError(f"Metodo '{method_name}' nao reconhecido. Escolha entre: {self.under_samplig_methods}")

        # Obtem a funcao correspondente e a executa
        sampling_func = self.methods[method_name]
        return sampling_func(X_train, y_train)


if __name__ == "__main__":
   print("UnderSampling carregado.")
