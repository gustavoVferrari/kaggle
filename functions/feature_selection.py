"""Funcoes para aplicar tecnicas de selecao e analise de features.

O modulo agrupa metodos estatisticos, filtros baseados em informacao mutua,
correlacao, remocao de variaveis constantes/duplicadas e seletores da
biblioteca feature-engine. As funcoes retornam rankings, matrizes ou listas de
features sugeridas para remocao, conforme a tecnica aplicada.
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.selection import (
    SelectBySingleFeaturePerformance,
    SmartCorrelatedSelection,
    DropConstantFeatures,
    DropDuplicateFeatures,
    MRMR
)


def apply_MRMR(X_train, y_train, method):

    """Aplica selecao de features com MRMR.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo usado para avaliar relevancia e redundancia.
        method (str): Metodo MRMR aceito por `feature_engine.selection.MRMR`.

    Returns:
        dict: Dicionario com os scores de relevancia e as features sugeridas
        para remocao.
    """

    sel = MRMR(
        method=method,
        random_state=23
    )

    sel.fit(X_train, y_train)

    return {"relevance": sel.relevance_, "features_to_drop": sel.features_to_drop_}


def apply_DropConstantFeatures(X_train, y_train, threshold):

    """Identifica features constantes ou quase constantes.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo recebido por compatibilidade com o
            orquestrador. Nao e usado pela funcao.
        threshold (float): Tolerancia usada pelo `DropConstantFeatures`.

    Returns:
        list: Features identificadas para remocao.
    """

    sel = DropConstantFeatures(
        tol=threshold,
        missing_values="raise"
    )

    sel.fit(X_train)

    return sel.features_to_drop_


def apply_DropDuplicateFeatures(X_train, y_train):

    """Identifica features duplicadas.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo recebido por compatibilidade com o
            orquestrador. Nao e usado pela funcao.

    Returns:
        list: Features identificadas para remocao.
    """

    sel = DropDuplicateFeatures(
        missing_values="raise"
    )

    sel.fit(X_train)

    return sel.features_to_drop_


def apply_MutualInformation_reg(X_train, y_train):

    """Calcula informacao mutua para problema de regressao.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo continuo usado no calculo.

    Returns:
        pd.Series: Scores de informacao mutua por feature, ordenados do maior
        para o menor.
    """

    mi = mutual_info_regression(X_train, y_train)
    mi = pd.Series(mi, name='mutual information')
    mi.index = X_train.columns
    mi.sort_values(ascending=False, inplace=True)

    return mi


def apply_MutualInformation_classif(X_train, y_train):

    """Calcula informacao mutua para problema de classificacao.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo categorico usado no calculo.

    Returns:
        pd.Series: Scores de informacao mutua por feature, ordenados do maior
        para o menor.
    """

    mi = mutual_info_classif(X_train, y_train)
    mi = pd.Series(mi, name='mutual information')
    mi.index = X_train.columns
    mi.sort_values(ascending=False, inplace=True)

    return mi


def apply_Anova(X_train, y_train):

    """Aplica ANOVA F-test nas variaveis numericas.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo categorico usado no teste.

    Returns:
        pd.Series: P-valores por feature numerica, ordenados do menor para o
        maior.
    """

    numerical_col = list(X_train.select_dtypes(include=['number']).columns)
    anova = f_classif(X_train[numerical_col], y_train)
    s = pd.Series(anova[1], index=numerical_col, name='Anova')
    s.sort_values(ascending=True, inplace=True)

    return s


def apply_QuiSquare(X_train, y_train):

    """Aplica teste qui-quadrado nas variaveis categoricas.

    Para cada coluna categorica, cria uma tabela de contingencia entre o alvo e
    a feature e calcula o p-valor do teste qui-quadrado.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo categorico usado no teste.

    Returns:
        pd.Series: P-valores por feature categorica, ordenados do menor para o
        maior.
    """

    categorical_col = list(X_train.select_dtypes(include=['category', 'object', 'bool']).columns)

    chi_ls = []

    for feature in categorical_col:
        print("Feature:", feature)
        # create contingency table
        arr1 = np.array((y_train.values.flatten()))
        arr2 = np.array(X_train[feature].values.flatten())
        c = pd.crosstab(arr1, arr2)
        # chi_test
        p_value = stats.chi2_contingency(c)[1]
        chi_ls.append(p_value)

    chi = pd.Series(chi_ls, index=categorical_col, name='QuiSquare')
    chi_sorted = chi.sort_values(ascending=True)

    return chi_sorted


def apply_PearsonCorrelation(X_train, y_train):

    """Calcula a matriz de correlacao de Pearson das variaveis numericas.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (array-like): Alvo recebido por compatibilidade com o
            orquestrador. Nao e usado pela funcao.

    Returns:
        pd.DataFrame: Matriz de correlacao de Pearson entre features numericas.
    """

    numerical_col = list(X_train.select_dtypes(include=['number']).columns)

    corr_matrix = X_train[numerical_col].corr(method='pearson')

    return corr_matrix


def apply_SmartCorrelatedSelection(X_train, y_train):

    """Seleciona features correlacionadas usando SmartCorrelatedSelection.

    Identifica grupos de variaveis numericas com correlacao de Pearson acima do
    limiar configurado e usa desempenho de modelo para decidir quais features
    devem ser removidas.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (pd.DataFrame | array-like): Alvo usado para treinar o estimador
            interno do seletor.

    Returns:
        dict: Dicionario com o primeiro grupo de features correlacionadas em
        `corr_feature` e as features sugeridas para remocao em `corr_2_drop`.
    """

    numerical_col = list(X_train.select_dtypes(include=['number']).columns)

    rf = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    # correlation selector
    sel = SmartCorrelatedSelection(
        variables=numerical_col,  # if none, selector examines all numerical variables
        method="pearson",
        threshold=0.8,
        missing_values="raise",
        selection_method="model_performance",
        estimator=rf,
        scoring="roc_auc",
        cv=3,
    )

    sel.fit(X_train, y_train.values.ravel())

    if not sel.correlated_feature_sets_:
        return {
            "corr_feature": [],
            "corr_2_drop": sel.features_to_drop_,
        }

    corr_features = list(sel.correlated_feature_sets_[0])

    return {
        "corr_feature": corr_features,
        "corr_2_drop": sel.features_to_drop_,
    }


class FeatureSelectionOrchestrator:
    """Orquestra a execucao dos metodos de selecao de features disponiveis."""

    def __init__(self):
        """Inicializa o mapa de metodos de selecao de features."""
        self.methods = {
            "QuiSquare": apply_QuiSquare,
            "Anova": apply_Anova,
            "MutualInformationClassif": apply_MutualInformation_classif,
            "MutualInformationReg": apply_MutualInformation_reg,
            "PearsonCorrelation": apply_PearsonCorrelation,
            "SmartCorrelatedSelection": apply_SmartCorrelatedSelection,
            "DropConstantFeatures": apply_DropConstantFeatures,
            "DropDuplicateFeatures": apply_DropDuplicateFeatures,
            "MRMR": apply_MRMR
        }

        feature_selection_methods = list(self.methods.keys())

    def apply(self, method_name, X_train, y_train, **kwargs):
        """Executa o metodo de selecao de features escolhido.

        Args:
            method_name (str): Nome do metodo a ser aplicado.
            X_train (pd.DataFrame): Matriz de atributos de treino.
            y_train (array-like): Alvo de treino.
            **kwargs: Argumentos adicionais repassados ao metodo escolhido.

        Returns:
            object: Resultado retornado pelo metodo de selecao executado.

        Raises:
            ValueError: Se `method_name` nao estiver registrado no orquestrador.
        """
        if method_name not in self.methods:
            raise ValueError(f"Metodo '{method_name}' nao reconhecido. Escolha entre: {self.feature_selection_methods}")

        # Obtem a funcao correspondente e a executa
        sampling_func = self.methods[method_name]
        return sampling_func(X_train, y_train, **kwargs)


def SelectSingleFeature(clf, metric, X_train: pd.DataFrame, y_train: pd.DataFrame, threshold=0.5):
    """Avalia o desempenho individual de cada feature com um estimador.

    Usa `SelectBySingleFeaturePerformance` para treinar o estimador com uma
    feature por vez e calcular a metrica informada via validacao cruzada.

    Args:
        clf: Estimador compativel com scikit-learn.
        metric (str): Metrica de scoring usada pelo seletor.
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (pd.DataFrame): Alvo de treino.
        threshold (float, optional): Limiar de desempenho usado pelo seletor.
            Padrao: 0.5.

    Returns:
        list: Lista com desempenho medio por feature e desvio-padrao do
        desempenho.
    """

    # set up the selector
    sel = SelectBySingleFeaturePerformance(
        variables=None,
        estimator=clf,
        scoring=metric,
        cv=3,
        threshold=threshold,
    )

    # find predictive features
    sel.fit(X_train, y_train)

    return [sel.feature_performance_, sel.feature_performance_std_]


if __name__ == "__main__":
   print("Feature Selection carregado.")
