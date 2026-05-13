"""Funcoes auxiliares para busca e selecao de hiperparametros de modelos.

O modulo concentra wrappers para `GridSearchCV` e `RandomizedSearchCV`,
incluindo variantes com K-Fold, Stratified K-Fold e busca em multiplas
configuracoes de modelos.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold, RandomizedSearchCV

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


def grid_search_models(X_train, y_train, models, scoring, cv):
    """Executa GridSearchCV para varias configuracoes de modelos.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (pd.DataFrame): Alvo de treino alinhado a `X_train`.
        models (dict): Mapeamento entre nome do modelo e tupla
            `(estimator, param_grid)`.
        scoring (str): Metrica de scoring aceita por `GridSearchCV`.
        cv: Estrategia de validacao cruzada, como `StratifiedKFold` ou `KFold`.

    Returns:
        dict: Melhores hiperparametros do ultimo modelo avaliado, com os
        prefixos do pipeline removidos.
    """

    for model in models.keys():

        classifier = models[model][0]
        params = models[model][1]

        print("Running: ", classifier)

        len_ = len([k.split('__')[0] for k in params.keys()][0])
        name_ = [k.split('__')[0] for k in params.keys()][0]

        grid_pipeline = make_pipeline(
            classifier)

        # Grid Search
        grid = GridSearchCV(
            grid_pipeline,
            scoring=scoring,
            param_grid=params,
            cv=cv,
            n_jobs=-1)

        grid_fit = grid.fit(X_train, y_train)

        # Best params
        dict_best_params = {}
        for k, v in grid_fit.best_params_.items():
            dict_best_params[k[len_+2:]] = v

    return dict_best_params


def grid_search_single_model_kfold(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model,
    param_grid: dict,
    n_splits: int = 5,
    metric: str = 'accuracy',
):
    """Executa GridSearchCV com K-Fold para um unico estimador.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (pd.DataFrame): Alvo de treino alinhado a `X_train`.
        model: Estimador compativel com scikit-learn.
        param_grid (dict): Grade de hiperparametros passada ao `GridSearchCV`.
        n_splits (int, optional): Numero de folds. Padrao: 5.
        metric (str, optional): Metrica de scoring. Padrao: `"accuracy"`.

    Returns:
        dict: Melhor conjunto de parametros encontrado pelo `GridSearchCV`.
    """

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=23
    )

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=metric,
        cv=kf,
        refit=True
    )

    search = gs.fit(X_train, y_train)

    return search.best_params_


def grid_search_single_model_StratifiedKFold(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model,
    param_grid: dict,
    n_splits: int = 3,
    scoring: str = 'accuracy',
):
    """Executa GridSearchCV com Stratified K-Fold para um unico estimador.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (pd.DataFrame): Alvo de treino alinhado a `X_train`.
        model: Estimador compativel com scikit-learn.
        param_grid (dict): Grade de hiperparametros passada ao `GridSearchCV`.
        n_splits (int, optional): Numero de folds estratificados. Padrao: 3.
        scoring (str, optional): Metrica de scoring. Padrao: `"accuracy"`.

    Returns:
        dict: Melhor conjunto de parametros encontrado pelo `GridSearchCV`.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=23
    )

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=skf,
        refit=True,
        verbose=1,
        n_jobs=-1
    )

    search = gs.fit(X_train, y_train)

    return search.best_params_


def randomized_single_model_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model,
    param_distributions: dict,
    n_splits: int = 5,
    scoring: str = 'accuracy',
    n_iter: int = 50
):
    """Executa RandomizedSearchCV para um unico estimador.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (pd.DataFrame): Alvo de treino alinhado a `X_train`.
        model: Estimador compativel com scikit-learn.
        param_distributions (dict): Distribuicoes ou listas de valores para
            amostragem dos hiperparametros.
        n_splits (int, optional): Numero de folds usado pelo `RandomizedSearchCV`.
            Padrao: 5.
        scoring (str, optional): Metrica de scoring. Padrao: `"accuracy"`.
        n_iter (int, optional): Numero de combinacoes amostradas. Padrao: 50.

    Returns:
        dict: Melhor conjunto de parametros encontrado pelo `RandomizedSearchCV`.
    """

    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        scoring=scoring,
        cv=n_splits,
        n_iter=n_iter,
        random_state=23,
        n_jobs=-1,
        refit=True
    )

    search = rs.fit(X_train, y_train)

    return search.best_params_


if __name__ == "__main__":
   print("Model Selection carregado.")
