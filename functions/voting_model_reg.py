"""Funcoes auxiliares para preparar modelos base de um ensemble voting regressor.

O modulo reutiliza as configuracoes definidas em `single_model_reg.py` para
executar buscas de hiperparametros em regressores candidatos e fornecer
instancias de modelos base que podem ser usadas em um `VotingRegressor`.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline

try:
    from single_model_reg import SingleModelOrchestrator
except ImportError:
    from functions.single_model_reg import SingleModelOrchestrator


def voting_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    cv=3,
    scoring="neg_root_mean_squared_error",
    model_names=None,
    search_type="grid",
    n_iter=50,
):
    """Executa busca de hiperparametros para candidatos de um voting regressor.

    Para cada regressor configurado em `single_model_reg.py`, cria um pipeline
    simples, executa `GridSearchCV` ou `RandomizedSearchCV` e retorna os
    melhores parametros encontrados com os prefixos do pipeline removidos.

    Args:
        X_train (pd.DataFrame): Matriz de atributos de treino.
        y_train (pd.DataFrame): Alvo de treino alinhado a `X_train`.
        cv (int, optional): Numero de folds usado na validacao cruzada.
            Padrao: 3.
        scoring (str, optional): Metrica usada pelo `GridSearchCV`.
            Padrao: `"neg_root_mean_squared_error"`.
        model_names (list, optional): Lista de nomes de modelos registrados em
            `SingleModelOrchestrator`. Se None, usa todos os modelos
            disponiveis. Padrao: None.
        search_type (str, optional): Tipo de busca de hiperparametros. Aceita
            `"grid"` para `GridSearchCV` ou `"randomized"` para
            `RandomizedSearchCV`. Padrao: `"grid"`.
        n_iter (int, optional): Numero de combinacoes amostradas quando
            `search_type="randomized"`. Padrao: 50.

    Returns:
        dict: Dicionario indexado pelo nome do modelo, contendo os melhores
        parametros encontrados para cada regressor.

    Raises:
        ValueError: Se `search_type` nao for `"grid"` nem `"randomized"`.
    """

    if search_type not in {"grid", "randomized"}:
        raise ValueError("search_type deve ser 'grid' ou 'randomized'.")

    orchestrator = SingleModelOrchestrator()
    selected_models = model_names or orchestrator.model_methods
    dict_best_model_params = {}

    for model_name in selected_models:
        config = orchestrator.apply(model_name)
        model = config["model"]

        if search_type == "grid":
            params = config.get("models_gs")

            if not params:
                continue

            step_name = next(iter(params))
            search_params = params[step_name][1]
            search = GridSearchCV(
                make_pipeline(model),
                scoring=scoring,
                param_grid=search_params,
                cv=cv,
                n_jobs=-1,
            )

        else:
            step_name = model.__class__.__name__.lower()
            param_distributions = config.get("param_distributions")

            if not param_distributions:
                continue

            search_params = _prefix_params(step_name, param_distributions)
            search = RandomizedSearchCV(
                make_pipeline(model),
                scoring=scoring,
                param_distributions=search_params,
                cv=cv,
                n_iter=n_iter,
                random_state=23,
                n_jobs=-1,
                refit=True,
            )

        search_fit = search.fit(X_train, y_train)

        dict_best_params = {}
        prefix = f"{step_name}__"
        for key, value in search_fit.best_params_.items():
            dict_best_params[key.replace(prefix, "", 1)] = value

        dict_best_model_params[model_name] = dict_best_params

    return dict_best_model_params


def _prefix_params(step_name, params):
    """Adiciona prefixo de pipeline a parametros de busca.

    Args:
        step_name (str): Nome do passo do pipeline.
        params (dict | list): Parametros ou distribuicoes de parametros.

    Returns:
        dict | list: Parametros com prefixo `<step_name>__`.
    """
    if isinstance(params, list):
        return [{f"{step_name}__{key}": value for key, value in grid.items()} for grid in params]

    return {f"{step_name}__{key}": value for key, value in params.items()}


def models(model_names=None):
    """Retorna instancias padrao dos regressores candidatos ao voting.

    Args:
        model_names (list, optional): Lista de nomes de modelos registrados em
            `SingleModelOrchestrator`. Se None, usa todos os modelos
            disponiveis. Padrao: None.

    Returns:
        dict: Dicionario com regressores base indexados pelo nome do modelo.
    """

    orchestrator = SingleModelOrchestrator()
    selected_models = model_names or orchestrator.model_methods

    return {
        model_name: orchestrator.apply(model_name)["model"]
        for model_name in selected_models
    }


if __name__ == "__main__":
    print("Voting Model Regression carregado.")
