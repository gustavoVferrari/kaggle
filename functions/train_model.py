import pickle
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier, VotingRegressor

try:
    from voting_model_clf import voting_model as search_voting_model_clf
    from voting_model_clf import models as voting_clf_models
    from voting_model_reg import voting_model as search_voting_model_reg
    from voting_model_reg import models as voting_reg_models
except ImportError:
    from functions.voting_model_clf import voting_model as search_voting_model_clf
    from functions.voting_model_clf import models as voting_clf_models
    from functions.voting_model_reg import voting_model as search_voting_model_reg
    from functions.voting_model_reg import models as voting_reg_models



def train_model(X_train, y_train, clf_model, best_model_params):       
    """Train a single estimator with tuned hyperparameters.

    Parameters
    ----------
    X_train : array-like or DataFrame
        Feature matrix used for fitting.
    y_train : array-like or Series
        Target labels aligned with `X_train`.
    clf_model : sklearn estimator
        Model instance to be fitted.
    best_model_params : dict
        Hyperparameters selected for `clf_model`.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted pipeline wrapping the configured estimator.
    """

    # apply best params    
    clf_model.set_params(**best_model_params)      
    
    pipeline = make_pipeline(             
            clf_model)
    
    clf = pipeline.fit(X_train, y_train)

    return clf

def train_voting_model_clf(
    X_train,
    y_train,
    models=None,
    best_models_params=None,
    model_names=None,
    search_type="grid",
    cv=3,
    scoring="roc_auc",
    n_iter=50,
    voting_type="soft",
):
    """Treina um ensemble VotingClassifier com modelos de `voting_model_clf.py`.

    Quando `best_models_params` nao e informado, executa primeiro a busca de
    hiperparametros definida em `voting_model_clf.voting_model`. Quando
    `models` nao e informado, carrega as instancias base com
    `voting_model_clf.models`.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Alvo de treino alinhado a `X_train`.
        models (dict, optional): Classificadores base indexados pelo nome do
            modelo. Se None, usa `voting_model_clf.models`. Padrao: None.
        best_models_params (dict, optional): Melhores parametros por modelo. Se
            None, executa a busca via `voting_model_clf.voting_model`.
            Padrao: None.
        model_names (list, optional): Subconjunto de modelos registrados em
            `SingleModelOrchestrator`. Se None, usa todos. Padrao: None.
        search_type (str, optional): Tipo de busca usado quando
            `best_models_params` e None. Aceita `"grid"` ou `"randomized"`.
            Padrao: `"grid"`.
        cv (int, optional): Numero de folds da busca de hiperparametros.
            Padrao: 3.
        scoring (str, optional): Metrica usada na busca. Padrao: `"roc_auc"`.
        n_iter (int, optional): Numero de iteracoes para busca randomized.
            Padrao: 50.
        voting_type (str, optional): Tipo de voting do `VotingClassifier`.
            Padrao: `"soft"`.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline treinado contendo o
        `VotingClassifier`.
    """

    if best_models_params is None:
        best_models_params = search_voting_model_clf(
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            model_names=model_names,
            search_type=search_type,
            n_iter=n_iter,
        )

    if models is None:
        models = voting_clf_models(model_names=model_names)

    estimators = []
    for model_name, model in models.items():
        model_params = best_models_params.get(model_name, {})
        estimators.append((model_name, model.set_params(**model_params)))

    if not estimators:
        raise ValueError("Nenhum classificador disponivel para treinar o VotingClassifier.")

    voting = VotingClassifier(
        estimators=estimators,
        voting=voting_type,
        flatten_transform=True
    )

    pipeline = make_pipeline(voting)
    voting_fitted = pipeline.fit(X_train, y_train)

    return voting_fitted


def train_voting_model(X_train, y_train, models=None, best_models_params=None, **kwargs):
    """Alias de compatibilidade para `train_voting_model_clf`."""
    return train_voting_model_clf(
        X_train,
        y_train,
        models=models,
        best_models_params=best_models_params,
        **kwargs
    )


def train_voting_model_reg(
    X_train,
    y_train,
    models=None,
    best_models_params=None,
    model_names=None,
    search_type="grid",
    cv=3,
    scoring="neg_root_mean_squared_error",
    n_iter=50,
):
    """Treina um ensemble VotingRegressor com modelos de `voting_model_reg.py`.

    Quando `best_models_params` nao e informado, executa primeiro a busca de
    hiperparametros definida em `voting_model_reg.voting_model`. Quando
    `models` nao e informado, carrega as instancias base com
    `voting_model_reg.models`.

    Args:
        X_train (array-like): Matriz de atributos de treino.
        y_train (array-like): Alvo de treino alinhado a `X_train`.
        models (dict, optional): Regressores base indexados pelo nome do
            modelo. Se None, usa `voting_model_reg.models`. Padrao: None.
        best_models_params (dict, optional): Melhores parametros por modelo. Se
            None, executa a busca via `voting_model_reg.voting_model`.
            Padrao: None.
        model_names (list, optional): Subconjunto de modelos registrados em
            `SingleModelOrchestrator`. Se None, usa todos. Padrao: None.
        search_type (str, optional): Tipo de busca usado quando
            `best_models_params` e None. Aceita `"grid"` ou `"randomized"`.
            Padrao: `"grid"`.
        cv (int, optional): Numero de folds da busca de hiperparametros.
            Padrao: 3.
        scoring (str, optional): Metrica usada na busca. Padrao:
            `"neg_root_mean_squared_error"`.
        n_iter (int, optional): Numero de iteracoes para busca randomized.
            Padrao: 50.

    Returns:
        sklearn.pipeline.Pipeline: Pipeline treinado contendo o
        `VotingRegressor`.
    """

    if best_models_params is None:
        best_models_params = search_voting_model_reg(
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            model_names=model_names,
            search_type=search_type,
            n_iter=n_iter,
        )

    if models is None:
        models = voting_reg_models(model_names=model_names)

    estimators = []
    for model_name, model in models.items():
        model_params = best_models_params.get(model_name, {})
        estimators.append((model_name, model.set_params(**model_params)))

    if not estimators:
        raise ValueError("Nenhum regressor disponivel para treinar o VotingRegressor.")

    voting = VotingRegressor(estimators=estimators)

    pipeline = make_pipeline(voting)
    voting_fitted = pipeline.fit(X_train, y_train)

    return voting_fitted
    
    
def save_model(model, path):
    """Persist a trained model to disk using pickle.

    Parameters
    ----------
    model : sklearn estimator or compatible object
        Trained model to serialize.
    path : str or PathLike
        Destination filepath for the pickle file.
    """

    with open(path, 'wb') as arquivo:
        pickle.dump(model, arquivo)
        
def save_pipeline(pipeline, path):
    """Persist a fitted pipeline to disk using pickle.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Fitted pipeline to serialize.
    path : str or PathLike
        Destination filepath for the pickle file.
    """

    with open(path, 'wb') as arquivo:
        pickle.dump(pipeline, arquivo)
    
if __name__ == "__main__":    
    print("Train Model carregado.")
