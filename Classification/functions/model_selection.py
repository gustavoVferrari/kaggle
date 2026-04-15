
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
    """Run grid search across multiple model configurations and return best params.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.DataFrame): Target labels aligned with `X_train`.
        models (dict): Mapping of model name to tuple ``(estimator, param_grid)``.
        scoring (str): Scoring metric understood by `GridSearchCV`.
        cv: Cross-validation splitter (e.g., `StratifiedKFold`, `KFold`).

    Returns:
        dict: Best hyperparameters for the last evaluated model (structure mirrors
            the estimator's `set_params` signature).
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
    X_train:pd.DataFrame, 
    y_train:pd.DataFrame, 
    model, 
    param_grid:dict,
    n_splits:int=5,
    metric:str='accuracy', 
    ):
    """Grid search with plain K-Fold CV for a single estimator.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.DataFrame): Target labels aligned with `X_train`.
        model: Sklearn-compatible estimator.
        param_grid (dict): Hyperparameter grid passed to `GridSearchCV`.
        n_splits (int, optional): Number of folds. Defaults to 5.
        metric (str, optional): Scoring metric. Defaults to ``'accuracy'``.

    Returns:
        dict: Best parameter set found by `GridSearchCV`.
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
    X_train:pd.DataFrame, 
    y_train:pd.DataFrame, 
    model, 
    param_grid:dict,
    n_splits:int=5,
    scoring:str='accuracy', 
    ):    
    """Grid search with stratified K-Fold CV for a single estimator.

    Args:
        X_train (pd.DataFrame): Training feature matrix.
        y_train (pd.DataFrame): Target labels aligned with `X_train`.
        model: Sklearn-compatible estimator.
        param_grid (dict): Hyperparameter grid passed to `GridSearchCV`.
        n_splits (int, optional): Number of folds. Defaults to 5.
        metric (str, optional): Scoring metric. Defaults to ``'accuracy'``.

    Returns:
        dict: Best parameter set found by `GridSearchCV`.
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
        refit=True
    )
    
    search = gs.fit(X_train, y_train)
    
    return search.best_params_

def randomized_single_model_grid_search(
    X_train:pd.DataFrame, 
    y_train:pd.DataFrame, 
    model, 
    param_distributions:dict,
    n_splits:int=5,
    scoring:str='accuracy',
    n_iter:int=50
):
    
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
   print("model selection loaded.")