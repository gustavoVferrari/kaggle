"""Utility for running stratified K-fold cross-validation on a configured model.

Fits the provided estimator with supplied hyperparameters using stratified folds,
and returns per-fold accuracy along with model metadata and a timestamp. Other
per-fold outputs (predictions, probabilities) are kept in-memory for potential
downstream analysis but only accuracy is persisted.
"""

import pandas as pd
from sklearn.pipeline import make_pipeline
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate


def cross_validate_stratified_gs(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_config,
    best_model_params: dict,
    n_splits: int = 5,
    random_state: int = 23,
):
    """Perform stratified K-fold CV and return per-fold accuracy scores.

    Args:
        X_train (pd.DataFrame): Feature matrix for training.
        y_train (pd.DataFrame): Target labels aligned with `X_train`.
        model_config (dict): Configuration containing at least:
            - `model`: sklearn-compatible estimator with `fit/predict/predict_proba`.
            - `model_name`: human-readable identifier for the model.
        best_model_params (dict): Hyperparameters to apply via `set_params` before fitting.
        n_splits (int, optional): Number of stratified folds. Defaults to 5.
        random_state (int, optional): Seed for reproducible shuffling. Defaults to 23.

    Returns:
        pd.DataFrame: DataFrame with columns `fold`, `accuracy`, `model`, `timestamp`
        summarizing accuracy per fold and metadata.

    Side Effects:
        Mutates `model_config['model']` by applying `best_model_params` with `set_params`.
    """

    model = model_config['model']

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    model.set_params(**best_model_params)

    dict_train_idx={}
    dict_val_idx={}
    dict_fit={}
    dict_predict={}
    dict_predict_proba={}
    dict_score={}


    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

            dict_train_idx[i] = [train_index]
            dict_val_idx[i] = [val_index]

            model_pipe_stf = make_pipeline(model)
            
            dict_fit[i] = model_pipe_stf.fit(
                X_train.iloc[dict_train_idx[i][0]], 
                y_train.iloc[dict_train_idx[i][0]])
            
            
            dict_predict[i] = dict_fit[i].predict(X_train.iloc[dict_val_idx[i][0]])

            # Keep probabilities for downstream metrics even though only accuracy is returned here.
            dict_predict_proba[i] = dict_fit[i].predict_proba(X_train.iloc[dict_val_idx[i][0]])[:,1]

            dict_score[i] = dict_fit[i].score(
                X_train.iloc[dict_val_idx[i][0]], 
                y_train.iloc[dict_val_idx[i][0]])
            
    df_score = pd.DataFrame(
        dict_score.items(), 
        columns=['fold', 'accuracy'])
    
    df_score['model'] = model_config['model_name']
    df_score['timestamp'] = datetime.now().isoformat()
    
    
    df_score = (df_score.assign(
        scoring='accuracy',
        model_type = "multi_model",
        model = lambda x: model_config['model_name'], 
        timestamp = lambda x: datetime.now().isoformat())
                )
    
    return df_score


def cross_validate_kfold(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model,
    scoring:str='accuracy',
    n_splits:int=5, 
    shuffle:bool=True):
    """Run basic K-Fold CV and return per-fold train/test scores.

    Args:
        X_train (pd.DataFrame): Feature matrix used in cross-validation.
        y_train (pd.DataFrame): Target labels aligned with `X_train` rows.
        model: Sklearn-compatible estimator exposing `fit` and `score`.
        scoring (str, optional): Scoring metric name understood by
            `sklearn.model_selection.cross_validate`. Defaults to ``'accuracy'``.
        n_splits (int, optional): Number of folds. Defaults to 5.
        shuffle (bool, optional): Whether to shuffle before splitting. Defaults to True.

    Returns:
        dict: Keys ``train_score`` and ``test_score`` containing arrays of per-fold
            scores produced by ``cross_validate``.
    """
    
    # kfolds cv
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=23)
    
    # search
    clf = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring=scoring,
        return_train_score=True,
        cv=kf
    )
    
    return {'train_score':clf['train_score'], 'test_score':clf['test_score']}

def cross_validate_StratifiedKFold(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_clf,
    model_config:dict,
    scoring:str='accuracy',
    n_splits:int=5, 
    shuffle:bool=True):
    """Run basic StratifiedKFold CV and return per-fold train/test scores.

    Args:
        X_train (pd.DataFrame): Feature matrix used in cross-validation.
        y_train (pd.DataFrame): Target labels aligned with `X_train` rows.
        model: Sklearn-compatible estimator exposing `fit` and `score`.
        scoring (str, optional): Scoring metric name understood by
            `sklearn.model_selection.cross_validate`. Defaults to ``'accuracy'``.
        n_splits (int, optional): Number of folds. Defaults to 5.
        shuffle (bool, optional): Whether to shuffle before splitting. Defaults to True.

    Returns:
        dict: Keys ``train_score`` and ``test_score`` containing arrays of per-fold
            scores produced by ``cross_validate``.
    """
    
    # kfolds cv
    kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=23)    
        
    # search
    clf = cross_validate(
        estimator=model_clf,
        X=X_train,
        y=y_train,
        scoring=scoring,
        return_train_score=True,
        cv=kf
    )
    
    cv_result = {'train_score':clf['train_score'], 'test_score':clf['test_score']}
    
    df_score = (pd.DataFrame
                .from_dict(cv_result, orient='columns')
                .reset_index()
                .rename(columns={'index':'fold'})
                .rename(columns={'test_score':'val_score'})
                )
    
    df_score = (df_score.assign(
        scoring=scoring,
        model_type = "single_model",
        model = lambda x: model_config['model_name'], 
        timestamp = lambda x: datetime.now().isoformat())
                )
    
    return df_score

if __name__ == "__main__":
   print("cross validate carregado.")
