"""Funcoes auxiliares para validacao cruzada de modelos.

Este modulo concentra rotinas de K-Fold e Stratified K-Fold para estimadores
compativeis com scikit-learn. As funcoes retornam os resultados por fold em
formato tabular, incluindo metricas de treino/validacao quando disponiveis,
nome do modelo, tipo de experimento, metrica usada e timestamp da execucao.
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
    """Executa validacao cruzada estratificada com parametros otimizados.

    Aplica os hiperparametros informados ao estimador em `model_config`,
    treina um pipeline simples em cada fold estratificado e calcula a acuracia
    no conjunto de validacao correspondente. Predicoes e probabilidades sao
    calculadas durante o loop, mas apenas a acuracia por fold e retornada.

    Args:
        X_train (pd.DataFrame): Matriz de atributos usada no treino.
        y_train (pd.DataFrame): Vetor ou matriz de rotulos alinhado a `X_train`.
        model_config (dict): Configuracao do modelo. Deve conter as chaves
            `model`, com um estimador compativel com scikit-learn, e
            `model_name`, com o nome usado no resultado.
        best_model_params (dict): Hiperparametros aplicados ao modelo via
            `set_params` antes da validacao cruzada.
        n_splits (int, optional): Numero de folds estratificados. Padrao: 5.
        random_state (int, optional): Semente usada no embaralhamento dos
            folds. Padrao: 23.

    Returns:
        pd.DataFrame: Resultado por fold com acuracia, metrica usada, tipo de
        modelo, nome do modelo e timestamp da execucao.

    Side Effects:
        Altera `model_config["model"]` ao aplicar `best_model_params` com
        `set_params`.
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
    model_config:dict,
    score:str,
    n_splits:int=5, 
    model_type:str='single_model',
    shuffle:bool=True):
    """Executa validacao cruzada com K-Fold e consolida os scores por fold.

    Usa `sklearn.model_selection.cross_validate` para calcular scores de treino
    e validacao em folds nao estratificados.

    Args:
        X_train (pd.DataFrame): Matriz de atributos usada na validacao cruzada.
        y_train (pd.DataFrame): Vetor ou matriz de rotulos alinhado a `X_train`.
        model: Estimador compativel com scikit-learn.
        model_config (dict): Configuracao contendo `model_name`, usado para
            identificar o modelo no resultado.
        score (str): Nome da metrica aceita por
            `sklearn.model_selection.cross_validate`.
        n_splits (int, optional): Numero de folds. Padrao: 5.
        model_type (str, optional): Rotulo do tipo de experimento registrado no
            resultado. Padrao: `"single_model"`.
        shuffle (bool, optional): Indica se os dados devem ser embaralhados
            antes da divisao em folds. Padrao: True.

    Returns:
        pd.DataFrame: Resultado por fold com `train_score`, `val_score`,
        metrica usada, tipo de modelo, nome do modelo e timestamp da execucao.
    """
    
    # kfolds cv
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=23)
    
    # search
    cv = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        scoring=score,
        return_train_score=True,
        cv=kf
    )
    
    cv_result = {'train_score':cv['train_score'], 'test_score':cv['test_score']}
    
    df_score = (pd.DataFrame
                .from_dict(cv_result, orient='columns')
                .reset_index()
                .rename(columns={'index':'fold'})
                .rename(columns={'test_score':'val_score'})
                )
    
    df_score = (df_score.assign(
        scoring=score,
        model_type = model_type,
        model = lambda x: model_config['model_name'], 
        timestamp = lambda x: datetime.now().isoformat())
                )
    
    return df_score

def cross_validate_StratifiedKFold(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    model_clf,
    model_config:dict,
    scoring:str,
    n_splits:int=5, 
    model_type:str='single_model',
    shuffle:bool=True):
    """Executa validacao cruzada estratificada e consolida os scores por fold.

    Usa `StratifiedKFold` para preservar a proporcao das classes em cada fold e
    `sklearn.model_selection.cross_validate` para calcular os scores de treino
    e validacao.

    Args:
        X_train (pd.DataFrame): Matriz de atributos usada na validacao cruzada.
        y_train (pd.DataFrame): Vetor ou matriz de rotulos alinhado a `X_train`.
        model_clf: Estimador classificador compativel com scikit-learn.
        model_config (dict): Configuracao contendo `model_name`, usado para
            identificar o modelo no resultado.
        scoring (str): Nome da metrica aceita por
            `sklearn.model_selection.cross_validate`.
        n_splits (int, optional): Numero de folds estratificados. Padrao: 5.
        model_type (str, optional): Rotulo do tipo de experimento registrado no
            resultado. Padrao: `"single_model"`.
        shuffle (bool, optional): Indica se os dados devem ser embaralhados
            antes da divisao em folds. Padrao: True.

    Returns:
        pd.DataFrame: Resultado por fold com `train_score`, `val_score`,
        metrica usada, tipo de modelo, nome do modelo e timestamp da execucao.
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
        model_type = model_type,
        model = lambda x: model_config['model_name'], 
        timestamp = lambda x: datetime.now().isoformat())
                )
    
    return df_score

if __name__ == "__main__":
   print("cross validate carregado.")
