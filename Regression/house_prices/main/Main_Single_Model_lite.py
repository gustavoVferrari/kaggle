import logging
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from utils.utils import to_jsonl
from functions.make_dataset import save_data
from functions.model_selection import grid_search_single_model_StratifiedKFold, randomized_single_model_grid_search
from functions.train_model import train_model, save_model
from functions.evaluate_model import evaluate_reg_model, MetricsOrchestrator
from functions.predict_model import make_prediction_reg
from functions.cross_validate import cross_validate_kfold
from functions.single_model_reg import SingleModelOrchestrator

from Regression.house_prices.src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def single_model_lite(
    pipeline_name:str, 
    model_name:str, 
    scoring:str, 
    grid_search_method:str,
    manual_params:dict=None,
    config:dict=None
    ):
    logger.info(
        "Iniciando pipeline de Machine Learning: model=%s, scoring=%s, pipeline=%s, grid_search_method=%s",
        model_name,
        scoring,
        pipeline_name,
        grid_search_method
    )


    # Get feature eng data
    logger.info("Carregando datasets de treino e validacao.")
    
    # Datasets X_train
    x_train_path = os.path.join(
        config['init_path'],
        config['data']['feature_eng'],
        f"X_train_feat_eng_{pipeline_name}.parquet"
    )
    X_train = pd.read_parquet(x_train_path)
   
    y_train_path = os.path.join(
        config['init_path'],
        config['data']['feature_eng'],
        f"y_train_feat_eng_{pipeline_name}.parquet"
    )
    y_train = pd.read_parquet(y_train_path)
    
    # Datasets Y_train
    x_val_path = os.path.join(
        config['init_path'],
        config['data']['feature_eng'],
        f"X_val_feat_eng_{pipeline_name}.parquet"
    )
    X_val = pd.read_parquet(x_val_path)
    
    y_val_path = os.path.join(
        config['init_path'],
        config['data']['feature_eng'],
        f"y_val_feat_eng_{pipeline_name}.parquet"
    )
    y_val = pd.read_parquet(y_val_path)

    logger.info(
        "Datasets carregados: X_train=%s, y_train=%s, X_val=%s, y_val=%s",
        X_train.shape,
        y_train.shape,
        X_val.shape,
        y_val.shape
    )

    # Drop columns    
    # X_train.drop(
    #         columns=config_model['single_model']['cols_2_drop'],
    #         inplace=True)        
    # X_val.drop(
    #         columns=config_model['single_model']['cols_2_drop'],
    #         inplace=True)   


    # 3. Model Selection   
    logger.info("Carregando configuracao do modelo: %s", model_name)
    model_orchestrator = SingleModelOrchestrator()
    model_config = model_orchestrator.apply(model_name)  
         
    if grid_search_method == "grid_search":
        # find best params     
        logger.info("Executando grid search: model=%s, scoring=%s", model_name, scoring)
        best_paramns = grid_search_single_model_StratifiedKFold(
            X_train, 
            y_train, 
            model_config['model'], 
            model_config['param_grid'], 
            scoring=scoring
            )     
    elif grid_search_method == "randomized_grid_search":
        logger.info("Executando randomized grid search: model=%s, scoring=%s", model_name, scoring)
        best_paramns = randomized_single_model_grid_search(
            X_train, 
            y_train, 
            model_config['model'], 
            model_config['param_distributions'], 
            scoring=scoring
            ) 
    elif grid_search_method == "no_grid_search":
        logger.info("Usando parametros manuais.")
        best_paramns = manual_params 
    else:
        raise KeyError('please select a grid_search method between:[ grid_search, randomized_grid_search, no_grid_search]')
    
    
    # save model info
    model_info = [{
        'model':model_name,
        'best_paramns': best_paramns,
        'undersamplig': None,
        'model_type':'single_model',
        'timestamp': datetime.now().isoformat()         
    }]       

    logger.info("Metadados do modelo: %s", model_info)
        
    # 4. train model
    logger.info("Iniciando treinamento do modelo: %s", model_config['model_name'])
    model_reg = train_model(
        X_train, 
        y_train, 
        model_config['model'], 
        best_paramns)
    logger.info("Treinamento concluido.")
    
    # 5. cross-validade 
    logger.info("Executando cross-validation com scoring=%s.", scoring)
    df_cv = cross_validate_kfold(
        X_train=X_train, 
        y_train=y_train, 
        model=model_reg,
        model_config=model_config,
        score=scoring        
        )
    
    cv_scoring = df_cv['scoring'].unique()[0]
    logger.info("Cross-validation concluida. Resultados:\n%s", df_cv)
    logger.info(
        "Mean train score %s: %s +- %s",
        cv_scoring,
        df_cv['train_score'].mean(),
        df_cv['train_score'].std()
    )
    logger.info(
        "Mean val score %s: %s +- %s",
        cv_scoring,
        df_cv['val_score'].mean(),
        df_cv['val_score'].std()
    )
        
    path_cv = os.path.join(
        config['init_path'],
        config['single_model']['tables'],
        "cross_validate.jsonl")    
    to_jsonl(df_cv, path_cv, mode='append')    
    logger.info("Resultados de cross-validation salvos em: %s", path_cv)
    
    # 5. Evaulate model
    logger.info("Avaliando modelo no conjunto de treino.")
    metrics_train = evaluate_reg_model(
        model_reg, 
        X_train,
        y_train
        )
    
    logger.info("Train metrics: %s", metrics_train)

    
    logger.info("Avaliando modelo no conjunto de validacao.")
    metrics_val = evaluate_reg_model(model_reg, X_val, y_val)
    
    logger.info("Validation metrics: %s", metrics_val)
    
    # Save Metrics
    file_path = os.path.join(
        config['init_path'],
        config['single_model']['tables']
        )     
    metric_orch = MetricsOrchestrator(output_dir=file_path)    
    metric_orch.save_all_metrics(metrics_train, model_config['model_name'], dataset='train') 
    metric_orch.save_all_metrics(metrics_val, model_config['model_name'], dataset='validation')    
    logger.info("Metricas salvas em: %s", file_path)
    
    # 6. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['single_model']['pkl'],
        f"{model_config['model_name']}_{pipeline_name}.pkl")     
    save_model(model_reg, path_model)
    logger.info("Modelo salvo em: %s", path_model)
    
    # 7. Make predict 
    logger.info("Gerando predicoes para o conjunto de validacao.")
    predictions = make_prediction_reg(model_reg, X_val)
    predictions['prediction'] = predictions['prediction'].apply(np.expm1)
   
        
    path_data = os.path.join(
        config['init_path'],
        config['single_model']['predicts'])    
    model_name = model_config['model_name']
    
    save_data(path_data, f"X_val_pred_{model_name}", predictions) 
    logger.info("Predicoes salvas em: %s", path_data)

def main():
       logger.info("Carregando configuracao: config.")
       config = load_config(load_all=['config'])
       logger.info("Configuracao carregada com sucesso.")
       
       single_model_lite(
            pipeline_name="pipeline1", 
            model_name="RandomForestRegressor",
            scoring="neg_mean_absolute_percentage_error",
            grid_search_method='randomized_grid_search',
            config=config        
        )
   
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Falha inesperada no processamento")
        raise 

