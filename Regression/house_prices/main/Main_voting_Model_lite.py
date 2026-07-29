import os
import pandas as pd
import numpy as np
import sys
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)
from utils.utils import to_jsonl

from functions.make_dataset import save_data
from functions.train_model import train_voting_model_reg, save_model
from functions.evaluate_model import evaluate_reg_model, MetricsOrchestrator
from functions.predict_model import make_prediction_reg
from functions.voting_model_reg import voting_model, models
from functions.cross_validate import cross_validate_kfold
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

from Regression.house_prices.src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main_voting_model_lite(
    pipeline_name:str, 
    scoring:str,
    models:str=models(),
    config:dict=None
    ):

    logger.info(
        "Iniciando pipeline de Machine Learning voting: pipeline=%s, scoring=%s, models=%s",
        pipeline_name,
        scoring,
        models
    )


    # Get feature eng data     
    
    # 1. Datasets
    logger.info("Carregando datasets de treino e validacao.")
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
    logger.info("Executando selecao do voting model: scoring=%s", scoring)
    best_params = voting_model(
        X_train, 
        y_train,
        scoring=scoring
        )     
    logger.info("Melhores parametros encontrados: %s", best_params)

    # 4. train model
    model_config=dict(model_name = 'voting')
    model_name = 'voting'
    
    logger.info("Iniciando treinamento do voting model.")
    model_reg = train_voting_model_reg(
        X_train=X_train,
        y_train=y_train, 
        models=models, 
        best_models_params=best_params,
        search_type="randomized"
        )        
    logger.info("Treinamento concluido.")
    
    # save model info
    model_info = [{
        'model':model_name,
        'best_paramns': best_params,
        'undersamplig': None,
        'model_type':'voting_model',
        'timestamp': datetime.now().isoformat()        
    }]      
   
    logger.info("Metadados do modelo: %s", model_info)
    
    path_model_info = os.path.join(
        config['init_path'],
        config['voting_model']['tables'],
        "model_info.jsonl")    
    to_jsonl(
        pd.DataFrame(model_info), 
        path_model_info, 
        mode='append')
    logger.info("Metadados do modelo salvos em: %s", path_model_info)
        
    # cross validation
    logger.info("Executando cross-validation com scoring=%s.", scoring)
    df_cv = cross_validate_kfold(
        X_train, 
        y_train, 
        model_reg,
        model_config,
        scoring=scoring,
        model_type='voting'
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
        

    
    # 5. Evaulate model
    logger.info("Avaliando modelo no conjunto de treino.")
    metrics_train = evaluate_reg_model(model_reg, X_train, y_train)
    
    logger.info("Train metrics: %s", metrics_train)

    
    logger.info("Avaliando modelo no conjunto de validacao.")
    metrics_val = evaluate_reg_model(model_reg, X_val, y_val)
    
    logger.info("Validation metrics: %s", metrics_val)

    
    # Save Metrics
    file_path = os.path.join(
        config['init_path'],
        config['voting_model']['tables']
        )     
    metric_orch = MetricsOrchestrator(output_dir=file_path)    
    
    metric_orch.save_all_metrics(
        metrics_train, 
        'voting_model', 
        dataset='train'
        ) 
    metric_orch.save_all_metrics(
        metrics_val,
        'voting_model',
        dataset='validation'
        )    
    logger.info("Metricas salvas em: %s", file_path)
    
    # 6. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['voting_model']['pkl'],
        f'voting_model_{pipeline_name}.pkl')     
    save_model(model_reg, path_model)
    logger.info("Modelo salvo em: %s", path_model)
    
    # 7. Make predict 
    logger.info("Gerando predicoes para o conjunto de validacao.")
    predictions = make_prediction_reg(model_reg, X_val)
    predictions['prediction'] = predictions['prediction'].apply(np.expm1)
        
    path_data = os.path.join(
        config['init_path'],
        config['voting_model']['predicts'])    
    model_name = 'voting_model'
    
    save_data(
        path_data, 
        f"X_val_pred_{model_name}", 
        predictions
        )      
    logger.info("Predicoes salvas em: %s", path_data)

def main():
    logger.info("Carregando configuracao: config.")
    config = load_config(load_all=['config'])
    logger.info("Configuracao carregada com sucesso.")
      
    models_list = [
        "RidgeRegressor", 
        "RandomForestRegressor"
        ]
    
    main_voting_model_lite(
        pipeline_name="pipeline1",
        scoring='neg_mean_absolute_percentage_error',
        models=models_list,
        config=config
        )
   
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Falha inesperada no processamento")
        raise 
