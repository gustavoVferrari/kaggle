import logging
import pandas as pd
import numpy as np
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from Regression.house_prices.src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def submission(pipeline_name: str, model_name: str, config:dict, config_pipe:dict):    
    logger.info(
        "Iniciando geracao de submission single model: model=%s, pipeline=%s",
        model_name,
        pipeline_name
    )
    
    # Get feature eng data
    x_test_path = os.path.join(
        config['init_path'],
        config['data']['feature_eng'],
        f"X_test_feat_eng_{pipeline_name}.parquet"
    )
    logger.info("Carregando dataset de teste com feature engineering: %s", x_test_path)
    X_test = pd.read_parquet(x_test_path)
   
    test_features_path = os.path.join(
        config['init_path'],
        config['data']['processed'],
        "test_features.parquet"
    )
    logger.info("Carregando IDs do conjunto de teste: %s", test_features_path)
    y_test_id = pd.read_parquet(test_features_path)

    y_test_id = y_test_id[['Id']].copy()
    logger.info("Dados carregados: X_test=%s, y_test_id=%s", X_test.shape, y_test_id.shape)
    
    # X_test.drop(
    #     columns = config_model['single_model']['cols_2_drop'],
    #     inplace=True
    # )   
    
    model_path = os.path.join(
           config['init_path'],
           config['single_model']['pkl'],
            f"{model_name}_{pipeline_name}.pkl")
    # open model
    logger.info("Carregando modelo single model: %s", model_path)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
    # predict
    target_col = config_pipe['features']['target'][0]
    logger.info("Gerando predicoes para coluna target: %s", target_col)
    y_test_id.loc[:, target_col] = np.expm1(model.predict(X_test))
    
    submission_path = os.path.join(
        config['init_path'],
        config['data']['submission'],
        f'submission_{model_name}_{pipeline_name}.csv'
    )
    y_test_id.to_csv(submission_path, index=False)
    logger.info("Submission single model salva com sucesso: %s", submission_path)
    

def main():
       logger.info("Carregando configuracoes: config, config_pipe.")
       config, config_pipe = load_config(load_all=['config', 'config_pipe'])
       logger.info("Configuracoes carregadas com sucesso.")
       
       submission(
           pipeline_name="pipeline1",
           model_name="RandomForestRegressor",
           config=config,
           config_pipe=config_pipe
           )
    
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Falha inesperada no processamento")
        raise 
    
