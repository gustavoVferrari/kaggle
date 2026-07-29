import logging
import pandas as pd
import numpy as np
import os
import sys
import pickle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from Regression.house_prices.src.utils.config import load_config


def submission(pipeline_name: str, config:dict, config_pipe:dict):    
    logger.info("Iniciando geracao de submission ANN: pipeline=%s", pipeline_name)
    
    # Get feature eng data
    x_test_path = os.path.join(
        config['init_path'],
        config['data']['feature_eng'],
        f"X_test_feat_eng_{pipeline_name}.parquet"
    )
    logger.info("Carregando dataset de teste com feature engineering: %s", x_test_path)
    X_test = pd.read_parquet(
       x_test_path
    )

    test_features_path = os.path.join(
        config['init_path'],
        config['data']['processed'],
        f"test_features.parquet"
    )
    logger.info("Carregando IDs do conjunto de teste: %s", test_features_path)
    y_test_id = pd.read_parquet(
       test_features_path
    )

    y_test_id = y_test_id[['Id']].copy()
    logger.info("Dados carregados: X_test=%s, y_test_id=%s", X_test.shape, y_test_id.shape)
    
    # X_test.drop(
    #     columns = config_model['single_model']['cols_2_drop'],
    #     inplace=True
    # )   
    
    model_path = os.path.join(
           config['init_path'],
           config['ann_model']['h5'],
            f"ann_model_{pipeline_name}.h5")
    # open model
    logger.info("Carregando modelo ANN: %s", model_path)
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
    # predict
    target_col = config_pipe['features']['target'][0]
    logger.info("Gerando predicoes para coluna target: %s", target_col)
    y_test_id.loc[:, target_col] = np.expm1(model.predict(X_test))

    submission_path = os.path.join(
       config['init_path'],
       config['data']['submission'],
        f'submission_ann_model_{pipeline_name}.csv'
    )
    y_test_id.to_csv(submission_path, index=False)
    logger.info("Submission ANN salva com sucesso: %s", submission_path)
    
def main():
       logger.info("Carregando configuracoes: config, config_pipe.")
       config, config_pipe = load_config(load_all=['config', 'config_pipe'])
       logger.info("Configuracoes carregadas com sucesso.")
       
       submission(
               pipeline_name='pipeline1',
               config=config,
               config_pipe=config_pipe              
               )
   
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Falha inesperada no processamento")
        raise 
