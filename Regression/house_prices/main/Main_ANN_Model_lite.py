from xml.parsers.expat import model
import logging
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from functions.make_dataset import save_data
from utils.utils import to_jsonl
from functions.train_model import save_model
from functions.evaluate_model import evaluate_reg_model, MetricsOrchestrator
from functions.predict_model import make_prediction_reg
from functions.ann_model import KerasRegressor

from Regression.house_prices.src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

def ann_model_lite(pipeline_name: str, model_name:str, config:dict, config_model:dict, drop_columns:bool=True):
    """
    Executa o treinamento e a avaliacao de um modelo ANN para regressao.

    A funcao carrega as configuracoes do projeto, le os datasets de treino e
    validacao gerados pela etapa de feature engineering, instancia o modelo
    `KerasRegressor`, treina a rede neural, avalia as metricas de regressao,
    salva os resultados, persiste o modelo treinado em disco e gera predicoes
    para o conjunto de validacao.

    Args:
        pipeline_name (str): Nome do pipeline de feature engineering usado para
            localizar os arquivos parquet de treino e validacao.
        model_name (str): Nome do modelo usado nos metadados e no arquivo de
            predicoes.
        drop_columns (bool, optional): Indica se as colunas configuradas em
            `model.yaml` devem ser removidas antes do treinamento. O padrao e
            True.

    Returns:
        None: A funcao executa o fluxo completo e salva os artefatos em disco.
    """

    logger.info(
        "Iniciando pipeline de Rede Neural: pipeline=%s, model=%s, drop_columns=%s",
        pipeline_name,
        model_name,
        drop_columns
    )    

    logger.info("Configuracoes carregadas com sucesso.")
    
    # Datasets
    logger.info("Carregando datasets de treino e validacao.")
    X_train = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"X_train_feat_eng_{pipeline_name}.parquet")
   )
   
    y_train = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"y_train_feat_eng_{pipeline_name}.parquet")
   )
    
    X_val = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"X_val_feat_eng_{pipeline_name}.parquet")
   )
    
    y_val = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"y_val_feat_eng_{pipeline_name}.parquet")
   )

    logger.info(
        "Datasets carregados: X_train=%s, y_train=%s, X_val=%s, y_val=%s",
        X_train.shape,
        y_train.shape,
        X_val.shape,
        y_val.shape
    )

    if drop_columns == True:
    # 2.  Drop columns
        logger.info(
            "Removendo colunas configuradas para ANN: %s",
            config_model['ann_model']['cols_2_drop']
        )
        X_train.drop(
            columns=config_model['ann_model']['cols_2_drop'],
            inplace=True)
        
        X_val.drop(
            columns=config_model['ann_model']['cols_2_drop'],
            inplace=True)   
    else:
        logger.warning("Nenhuma coluna sera removida antes do treinamento.")


    # 3. Model Selection 
    logger.info("Instanciando modelo ANN com input_dim=%s.", X_train.shape[1])
    model = KerasRegressor(
        input_dim=X_train.shape[1], 
        hidden_units=(50, 50, 50),
        nonlinear=True,
        learning_rate=0.01,
        dropout_rate=0.0001,
        epochs=300)    
    
      
    model_info = [{
        'model':model_name,
        'best_paramns': model.get_params(),
        'undersamplig': None,
        'model_type':model_name,
        'timestamp': datetime.now().isoformat()        
    }]       
 
    logger.info("Parametros do modelo: %s", model.get_params())

    
    path_model_info = os.path.join(
        config['init_path'],
        config['ann_model']['tables'],
        "model_info.jsonl")    
    
    to_jsonl(
        pd.DataFrame(model_info), 
        path_model_info, 
        mode='append')
    logger.info("Metadados do modelo salvos em: %s", path_model_info)
        
    # 4. train model
    logger.info("Iniciando treinamento do modelo ANN.")
    model_reg = model.fit(X_train, y_train)             
    logger.info("Treinamento concluido.")
    
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
        config['ann_model']['tables']
        )     
    metric_orch = MetricsOrchestrator(output_dir=file_path)    
    metric_orch.save_all_metrics(metrics_train, 'ann_model', dataset='train') 
    metric_orch.save_all_metrics(metrics_val, 'ann_model', dataset='validation')    
    logger.info("Metricas salvas em: %s", file_path)
    
    # 6. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['ann_model']['h5'],
        f'ann_model_{pipeline_name}.h5')     
    save_model(model_reg, path_model)
    logger.info("Modelo salvo em: %s", path_model)
    
    # 7. Make predict 
    logger.info("Gerando predicoes para o conjunto de validacao.")
    predictions = make_prediction_reg(model, X_val)
    predictions['prediction'] = predictions['prediction'].apply(np.expm1)
        
    path_data = os.path.join(
        config['init_path'],
        config['ann_model']['predicts'])    
    
    save_data(path_data, f"X_val_pred_{model_info[0]['model']}", predictions)     
    logger.info("Predicoes salvas em: %s", path_data)
  
   
def main():
       config, config_model = load_config(load_all=['config', 'config_model'])
       
       ann_model_lite(
               pipeline_name='pipeline1',
               model_name='ANN', 
               config=config,
               config_model=config_model,
               drop_columns=False
               )
   
if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Falha inesperada no processamento")
        raise 