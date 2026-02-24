import yaml
import pandas as pd
import os
import sys
from sklearn.pipeline import Pipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.features.make_dataset import split_data
from src.features.build_features import PreprocessingPipeline_3
from src.features.single_model import log_reg, rforest, mlp, adaboost
from Classification.Titanic.old.model_selection_sklearn import model_selection
from Classification.Titanic.old.train_model_single import train_single_model
from src.features.undersamplig import UnderSampligOrchestrator

def main_single_model():
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "config/pipeline_1.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando pipeline de Machine Learning...")
    
    # 1. Carregar dados processados
    df = pd.read_parquet(
        os.path.join(
            config['init_path'],
            config['data']['processed'],
            "train_features.parquet"
            )       
        )
    
    # 2. Processamento de dados    
    X_train, X_val, y_train, y_val = split_data(
        df, 
        target_column=config_pipe['features']['target'][0]
        )

    # 3. Feature Engineering        
    preprocessor = PreprocessingPipeline_3(
        numerical_con=config_pipe['features']['num_con'], 
        numerical_dis=config_pipe['features']['num_dis'], 
        categorical_var=config_pipe['features']['cat_var'])
    
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    
    X_train.drop(
        columns=config_model['single_model']['cols_2_drop'],
        inplace=True)
    
    X_val.drop(
        columns=config_model['single_model']['cols_2_drop'],
        inplace=True)
    
    
    orchestrator = UnderSampligOrchestrator()
    
    X_resampled, y_resampled = orchestrator.apply(
        "OneSidedSelection", 
        X_train, 
        y_train)
    
    
    # 3. Model Selection
    args_ms = {
        'X_train': X_resampled,
        'y_train': y_resampled,
        'X_val': X_val,
        'y_val': y_val,
        
        # select model
        'models': adaboost['model_params'],
        'metric': 'roc_auc',
        'cv': 5,
        'random_state':23,
        'reports': os.path.join(
            project_root, 
            config_model['single_model']['tables']),
        'plot': os.path.join(
            project_root, 
            config_model['single_model']['figures']),
        }    
    
    model_selection(**args_ms) 
    
    # 5. Train Model
    print("Treinando o modelo...")
    
    args_tm = {
        'X_train': X_resampled,
        'y_train': y_resampled,        
        'metric': 'roc_auc',
        'cv': 5,
        'random_state':23,
        
        # select model
        'model_params': os.path.join(
            project_root, 
            config_model['single_model']['tables'], 
            'best_model_params.jsonl'),   
           
        'model': adaboost['model'],
        'model_name': adaboost['model_name'],        
        
        'predict': os.path.join(
            project_root, 
            config_model['single_model']['predicts']),
        'pkl': os.path.join(
            project_root, 
            config_model['single_model']['pkl']),
        'reports': os.path.join(
            project_root, 
            config_model['single_model']['tables']),
        'plot': os.path.join(
            project_root, 
            config_model['single_model']['figures'])
        }    
        
    train_single_model(**args_tm)

    # # 6. Avaliação
    # print("Avaliando o modelo...")
    # report, matrix = evaluate_model(full_pipeline, X_test, y_test)
    # print("\nRelatório de Classificação:\n", report)

    # # 7. Salvar Modelo
    # save_model(full_pipeline, config['model_path'])
    # print(f"Modelo salvo em: {config['model_path']}")

if __name__ == "__main__":
    main_single_model()