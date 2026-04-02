import yaml
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

from functions.make_dataset import split_data, save_data
from Titanic.src.features.feature_eng import PreprocessingOrchestrator


def main_feature_eng(pipeline_name: str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)

    print(f"Iniciando pipeline de Feature enginneering {pipeline_name}...")
    
    # 1. Carregar dados processados
    df = pd.read_parquet(
        os.path.join(
            config['init_path'],
            config['data']['processed'],
            "train_features.parquet"
            )       
        )
    
    X_test = pd.read_parquet(
        os.path.join(
            config['init_path'],
            config['data']['processed'],
            "test_features.parquet"
            )       
        )
    
    # 2. Processamento de dados    
    X_train, X_val, y_train, y_val = split_data(
        df, 
        target_column=config_pipe['features']['target'][0]
        )

    # 3. Feature Engineering        
    preprocessor = PreprocessingOrchestrator(
        numerical_con=config_pipe['features']['num_con'], 
        numerical_dis=config_pipe['features']['num_dis'], 
        categorical_var=config_pipe['features']['cat_var'])
    
    # define pipiline   
    
    pipe = preprocessor.apply(pipeline_name)    
    X_train = pipe.fit_transform(X_train, y_train)
    X_val = pipe.transform(X_val)    
    X_test = pipe.transform(X_test)
    
    y_val = pd.DataFrame(y_val)
    y_train = pd.DataFrame(y_train)
    
    # Sava datasets
    path_data = os.path.join(
        config['init_path'],
        config['data']['feature_eng'])
    
    save_data(path_data, f"X_test_feat_eng_{pipeline_name}", X_test)
    
    save_data(path_data, f"X_val_feat_eng_{pipeline_name}", X_val)
    save_data(path_data, f"Y_val_feat_eng_{pipeline_name}", y_val)
    
    save_data(path_data, f"X_train_feat_eng_{pipeline_name}", X_train)
    save_data(path_data, f"Y_train_feat_eng_{pipeline_name}", y_train)   
    
   
if __name__ == "__main__":
    main_feature_eng(pipeline_name = "Pipeline3")
    # main_feature_eng(pipeline_name = "Pipeline2")
    # main_feature_eng(pipeline_name = "Pipeline1")