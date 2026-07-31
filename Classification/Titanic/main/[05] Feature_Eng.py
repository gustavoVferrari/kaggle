import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from utils.logs import setup_logging
from Classification.Titanic.src.utils.config import load_config

from functions.make_dataset import split_data, save_data
from Classification.Titanic.src.features.feature_eng import PreprocessingOrchestrator


def feature_eng(pipeline_name: str, config:dict, config_pipe:dict):
    
    log_path = os.path.join(project_root, "Classification/Titanic")
    logger = setup_logging(log_path)     
   
    try:
        logger.info("Starting feature engineering pipeline: %s", pipeline_name)
        
        train_features_path = os.path.join(
            config['init_path'],
            config['data']['processed'],
            "train_features.parquet"
            )
        
        test_features_path = os.path.join(
            config['init_path'],
            config['data']['processed'],
            "test_features.parquet"
            )
        
        # 1. Carregar dados processados
        logger.info("Loading training features from: %s", train_features_path)
        df = pd.read_parquet(train_features_path)
        logger.info("Training features loaded with shape: %s", df.shape)
        
        logger.info("Loading test features from: %s", test_features_path)
        X_test = pd.read_parquet(test_features_path)
        logger.info("Test features loaded with shape: %s", X_test.shape)
        
        # 2. Processamento de dados
        target_column = config_pipe['features']['target'][0]
        logger.info("Splitting training and validation data using target: %s", target_column)
        X_train, X_val, y_train, y_val = split_data(
            df, 
            target_column=target_column
            )
        logger.info(
            "Split completed | X_train: %s | X_val: %s | y_train: %s | y_val: %s",
            X_train.shape,
            X_val.shape,
            y_train.shape,
            y_val.shape,
        )

        # 3. Feature Engineering
        logger.info("Building preprocessing orchestrator for pipeline: %s", pipeline_name)
        preprocessor = PreprocessingOrchestrator(
            numerical_con=config_pipe['features']['num_con'], 
            numerical_dis=config_pipe['features']['num_dis'], 
            categorical_var=config_pipe['features']['cat_var'])
        
        logger.info("Applying preprocessing pipeline: %s", pipeline_name)
        pipe = preprocessor.apply(pipeline_name)
        
        logger.info("Fitting pipeline and transforming training data")
        X_train = pipe.fit_transform(X_train, y_train)
        
        logger.info("Transforming validation data")
        X_val = pipe.transform(X_val)
        
        logger.info("Transforming test data")
        X_test = pipe.transform(X_test)
        
        y_val = pd.DataFrame(y_val)
        y_train = pd.DataFrame(y_train)
        
        logger.info(
            "Feature engineering completed | X_train: %s | X_val: %s | X_test: %s",
            X_train.shape,
            X_val.shape,
            X_test.shape,
        )
        
        # Save datasets
        path_data = os.path.join(
            config['init_path'],
            config['data']['feature_eng'])
        
        logger.info("Saving feature engineering outputs to: %s", path_data)
        save_data(path_data, f"X_test_feat_eng_{pipeline_name}", X_test)
        logger.info("Saved X_test feature engineering dataset for pipeline: %s", pipeline_name)
        
        save_data(path_data, f"X_val_feat_eng_{pipeline_name}", X_val)
        logger.info("Saved X_val feature engineering dataset for pipeline: %s", pipeline_name)
        
        save_data(path_data, f"Y_val_feat_eng_{pipeline_name}", y_val)
        logger.info("Saved y_val feature engineering dataset for pipeline: %s", pipeline_name)
        
        save_data(path_data, f"X_train_feat_eng_{pipeline_name}", X_train)
        logger.info("Saved X_train feature engineering dataset for pipeline: %s", pipeline_name)
        
        save_data(path_data, f"Y_train_feat_eng_{pipeline_name}", y_train)
        logger.info("Saved y_train feature engineering dataset for pipeline: %s", pipeline_name)
        
        logger.info("Feature engineering pipeline completed: %s", pipeline_name)
    except Exception:
        logger.exception("Feature engineering pipeline failed: %s", pipeline_name)
        raise
    
def main():
    config, config_pipe = load_config(['config', 'config_pipe'])
    feature_eng(pipeline_name = "Pipeline3", config=config, config_pipe=config_pipe)
    feature_eng(pipeline_name = "Pipeline2", config=config, config_pipe=config_pipe)
    feature_eng(pipeline_name = "Pipeline1", config=config, config_pipe=config_pipe)

if __name__ == "__main__":
    main()
