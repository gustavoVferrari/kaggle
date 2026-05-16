import os
import pandas as pd
import numpy as np
import sys
import yaml

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

def main_voting_model_lite(
    pipeline_name:str, 
    scoring:str,
    models:str=models()
    ):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Regression/house_prices/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Regression/house_prices/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Regression/house_prices/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando pipeline de Machine Learning voting...")


    # Get feature eng data     
    
    # 1. Datasets
    X_train = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"X_train_feat_eng_{pipeline_name}.parquet"))
   
    y_train = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
           f"y_train_feat_eng_{pipeline_name}.parquet"))
    
    X_val = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"X_val_feat_eng_{pipeline_name}.parquet"))
    
    y_val = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"y_val_feat_eng_{pipeline_name}.parquet"))

  
    # Drop columns    
    # X_train.drop(
    #         columns=config_model['single_model']['cols_2_drop'],
    #         inplace=True)        
    # X_val.drop(
    #         columns=config_model['single_model']['cols_2_drop'],
    #         inplace=True)  


    # 3. Model Selection        
    best_params = voting_model(
        X_train, 
        y_train,
        scoring=scoring
        )     

    # 4. train model
    model_config=dict(model_name = 'voting')
    model_name = 'voting'
    
    model_reg = train_voting_model_reg(
        X_train=X_train,
        y_train=y_train, 
        models=models, 
        best_models_params=best_params,
        search_type="randomized"
        )        
    
    # save model info
    model_info = [{
        'model':model_name,
        'best_paramns': best_params,
        'undersamplig': None,
        'model_type':'voting_model',
        'timestamp': datetime.now().isoformat()        
    }]      
   
    print(model_info, end="\n")    
    
    path_model_info = os.path.join(
        config['init_path'],
        config['voting_model']['tables'],
        "model_info.jsonl")    
    to_jsonl(
        pd.DataFrame(model_info), 
        path_model_info, 
        mode='append')
        
    # cross validation
    df_cv = cross_validate_kfold(
        X_train, 
        y_train, 
        model_reg,
        model_config,
        scoring=scoring,
        model_type='voting'
        )
    
    print(df_cv, end='\n')
    print(f"Mean train score {df_cv['scoring'].unique()[0]}: {df_cv['train_score'].mean()} +- {df_cv['train_score'].std()}")
    print(f"Mean val score {df_cv['scoring'].unique()[0]}: {df_cv['val_score'].mean()} +- {df_cv['val_score'].std()}")
        

    
    # 5. Evaulate model
    metrics_train = evaluate_reg_model(model_reg, X_train, y_train)
    
    print('**Train metrics**')
    print(f"mean_absolute_error: {metrics_train['mean_absolute_error']}")
    print(f"mean_squared_error: {metrics_train['mean_squared_error']}")
    print(f"root_mean_squared_error: {metrics_train['root_mean_squared_error']}")
    print(f"r2_score: {metrics_train['r2_score']}", end='\n')

    
    metrics_val = evaluate_reg_model(model_reg, X_val, y_val)
    
    print('**Validation metrics**')
    print(f"mean_absolute_error: {metrics_val['mean_absolute_error']}")
    print(f"mean_squared_error: {metrics_val['mean_squared_error']}")
    print(f"root_mean_squared_error: {metrics_val['root_mean_squared_error']}")
    print(f"r2_score: {metrics_val['r2_score']}", end='\n')

    
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
    
    # 6. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['voting_model']['pkl'],
        f'voting_model_{pipeline_name}.pkl')     
    save_model(model_reg, path_model)
    
    # 7. Make predict 
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

   
   
if __name__ == "__main__":
    
    models_list = [
        "RidgeRegressor",
        "SVRRegressor",
        "RandomForestRegressor",
        "MLPRegressor",
        "HistGradientBoostingRegressor",
        "XGBRegressor",
        "LGBMRegressor" 
        ]
    main_voting_model_lite(
        pipeline_name="pipeline1",
        scoring='neg_mean_absolute_percentage_error',
        models=models_list
    )