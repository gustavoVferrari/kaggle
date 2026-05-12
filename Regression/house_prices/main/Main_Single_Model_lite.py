import yaml
import pandas as pd
import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from utils.utils import to_jsonl
from utils.plots import cross_validation_plot
from functions.make_dataset import save_data
from functions.model_selection import grid_search_single_model_StratifiedKFold, randomized_single_model_grid_search
from functions.train_model import train_model, save_model
from functions.evaluate_model import evaluate_reg_model, MetricsOrchestrator
from functions.predict_model import make_prediction_reg
from functions.cross_validate import cross_validate_StratifiedKFold
from functions.single_model_reg import SingleModelOrchestrator

def main_single_model_lite(pipeline_name:str, model_name:str, scoring:str, grid_search_method:str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Regression/house_prices/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Regression/house_prices/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Regression/house_prices/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print(f"Iniciando pipeline de Machine Learning {model_name}, scoring {scoring} with {pipeline_name}")


    # Get feature eng data   
    
    # Datasets
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

    # Drop columns
    
    X_train.drop(
            columns=config_model['single_model']['cols_2_drop'],
            inplace=True)
        
    X_val.drop(
            columns=config_model['single_model']['cols_2_drop'],
            inplace=True)   


    # 3. Model Selection   
    model_orchestrator = SingleModelOrchestrator()
    model_config = model_orchestrator.apply(model_name)  
         
    if grid_search_method == "grid_search":
        # find best params     
        print(f"Running grid search with {scoring} as metric and {model_name} as model")
        best_paramns = grid_search_single_model_StratifiedKFold(
            X_train, 
            y_train, 
            model_config['model'], 
            model_config['param_grid'], 
            scoring=scoring
            )     
    elif grid_search_method == "randomized_grid_search":
        print(f"Running randomized grid search with {scoring} as metric and {model_name} as model")
        best_paramns = randomized_single_model_grid_search(
            X_train, 
            y_train, 
            model_config['model'], 
            model_config['param_distributions'], 
            scoring=scoring
            ) 
    else:
        raise KeyError('please select a grid_search method between:[ grid_search, randomized_grid_search]')  
    
    
    # save model info
    model_info = [{
        'model':model_name,
        'best_paramns': best_paramns,
        'undersamplig': None,
        'model_type':'single_model',
        'timestamp': datetime.now().isoformat()        
    }]       
    print("\n")
    print(model_info)
    print("\n")
        
    # 4. train model
    model_reg = train_model(
        X_train, 
        y_train, 
        model_config['model'], 
        best_paramns)
    
    # 5. cross-validade
    df_cv = cross_validate_StratifiedKFold(
        X_train, 
        y_train, 
        model_reg,
        model_config,
        scoring=scoring
        )
    print(df_cv, end='\n')
    print(f"Mean train score {df_cv['scoring'].unique()[0]}: {df_cv['train_score'].mean()} +- {df_cv['train_score'].std()}", end='\n')
    print(f"Mean val score {df_cv['scoring'].unique()[0]}: {df_cv['val_score'].mean()} +- {df_cv['val_score'].std()}", end='\n')
        
    path_cv = os.path.join(
        config['init_path'],
        config['single_model']['tables'],
        "cross_validate.jsonl")    
    to_jsonl(df_cv, path_cv, mode='append')    
    
    # 5. Evaulate model
    metrics_train = evaluate_reg_model(
        model_reg, 
        X_train,
        y_train
        )
    
    print('train metrics')
    print(f"mean_absolute_error: {metrics_train['mean_absolute_error']}")
    print(f"mean_squared_error: {metrics_train['mean_squared_error']}")
    print(f"root_mean_squared_error: {metrics_train['root_mean_squared_error']}")
    print(f"r2_score: {metrics_train['r2_score']}")
    print('\n')
    
    metrics_val = evaluate_reg_model(model_reg, X_val, y_val)
    
    print('Validation metrics')
    print(f"mean_absolute_error: {metrics_val['mean_absolute_error']}")
    print(f"mean_squared_error: {metrics_val['mean_squared_error']}")
    print(f"root_mean_squared_error: {metrics_val['root_mean_squared_error']}")
    print(f"r2_score: {metrics_val['r2_score']}")
    print('\n')
    
    # Save Metrics
    file_path = os.path.join(
        config['init_path'],
        config['single_model']['tables']
        )     
    metric_orch = MetricsOrchestrator(output_dir=file_path)    
    metric_orch.save_all_metrics(metrics_train, model_config['model_name'], dataset='train') 
    metric_orch.save_all_metrics(metrics_val, model_config['model_name'], dataset='validation')    
    
    # 6. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['single_model']['pkl'],
        f"{model_config['model_name']}_{pipeline_name}.pkl")     
    save_model(model_reg, path_model)
    
    # 7. Make predict 
    predictions = make_prediction_reg(model_reg, X_val)
        
    path_data = os.path.join(
        config['init_path'],
        config['single_model']['predicts'])    
    model_name = model_config['model_name']
    
    save_data(path_data, f"X_val_pred_{model_name}", predictions) 
    
   
if __name__ == "__main__":
    main_single_model_lite(
        pipeline_name="pipeline1", 
        model_name="RandomForestRegressor",
        scoring="neg_mean_absolute_percentage_error",
        grid_search_method='randomized_grid_search'
        )