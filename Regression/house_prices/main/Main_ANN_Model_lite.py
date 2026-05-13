from xml.parsers.expat import model
import yaml
import pandas as pd
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

def main_ann_model_lite(pipeline_name: str, model_name:str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Regression/house_prices/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Regression/house_prices/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Regression/house_prices/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando pipeline de Rede Neural...")  
    
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

    #2.  Drop columns
    # X_train.drop(
    #     columns=config_model['ann_model']['cols_2_drop'],
    #     inplace=True)
    
    # X_val.drop(
    #     columns=config_model['ann_model']['cols_2_drop'],
    #     inplace=True)   


    # 3. Model Selection 
    model = KerasRegressor(
        input_dim=X_train.shape[1], 
        hidden_units=(32, 16),
        nonlinear=True,
        epochs=100)    
    
      
    model_info = [{
        'model':model_name,
        'best_paramns': model.get_params(),
        'undersamplig': None,
        'model_type':model_name,
        'timestamp': datetime.now().isoformat()        
    }]       
 
    print(model.get_params(), end='\n')

    
    path_model_info = os.path.join(
        config['init_path'],
        config['ann_model']['tables'],
        "model_info.jsonl")    
    
    to_jsonl(
        pd.DataFrame(model_info), 
        path_model_info, 
        mode='append')
        
    # 4. train model
    clf = model.fit(X_train, y_train)             
    
    # 5. Evaulate model
    metrics_train = evaluate_reg_model(clf, X_train, y_train)
    
    print('**Train metrics**')
    print(f"mean_absolute_error: {metrics_train['mean_absolute_error']}")
    print(f"mean_squared_error: {metrics_train['mean_squared_error']}")
    print(f"root_mean_squared_error: {metrics_train['root_mean_squared_error']}")
    print(f"r2_score: {metrics_train['r2_score']}", end="\n")

    
    metrics_val = evaluate_reg_model(clf, X_val, y_val)
    
    print('**Validation metrics**')
    print(f"mean_absolute_error: {metrics_val['mean_absolute_error']}")
    print(f"mean_squared_error: {metrics_val['mean_squared_error']}")
    print(f"root_mean_squared_error: {metrics_val['root_mean_squared_error']}")
    print(f"r2_score: {metrics_val['r2_score']}", end="\n")
    
    # Save Metrics
    file_path = os.path.join(
        config['init_path'],
        config['ann_model']['tables']
        )     
    metric_orch = MetricsOrchestrator(output_dir=file_path)    
    metric_orch.save_all_metrics(metrics_train, 'ann_model', dataset='train') 
    metric_orch.save_all_metrics(metrics_val, 'ann_model', dataset='validation')    
    
    # 6. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['ann_model']['h5'],
        f'ann_model_{pipeline_name}.h5')     
    save_model(clf, path_model)
    
    # 7. Make predict 
    predictions = make_prediction_reg(model, X_val)
        
    path_data = os.path.join(
        config['init_path'],
        config['single_model']['predicts'])    
    model_name = model_info['model']
    
    save_data(path_data, f"X_val_pred_{model_name}", predictions)     
  
   
   
   
if __name__ == "__main__":
    main_ann_model_lite(pipeline_name='pipeline1', model_name='ANN')