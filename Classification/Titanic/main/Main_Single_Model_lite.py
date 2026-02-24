import yaml
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.utils.utils import to_jsonl
from src.features.make_dataset import save_data
from src.features.single_model import ModelOrchestrator
from src.features.model_selection import grid_search
from src.features.train_model import train_model, save_model
from src.features.evaluate_model import evaluate_model, MetricsOrchestrator
from src.features.predict_model import make_prediction
from src.features.cross_validate import cross_validate
from src.features.threshold_analysis import threshold_optimization

def main_single_model_lite():
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando pipeline de Machine Learning...")


    # Get feature eng data
    pipeline_name = "Pipeline1"
    
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
    
    model_name = "LGBM"
    model_orchestrator = ModelOrchestrator(seed_=23)    
    model_config = model_orchestrator.apply(model_name)    
         
    best_paramns = grid_search(X_train, y_train, model_config['models_gs'], 'roc_auc', cv=5)     
    print(best_paramns)
    
    # 4. train model
    clf = train_model(X_train, y_train, model_config['model'], best_paramns)
    
    # cross-validade
    df_cv = cross_validate(X_train, y_train, model_config, best_paramns)
    
    print(df_cv)
    
    path_cv = os.path.join(
        config['init_path'],
        config['single_model']['tables'],
        "cross_validate.jsonl")    
    to_jsonl(df_cv, path_cv, mode='append')
    
    
    # 5. Evaulate model
    metrics = evaluate_model(clf, X_val, y_val)
    
    print(f'report: {metrics['classification_report']}')
    print('\n')
    print(f'acurácia: {metrics['accuracy_score']}')   
    print(f'f1: {metrics['f1_score']}')
    print(f'roc_auc: {metrics['roc_auc_score']}')
    
    file_path = os.path.join(
        config['init_path'],
        config['single_model']['tables'])          
    
    # Save Metrics
    metric_orch = MetricsOrchestrator(output_dir=file_path)    
    metric_orch.save_all_metrics(metrics, model_config['model_name'])    
    
    # 6. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['single_model']['pkl'],
        f'{model_config['model_name']}.pkl')
    
    save_model(clf, path_model)
    
    # 7. Make predict
    predictions, probabilities = make_prediction(clf, X_val)
    
    path_data = os.path.join(
        config['init_path'],
        config['single_model']['predicts'])
    
    model_name = model_config['model_name']
    
    save_data(path_data, f"X_val_pred_{model_name}", predictions)
    save_data(path_data, f"X_val_proba_{model_name}", probabilities)    
    
    # threshold analysis
    path_threshold = os.path.join(
        config['init_path'],
        config['single_model']['figures'])
    
    threshold_point =threshold_optimization(
        y_val, 
        probabilities, 
        model_name=model_name, 
        path=path_threshold)
    
    print(threshold_point)
    
   
if __name__ == "__main__":
    main_single_model_lite()