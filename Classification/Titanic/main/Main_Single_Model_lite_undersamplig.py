import yaml
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

from utils.utils import to_jsonl
from functions.make_dataset import save_data
from functions.single_model import ModelOrchestrator
from functions.model_selection import grid_search
from functions.train_model import train_model, save_model
from functions.evaluate_model import evaluate_model, MetricsOrchestrator
from functions.undersamplig import UnderSampligOrchestrator
from functions.predict_model import make_prediction
from functions.cross_validate import cross_validate


def main_single_model_undersamplig():
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Titanic/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando pipeline de Machine Learning...")


    # Get feature eng data
    pipeline_name = "Pipeline3"
    
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

    # Apply UnderSamplig    
    
    undersamplig_orchestrator = UnderSampligOrchestrator()    
    X_resampled, y_resampled = undersamplig_orchestrator.apply(
        "InstanceHardnessThreshold", 
        X_train, 
        y_train
        )
    
    # 3. Model Selection 
    model_name = "LogisticRegression"
    model_orchestrator = ModelOrchestrator(seed_=23)    
    model_config = model_orchestrator.apply(model_name)    
         
    best_paramns = grid_search(X_resampled, y_resampled, model_config['models_gs'], 'roc_auc', cv=5) 
    
    # 4. train model
    clf = train_model(X_resampled, y_resampled, model_config['model'], best_paramns)
    
    # cross-validade
    df_cv = cross_validate(X_resampled, y_resampled, model_config, best_paramns)
    
    print(df_cv)
    path_cv = os.path.join(
        config['init_path'],
        config['single_model']['tables'],
        "cross_validate.jsonl")
    
    to_jsonl(df_cv, path_cv, mode='append')
    
    # 5. Evaulate model
    metrics_train = evaluate_model(clf, X_train, y_train)
    
    print('train metrics')
    print(f'report: {metrics_train['classification_report']}')
    print(f'acurácia: {metrics_train['accuracy_score']}')   
    print(f'f1: {metrics_train['f1_score']}')
    print(f'roc_auc: {metrics_train['roc_auc_score']}')
    print('\n')
    
    metrics_val = evaluate_model(clf, X_val, y_val)
    
    print('Validation metrics')
    print(f'report: {metrics_val['classification_report']}')
    print(f'acurácia: {metrics_val['accuracy_score']}')   
    print(f'f1: {metrics_val['f1_score']}')
    print(f'roc_auc: {metrics_val['roc_auc_score']}')
    print('\n')
    
    file_path = os.path.join(
        config['init_path'],
        config['single_model']['tables'])          
    
    # Save Metrics
    metric_orch = MetricsOrchestrator(output_dir=file_path)    
    metric_orch.save_all_metrics(
        metrics_train, 
        model_config['model_name'], 
        dataset='train', 
        undersampling='InstanceHardnessThreshold') 
    
    metric_orch.save_all_metrics(
        metrics_val, 
        model_config['model_name'], 
        dataset='validation', 
        undersampling='InstanceHardnessThreshold')      
    
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
    
   
if __name__ == "__main__":
    main_single_model_undersamplig()