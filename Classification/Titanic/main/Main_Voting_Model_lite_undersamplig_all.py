import yaml
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from functions.train_model import train_voting_model
from functions.evaluate_model import evaluate_model, MetricsOrchestrator
from functions.voting_model import voting_model, models
from functions.undersamplig import UnderSampligOrchestrator

def main_voting_model_undersamplig(pipeline_name: str, scoring:str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Classification/Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando Undersamplig methods analysis...")


    # Get feature eng data
    # 1. Datasets
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

    # 2. Drop columns
    X_train.drop(
        columns=config_model['voting_model']['cols_2_drop'][pipeline_name],
        inplace=True)
    
    X_val.drop(
        columns=config_model['voting_model']['cols_2_drop'][pipeline_name],
        inplace=True)   

    # 3. Apply UnderSamplig
    
    undersamplig_method = {
        "OneSidedSelection",
        "EditedNearestNeighbours",
        "TomekLinks",
        "RepeatedEditedNearestNeighbours",
        "NeighbourhoodCleaningRule",
        "NearMissV1",
        "NearMissV2",
        "NearMissV3",
        "InstanceHardnessThreshold"
        }    
    
    for sampling_method in undersamplig_method: 
        undersamplig_orchestrator = UnderSampligOrchestrator()    
        X_resampled, y_resampled = undersamplig_orchestrator.apply(
            sampling_method, 
            X_train, 
            y_train)
    
            # 3. Model Selection 
        best_params = voting_model(
            X_resampled, 
            y_resampled,
            scoring=scoring
            )     

        # 4. train model
        model_config=dict(model_name = 'voting')
        model_name = 'voting'
        
        model_clf = train_voting_model(
            X_resampled,
            y_resampled, 
            models(), 
            best_params
            ) 
    

        # 5. Evaulate model
        metrics_train = evaluate_model(model_clf, X_resampled, y_resampled)
        
        print('\n')
        print('train metrics')
        print(f"acurácia: {metrics_train['accuracy_score']}")   
        print(f"f1: {metrics_train['f1_score']}")
        print(f"roc_auc: {metrics_train['roc_auc_score']}")
        print('\n')
    
        metrics_val = evaluate_model(model_clf, X_val, y_val)
        
        print('\n')
        print('Validation metrics')
        print(f"acurácia: {metrics_val['accuracy_score']}")   
        print(f"f1: {metrics_val['f1_score']}")
        print(f"roc_auc: {metrics_val['roc_auc_score']}")
        print('\n')
    
        file_path = os.path.join(
            config['init_path'],
            config['voting_model']['tables'])          
        
        # Save Metrics
        metric_orch = MetricsOrchestrator(output_dir=file_path)   
         
        metric_orch.save_all_metrics(
            metrics_train, model_config['model_name'], 
            dataset='train', 
            undersampling=sampling_method) 
        
        metric_orch.save_all_metrics(
            metrics_val, 
            model_config['model_name'], 
            dataset='validation', 
            undersampling=sampling_method)     
    
   
if __name__ == "__main__":
    main_voting_model_undersamplig(
        pipeline_name="Pipeline3",
        scoring='accuracy'
        )