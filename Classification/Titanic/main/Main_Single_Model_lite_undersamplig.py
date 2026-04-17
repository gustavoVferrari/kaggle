import yaml
import pandas as pd
import os
import sys
from datetime import datetime
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from utils.utils import to_jsonl
from utils.plots import cross_validation_plot, separation_plan_plot
from functions.make_dataset import save_data
from functions.single_model import SingleModelOrchestrator
from functions.model_selection import grid_search_single_model_StratifiedKFold
from functions.train_model import train_model, save_model
from functions.evaluate_model import evaluate_model, MetricsOrchestrator
from functions.undersamplig import UnderSampligOrchestrator
from functions.predict_model import make_prediction
from functions.cross_validate import cross_validate_StratifiedKFold


def main_single_model_undersamplig(pipeline_name:str, model_name:str, undersampling_method:str, scoring:str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Classification/Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando modelo de Machine Learning...")
    
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
        columns=config_model['single_model']['cols_2_drop'],
        inplace=True)
    
    X_val.drop(
        columns=config_model['single_model']['cols_2_drop'],
        inplace=True)   

    # 3. Apply UnderSamplig        
    undersamplig_orchestrator = UnderSampligOrchestrator()    
    X_resampled, y_resampled = undersamplig_orchestrator.apply(
        undersampling_method, 
        X_train, 
        y_train
        )
    
    # 4. Model Selection 
    model_orchestrator = SingleModelOrchestrator()    
    model_config = model_orchestrator.apply(model_name)    
         
    best_paramns = grid_search_single_model_StratifiedKFold(
        X_resampled, 
        y_resampled, 
        model_config['model'], 
        model_config['param_grid'],
        scoring=scoring) 
    
    # 5. train model
    model_clf = train_model(
        X_resampled,
        y_resampled,
        model_config['model'], 
        best_paramns)
    
    
    # 6. cross-validade
    df_cv = cross_validate_StratifiedKFold(
        X_resampled, 
        y_resampled, 
        model_clf,
        model_config,
        scoring=scoring)
    
    save_path = os.path.join(
        config['init_path'],
        config['single_model']['plots'],
        f"cross_validation_{model_name}.png")
    
    cross_validation_plot(save_path, model_name, df_cv)
    
    path_cv = os.path.join(
        config['init_path'],
        config['single_model']['tables'],
        "cross_validate.jsonl")
    
    to_jsonl(df_cv, path_cv, mode='append')
    
    print("\n")
    print(df_cv, end='\n')
    print(f"Mean train score {df_cv['scoring'].unique()[0]}: {df_cv['train_score'].mean()} +- {df_cv['train_score'].std()}")
    print(f"Mean val score {df_cv['scoring'].unique()[0]}: {df_cv['val_score'].mean()} +- {df_cv['val_score'].std()}")
    
    # model info
    model_info = [{
        'model':model_name,
        'best_paramns': best_paramns,
        'undersamplig': undersampling_method,
        'model_type':'single_model',
        'timestamp': datetime.now().isoformat()        
    }]       
    print("\n")
    print(model_info)
    print("\n")
    
    path_model_info = os.path.join(
        config['init_path'],
        config['single_model']['tables'],
        "model_info.jsonl")    
    to_jsonl(
        pd.DataFrame(model_info), 
        path_model_info, 
        mode='append')
    
    # 7. Evaulate model
    metrics_train = evaluate_model(
        model_clf, 
        X_train, 
        y_train)
    
    print('\n')
    print('train metrics')
    # print(f"report: {metrics_train['classification_report']}")
    print(f"acurácia: {metrics_train['accuracy_score']}")   
    print(f"f1: {metrics_train['f1_score']}")
    print(f"roc_auc: {metrics_train['roc_auc_score']}")
    print('\n')
    
    metrics_val = evaluate_model(
        model_clf,
        X_val,
        y_val)
    
    print('\n')
    print('Validation metrics')
    # print(f"report: {metrics_val['classification_report']}")
    print(f"acurácia: {metrics_val['accuracy_score']}")   
    print(f"f1: {metrics_val['f1_score']}")
    print(f"roc_auc: {metrics_val['roc_auc_score']}")
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
        undersampling=undersampling_method) 
    
    metric_orch.save_all_metrics(
        metrics_val, 
        model_config['model_name'], 
        dataset='validation', 
        undersampling=undersampling_method)      
    
    # 8. Save Model    
    path_model = os.path.join(
        config['init_path'],
        config['single_model']['pkl'],
        f"{model_config['model_name']}_{pipeline_name}.pkl")
    
    save_model(model_clf, path_model)
    
    # 9. Make predict
    predictions, probabilities = make_prediction(model_clf, X_val)
    
    path_data = os.path.join(
        config['init_path'],
        config['single_model']['predicts'])
    
    model_name = model_config['model_name']
    
    save_data(path_data, f"X_val_pred_{model_name}", predictions)
    save_data(path_data, f"X_val_proba_{model_name}", probabilities)    
    
    df_separation_plan = pd.concat([probabilities, y_val], axis=1)
    
    save_path = os.path.join(
        config['init_path'],
        config['single_model']['plots'],
        f"separation_plan_{model_name}.png")
    
    separation_plan_plot(
        save_path,
        model_name,
        df_separation_plan,
        config_pipe['features']['target'][0]
    )
    
   
if __name__ == "__main__":
    main_single_model_undersamplig(
        pipeline_name="Pipeline3", 
        model_name="LGBMClassifier",
        undersampling_method="TomekLinks",
        scoring='accuracy'
        )