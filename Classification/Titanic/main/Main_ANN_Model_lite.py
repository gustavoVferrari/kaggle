from xml.parsers.expat import model

import yaml
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

from functions.make_dataset import save_data
from functions.train_model import save_model
from functions.evaluate_model import evaluate_model, MetricsOrchestrator
from functions.predict_model import make_prediction
from functions.ann_model import KerasBinaryClassifier

def main_ann_model_lite(pipeline_name: str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Titanic/config/model.yaml"), "r") as f:
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

    # Drop columns
    X_train.drop(
        columns=config_model['ann_model']['cols_2_drop'],
        inplace=True)
    
    X_val.drop(
        columns=config_model['ann_model']['cols_2_drop'],
        inplace=True)   


    # 3. Model Selection 
    model = KerasBinaryClassifier(input_dim=X_train.shape[1], epochs=70)    
        
    # 4. train model
    clf = model.fit(X_train, y_train)             
    
    # 5. Evaulate model
    metrics_train = evaluate_model(clf, X_train, y_train)
    
    print('train metrics')
    print(f"report: {metrics_train['classification_report']}")
    print(f"acurácia: {metrics_train['accuracy_score']}")   
    print(f"f1: {metrics_train['f1_score']}")
    print(f"roc_auc: {metrics_train['roc_auc_score']}")
    print('\n')
    
    metrics_val = evaluate_model(clf, X_val, y_val)
    
    print('Validation metrics')
    print(f"report: {metrics_val['classification_report']}")
    print(f"acurácia: {metrics_val['accuracy_score']}")   
    print(f"f1: {metrics_val['f1_score']}")
    print(f"roc_auc: {metrics_val['roc_auc_score']}")
    print('\n')
    
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
    predictions, probabilities = make_prediction(clf, X_val)
        
    path_data = os.path.join(
        config['init_path'],
        config['ann_model']['predicts'])     
    
    save_data(path_data, f"X_val_pred_ann_model", predictions)
    save_data(path_data, f"X_val_proba_ann_model", probabilities)    
    
   
   
if __name__ == "__main__":
    main_ann_model_lite(pipeline_name='Pipeline3')