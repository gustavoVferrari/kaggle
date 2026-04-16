import os
import pandas as pd
import sys
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from utils.plots import separation_plan_plot
from functions.make_dataset import save_data
from functions.train_model import train_voting_model, save_model
from functions.evaluate_model import evaluate_model, MetricsOrchestrator
from functions.predict_model import make_prediction
from functions.voting_model import voting_model, models

def main_voting_model_lite(pipeline_name:str, scoring:str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Classification/Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    print("Iniciando pipeline de Machine Learning...")


    # Get feature eng data     
    
    # Datasets
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
    X_train.drop(
        columns=config_model['single_model']['cols_2_drop'],
        inplace=True)
    
    X_val.drop(
        columns=config_model['single_model']['cols_2_drop'],
        inplace=True)   


    # 3. Model Selection        
    best_params = voting_model(
        X_train, 
        y_train,
        scoring=scoring
        )     

    # 4. train model
    clf = train_voting_model(
        X_train,
        y_train, 
        models(), 
        best_params
        )        
    
    # cross validation
    # df_cv = cross_validate_StratifiedKFold(
    #     X_train, 
    #     y_train, 
    #     model_clf,
    #     model_config,
    #     scoring=scoring
    #     )
    # print("\n")
    # print(df_cv)
    # print("\n")
    # print(f"Mean train score {df_cv['scoring'].unique()[0]}: {df_cv['train_score'].mean()} +- {df_cv['train_score'].std()}")
    # print(f"Mean val score {df_cv['scoring'].unique()[0]}: {df_cv['val_score'].mean()} +- {df_cv['val_score'].std()}")
    
    
    # save_path = os.path.join(
    #     config['init_path'],
    #     config['single_model']['plots'],
    #     f"cross_validation_{model_name}.png")
    
    # cross_validation_plot(save_path, model_name, df_cv)
    
    # path_cv = os.path.join(
    #     config['init_path'],
    #     config['single_model']['tables'],
    #     "cross_validate.jsonl")    
    # to_jsonl(df_cv, path_cv, mode='append')
    
    # 5. Evaulate model
    metrics_train = evaluate_model(clf, X_train, y_train)
    
    print("\n")
    print('train metrics')
    # print(f'report: {metrics_train['classification_report']}')
    print(f'acurácia: {metrics_train["accuracy_score"]}')   
    print(f'f1: {metrics_train["f1_score"]}')
    print(f'roc_auc: {metrics_train["roc_auc_score"]}')
    print('\n')
    
    metrics_val = evaluate_model(clf, X_val, y_val)
    
    print("\n")
    print('Validation metrics')
    # print(f'report: {metrics_val['classification_report']}')
    print(f'acurácia: {metrics_val["accuracy_score"]}')   
    print(f'f1: {metrics_val["f1_score"]}')
    print(f'roc_auc: {metrics_val["roc_auc_score"]}')
    print('\n')
    
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
    save_model(clf, path_model)
    
    # 7. Make predict 
    predictions, probabilities = make_prediction(clf, X_val)
        
    path_data = os.path.join(
        config['init_path'],
        config['voting_model']['predicts'])    
    model_name = 'voting_model'
    
    save_data(
        path_data, 
        f"X_val_pred_{model_name}", 
        predictions
        )
    save_data(
        path_data, 
        f"X_val_proba_{model_name}",
        probabilities
        )        
    
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
    main_voting_model_lite(
        pipeline_name="Pipeline3",
        scoring='accuracy'
    )