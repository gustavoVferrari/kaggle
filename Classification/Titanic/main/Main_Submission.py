import yaml
import pandas as pd
import numpy as np
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)



def main_submission(threshold=0.5):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    
    
    # Get feature eng data
    pipeline_name = "Pipeline2"
    model_name = "rf"
    
    X_test = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['feature_eng'],
            f"X_test_feat_eng_{pipeline_name}.parquet")
   )
   
    y_test_id = pd.read_parquet(
       os.path.join(
           config['init_path'],
           config['data']['processed'],
            f"test_features.parquet")
   )

    y_test_id = y_test_id[['PassengerId']].copy()
    
    X_test.drop(
        columns = config_model['single_model']['cols_2_drop'],
        inplace=True
    )   
    
    model_path = os.path.join(
           config['init_path'],
           config['single_model']['pkl'],
            f"{model_name}.pkl")
    # open model    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
    # predict    
    # y_test_id.loc[:,f'{config_pipe['features']['target'][0]}'] = model.predict(X_test)
    
    y_test_id.loc[:,f'{config_pipe['features']['target'][0]}'] = np.where(model.predict_proba(X_test)[:,1]>=threshold,1,0)
    
    y_test_id.to_csv(
        os.path.join(
           config['init_path'],
           config['data']['submission'],
            f'submission_{model_name}.csv'),
        index=False)
    
if __name__ == "__main__":
    main_submission(threshold=0.41)