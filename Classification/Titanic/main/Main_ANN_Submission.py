import yaml
import pandas as pd
import numpy as np
import os
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)



def main_ann_submission(pipeline_name: str):
    
    # 1. Carregar configurações
    with open(os.path.join(project_root, "Classification/Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # pipeline selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)
    
    # model selection    
    with open(os.path.join(project_root, "Classification/Titanic/config/model.yaml"), "r") as f:
        config_model = yaml.safe_load(f)

    
    
    # Get feature eng data    
     
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
        columns = config_model['ann_model']['cols_2_drop'],
        inplace=True
    )   
    
    model_path = os.path.join(
           config['init_path'],
           config['ann_model']['h5'],
            f"ann_model_{pipeline_name}.h5")
    # open model    
    with open(model_path, "rb") as file:
        model = pickle.load(file)
        
     
    y_test_id.loc[:,f"{config_pipe['features']['target'][0]}"] = np.where(model.predict_proba(X_test)[:,1]>=0.5,1,0)
    
    y_test_id.to_csv(
        os.path.join(
           config['init_path'],
           config['data']['submission'],
            f'submission_ann_model_{pipeline_name}.csv'),
        index=False)
    
    print("dados salvos com sucesso")
    
if __name__ == "__main__":
    main_ann_submission(pipeline_name='Pipeline3')