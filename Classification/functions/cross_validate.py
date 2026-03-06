import pandas as pd
from sklearn.pipeline import make_pipeline
from datetime import datetime
from sklearn.model_selection import StratifiedKFold

def cross_validate(X_train:pd.DataFrame, y_train:pd.DataFrame, model_config, best_model_params:dict, n_splits=5, random_state=23):

    model = model_config['model']

    skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    model.set_params(**best_model_params)

    dict_train_idx={}
    dict_val_idx={}
    dict_fit={}
    dict_predict={}
    dict_predict_proba={}
    dict_score={}


    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

            dict_train_idx[i] = [train_index]
            dict_val_idx[i] = [val_index]

            model_pipe_stf = make_pipeline(model)
            
            dict_fit[i] = model_pipe_stf.fit(
                X_train.iloc[dict_train_idx[i][0]], 
                y_train.iloc[dict_train_idx[i][0]])
            
            
            dict_predict[i] = dict_fit[i].predict(X_train.iloc[dict_val_idx[i][0]])
            
            dict_predict_proba[i] = dict_fit[i].predict_proba(X_train.iloc[dict_val_idx[i][0]])[:,1]
            
            dict_score[i] = dict_fit[i].score(
                X_train.iloc[dict_val_idx[i][0]], 
                y_train.iloc[dict_val_idx[i][0]])
            
    df_score = pd.DataFrame(
        dict_score.items(), 
        columns=['fold', 'accuracy'])
    
    df_score['model'] = model_config['model_name']
    df_score['timestamp'] = datetime.now().isoformat()
    
    return df_score