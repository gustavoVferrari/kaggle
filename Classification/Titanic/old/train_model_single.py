import os
import pickle
import pandas as pd
import json
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report, 
    f1_score,     
    roc_auc_score)
import yaml

import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)



def train_single_model(**args):
    
    X_train = args['X_train']
    y_train = args['y_train']  
    
          
    y_train = y_train.astype('int')
    
    # config
    seed_ = args['random_state'] 

    
    # select the best params
    best_params = pd.read_json(
        args['model_params'],
        lines=True)
    
    for col in best_params.select_dtypes(include="float").columns:
        if (best_params[col].dropna() % 1 == 0).all():
            best_params[col] = best_params[col].astype("Int64")
    
    select = best_params.model == args['model_name']
    
    best_params = (best_params.loc[select]
                   .reset_index(drop=True)
                   .copy())
    
    idx_model_params = (best_params
                        ['date_processamento']
                        .argmax())

    best_model_params = (best_params.loc[[idx_model_params]]
                         .drop(columns=['date_processamento'])
                         .dropna(axis=1)
                         .to_dict(orient='records')[0])
    
    del best_model_params['model']
    

    # apply best params    
    clf_model = args['model']
    clf_model.set_params(**best_model_params, random_state=seed_)  
    
    
    pipeline = make_pipeline(             
            clf_model)
    
    print("train model")
    pipeline.fit(X_train, y_train)
    
    y_pred_train = pipeline.predict(X_train)
    y_proba_train = pipeline.predict_proba(X_train)
    
    pd.DataFrame(y_pred_train).to_parquet(
        os.path.join(
            args['predict'],
            f"X_train_pred_{args['model_name']}.parquet"
            )
        )    
    
    pd.DataFrame(y_proba_train).to_parquet(
         os.path.join(
            args['predict'],
            f"X_train_proba_{args['model_name']}.parquet"
            )
        )    
    
    model_path = os.path.join(
        args['pkl'],
        f'model_{args["model_name"]}.pkl')
    
    with open(model_path, 'wb') as arquivo:
        pickle.dump(pipeline, arquivo)
    print(f"Model saved")
    
    print('Train score:')
    
    dict_score = {}
    dict_score['f1'] = f1_score(y_train, y_pred_train)
    dict_score['accuracy'] = accuracy_score(y_train, y_pred_train)
    dict_score['roc_auc_score'] = roc_auc_score(y_train, y_proba_train[:,1])
    
    dict_score['classificatio_report'] = classification_report(
        y_train,
        y_pred_train,
        output_dict=True)    
    
    print(dict_score)
    
    metrics_path = os.path.join(
        args['reports'],
        'train_model_metrics.json'
    )
    with open(metrics_path, 'w') as arquivo:
        json.dump(dict_score, arquivo)
    
    
if __name__ == "__main__":    
    print("Train Model carregado.")