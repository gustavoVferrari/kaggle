
import datetime
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    accuracy_score,     
    roc_auc_score)
from sklearn.model_selection import GridSearchCV

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)



def model_selection(**args):     
    
    # load daataset    
    X_train = args['X_train']
    y_train = args['y_train']           
    X_val = args['X_val']               
    y_val = args['y_val']                
   
    
    # ensure int type
    y_train = y_train.astype('int')
    y_val = y_val.astype('int')
    
    # config
    seed_ = args['random_state']     
    models = args['models']
    metric = args['metric']
    cv_ = args['cv']
    
    # run moel selection    
    dict_best_model_params = {}
    dict_metrics = {}

    for model in models.keys():
        
        classifier = models[model][0]
        params = models[model][1]
        
        print("Running: ", classifier)
        
        len_ = len([k.split('__')[0] for k in params.keys()][0])
        name_ = [k.split('__')[0] for k in params.keys()][0]

        
        grid_pipeline = make_pipeline(
            classifier
            )
        
        # Grid Search
        grid = GridSearchCV(
            grid_pipeline,
            scoring=metric,
            param_grid=params, 
            cv=cv_,
            n_jobs=-1)
        
        grid_fit = grid.fit(X_train, y_train)
        
        pred_proba_test = grid.predict_proba(X_val)[:,1]
        pred_test = grid.predict(X_val)
        
        dict_metrics[model] = [
            f1_score(y_val, pred_test),
            accuracy_score(y_val, pred_test), 
            roc_auc_score(y_val, pred_proba_test)
            ] 
        
        # Best params    
        dict_best_params = {}
        for k, v in grid_fit.best_params_.items():
            dict_best_params[k[len_+2:]] = v
            
        dict_best_model_params[model] =  dict_best_params
    
    metrics =  pd.DataFrame(
        dict_metrics, 
        index=['f1', 'accuracy','roc']).T 
    
    metrics['model'] = metrics.index
    metrics['data_processamento'] = (datetime
                                     .datetime
                                     .now()
                                     .strftime('%Y-%m-%d %H:%M:%S'))
     
          
    metrics.to_json(
        os.path.join(
            args['reports'], 
            'metrics_model_selection_sklearn.jsonl'
                     ),
        lines=True, 
        orient='records',
        mode='a'
        )
    
    best_idx = (pd.DataFrame(
        dict_metrics, 
        index=['f1', 'accuracy', 'roc'])
                .loc['roc'].argmax())
    
    best_idx = (pd.DataFrame(
        dict_metrics, 
        index=['f1', 'accuracy', 'roc'])
                .columns[best_idx])
    
    print("Best model:", best_idx)
    
    # Cross-Validation with best model
    skf = StratifiedKFold(
        n_splits=cv_, 
        random_state=seed_, 
        shuffle=True)
    
    dict_train_idx={}
    dict_val_idx={}
    dict_fit={}
    dict_predict={}
    dict_predict_proba={}
    dict_score={}
    
    clf_best_params = models[best_idx][0]
    clf_best_params.set_params(**dict_best_model_params[best_idx])
    print(dict_best_model_params[best_idx])
    
    file_dict = os.path.join(
        args['reports'],
        'best_model_params.jsonl'
        )
    
    dict_best_params={}
    dict_best_params[best_idx] = dict_best_model_params[best_idx]
    dict_best_params[best_idx]['date_processamento'] = (datetime
                                              .datetime
                                              .now()
                                              .strftime('%Y-%m-%d %H:%M:%S'))
    
    dict_best_params[best_idx] = {**{'model': best_idx}, **dict_best_params[best_idx]}
    
    pd.DataFrame(dict_best_params).T.to_json(
        file_dict,
        orient='records',
        lines=True,
        mode='a'
        )
          
    
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):

        dict_train_idx[i] = [train_index]
        dict_val_idx[i] = [val_index]

        model_pipe_stf = make_pipeline(clf_best_params)
        
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
        columns=['fold', 'score'])
    
    best_fold = df_score.iloc[df_score.score.argmax()]['fold']
    
    df_score['date_processamento'] = (datetime
                                      .datetime
                                      .now()
                                      .strftime('%Y-%m-%d %H:%M:%S')
                                      )
    
    df_score['model'] = best_idx
    
    df_score.to_json(
        os.path.join(args['reports'], 
                     f'cv_score.jsonl'),
        lines=True,
        orient='records',
        mode='a'
        )    
    
    save_path = os.path.join(
        args['plot'],
        f"cv_score_{best_idx}.png"
    )     
     
    sns.pointplot(
        data=df_score, 
        y='score', 
        x='fold')
    
    plt.title('score per fold')
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close() 
    
    for fold in range(0, args['cv']):
        tmp = pd.DataFrame(y_train.iloc[dict_val_idx[fold][0]]).assign(escore = dict_predict_proba[fold])
        sns.histplot(data=tmp,x='escore', hue='Survived')
        plt.title(f'Fold : {fold}')
        save_path = os.path.join(
            args['plot'],
            f"separation_plane_{fold}_{best_idx}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()   
   

if __name__ == "__main__":
   print("Model Selection Sklearn carregado.")