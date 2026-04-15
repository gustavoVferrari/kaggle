

import os
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    AdaBoostClassifier, 
    VotingClassifier,
    GradientBoostingClassifier, 
    HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV



def voting_model(X_train:pd.DataFrame, y_train:pd.DataFrame, cv=3 ,metric="roc_auc"):           

    seed_ = 23
    
    models = dict(
    rf = [
        RandomForestClassifier(random_state=seed_),
        {'randomforestclassifier__n_estimators':[200, 250, 500],
         'randomforestclassifier__criterion': ['gini', 'entropy'], 
         'randomforestclassifier__max_depth': [None, 5, 10, 20],
         'randomforestclassifier__min_samples_split' :[2,4,6]}
        ]
    ,
    ab = [
        AdaBoostClassifier(random_state=seed_),
        {'adaboostclassifier__n_estimators':[150, 200, 250, 300],
         'adaboostclassifier__learning_rate': [0.01, 0.1, 0.001]}
        ]
    ,    
    gb = [
        GradientBoostingClassifier(random_state=seed_),
        {'gradientboostingclassifier__n_estimators':[100, 150, 200, 250],
         'gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.001]}
        ]
    ,
    lr = [
        LogisticRegression(),
        {'logisticregression__solver':['saga', 'lbfgs', 'liblinear'],
         'logisticregression__C': [0.1, 1]}
        ],
    
    sv = [
        SVC(probability=True),
        {'svc__kernel':['rbf', 'poly'],
         'svc__C': [0.1, 1]}
        ],     
    ml = [
        MLPClassifier(random_state=seed_),
        {'mlpclassifier__hidden_layer_sizes':[10, 20, 30],
         'mlpclassifier__activation': ['relu', 'tanh'],
         'mlpclassifier__learning_rate_init':[0.01, 0.001]
         }
        ],
    hg = [HistGradientBoostingClassifier(),
          {'histgradientboostingclassifier__learning_rate': [0.01, 0.1, 0.001],
           'histgradientboostingclassifier__max_iter': [50, 100, 150]}
    ]
          )
    
    # run moel selection    
    dict_best_model_params = {}

    for model in models.keys():
        
        classifier = models[model][0]
        params = models[model][1]        
        
        len_ = len([k.split('__')[0] for k in params.keys()][0])
        name_ = [k.split('__')[0] for k in params.keys()][0]

        grid_pipeline = make_pipeline(classifier)
        
        grid = GridSearchCV(
            grid_pipeline,
            scoring=metric,
            param_grid=params, 
            cv=cv,
            n_jobs=-1)
        
        grid_fit = grid.fit(X_train, y_train)      
      
        
        # Best params    
        dict_best_params = {}
        for k, v in grid_fit.best_params_.items():
            dict_best_params[k[len_+2:]] = v
            
        dict_best_model_params[model] =  dict_best_params
        
    return dict_best_model_params
        
        
def models():
        models = dict(
        rf = RandomForestClassifier(),        
        ab = AdaBoostClassifier(),   
        gb = GradientBoostingClassifier(),
        lr = LogisticRegression(),
        sv = SVC(probability=True),
        ml = MLPClassifier(),
        hg = HistGradientBoostingClassifier())
        
        return models
        
    
if __name__ == "__main__":    
    print("Voting Model carregado.")

    
    
    

    
