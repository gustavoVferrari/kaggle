
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)


def grid_search(X_train, y_train, models, metric, cv):     
    
        
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
            classifier)
        
        # Grid Search
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
    
    
    return dict_best_params   
   

 

