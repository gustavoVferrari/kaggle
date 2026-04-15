import os
import sys
import yaml
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

from functions.feature_selection import FeatureSelectionOrchestrator
from Titanic.src.features.feature_eng import PreprocessingOrchestrator
from utils.plots import Pearson_correlation, Bar_plot

def Main_Feature_Selection():
    
        # 1. Carregar configurações
    with open(os.path.join(project_root, "Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        
    with open(os.path.join(project_root, "Titanic/config/pipeline.yaml"), "r") as f:
        config_pipe = yaml.safe_load(f)  
        
    X_train = pd.read_parquet(
        os.path.join(
            config['init_path'],
            config['data']['processed'],
            "train_features.parquet")
    )
    y_train = X_train[['Survived']]
    
    preprocessor = PreprocessingOrchestrator(
        numerical_con=config_pipe['features']['num_con'], 
        numerical_dis=config_pipe['features']['num_dis'], 
        categorical_var=config_pipe['features']['cat_var'])
    
    pipe = preprocessor.apply("preprocessing")        
    X_train_trans = pipe.fit_transform(X_train)    
    
        
    feature_selection = FeatureSelectionOrchestrator()

    QuiSquare = feature_selection.apply(
        "QuiSquare", 
        X_train_trans.filter(like='categorical'), 
        y_train)
    
    Anova = feature_selection.apply(
        "Anova",
        X_train_trans.filter(like='numerical_pipe_con'),
        y_train)
    
    mi = feature_selection.apply(
        "MutualInformationClassif", 
        X_train_trans.filter(like='numerical'), 
        y_train)
    
    corr = feature_selection.apply(
        "PearsonCorrelation", 
        X_train_trans.filter(like='numerical'), 
        y_train)
        
    path_ =  os.path.join(
        config['init_path'],
        config['reports']['plots'])
    
    Pearson_correlation(corr, title = "corr", path=path_)
    
    Bar_plot(QuiSquare, title = "Qui_square" , path=path_)
    
    Bar_plot(Anova, title = "Anova" , path=path_)
    
    Bar_plot(mi, title = "Mutual_information", path=path_)
    
if __name__ == "__main__":
    Main_Feature_Selection()