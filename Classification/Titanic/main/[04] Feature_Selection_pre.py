import os
import sys
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from utils.logs import setup_logging
from Classification.Titanic.src.utils.config import load_config

from functions.feature_selection import FeatureSelectionOrchestrator
from Classification.Titanic.src.features.feature_eng import PreprocessingOrchestrator
from utils.plots import Pearson_correlation, Bar_plot

def Feature_Selection(config:dict, config_pipe:dict):
    
    log_path = os.path.join(project_root, "Classification/Titanic")
    logger = setup_logging(log_path)  
    
    logger.info("Feature selection preprocessing begin...")
    
    # 1. load dataset    
    X_train = pd.read_parquet(
        os.path.join(
            config['init_path'],
            config['data']['processed'],
            "train_features.parquet")
    )
    y_train = X_train[['Survived']]
    
    # 2. feat eng
    preprocessor = PreprocessingOrchestrator(
        numerical_con=config_pipe['features']['num_con'], 
        numerical_dis=config_pipe['features']['num_dis'], 
        categorical_var=config_pipe['features']['cat_var'])
    
    logger.info("Feature enginnering numerical continual cols: %s", config_pipe['features']['num_con'])
    logger.info("Feature enginnering numerical discrete cols: %s", config_pipe['features']['num_dis'])
    logger.info("Feature enginnering categorical cols: %s", config_pipe['features']['cat_var'])    
    
    
    pipe = preprocessor.apply("preprocessing")        
    X_train_trans = pipe.fit_transform(X_train)    
    
    # 3. feat selection    
    feature_selection = FeatureSelectionOrchestrator()

    QuiSquare = feature_selection.apply(
        "QuiSquare", 
        X_train_trans.filter(like='categorical'), 
        y_train)
    logger.info("Feature selection QuiSquare ran with sucess")
    
    Anova = feature_selection.apply(
        "Anova",
        X_train_trans.filter(like='numerical_pipe_con'),
        y_train)
    logger.info("Feature selection Anova ran with sucess")
    
    mi = feature_selection.apply(
        "MutualInformationClassif", 
        X_train_trans.filter(like='numerical'), 
        y_train)
    logger.info("Feature selection MutualInformationClassif ran with sucess")
    
    corr = feature_selection.apply(
        "PearsonCorrelation", 
        X_train_trans.filter(like='numerical'), 
        y_train)
    logger.info("Feature selection PearsonCorrelation ran with sucess")
        
    path_ =  os.path.join(
        config['init_path'],
        config['reports']['plots'])
    
    Pearson_correlation(corr, title = "corr", path=path_)
    
    Bar_plot(QuiSquare, title = "Qui_square" , path=path_)
    
    Bar_plot(Anova, title = "Anova" , path=path_)
    
    Bar_plot(mi, title = "Mutual_information", path=path_)
    
    logger.info("Feature selection completed")
    
def main():
    config, config_pipe = load_config(['config', 'config_pipe'])
    Feature_Selection(config, config_pipe)
    
if __name__ == "__main__":
    main()
    
