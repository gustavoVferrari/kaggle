import os
import sys
import yaml
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from functions.feature_selection import FeatureSelectionOrchestrator
from utils.plots import Pearson_correlation, Bar_plot
from utils.utils import to_jsonl


def Main_Feature_Selection(pipeline_name: str):
    
        # 1. Carregar configurações
    with open(os.path.join(project_root, "Classification/Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)        
  
    # 1. load dataset    
    X_train = pd.read_parquet(
        os.path.join(
            config['init_path'],
            config['data']['feature_eng'],
            f"X_train_feat_eng_{pipeline_name}.parquet")
    )
        
    y_train = pd.read_parquet(
        os.path.join(
            config['init_path'],
            config['data']['feature_eng'],
            f"y_train_feat_eng_{pipeline_name}.parquet")
    )   
    
    # 2. feat selection    
    feature_selection = FeatureSelectionOrchestrator()
        
    Anova = feature_selection.apply("Anova", X_train, y_train)    
    mi = feature_selection.apply("MutualInformationClassif", X_train, y_train)    
    corr = feature_selection.apply("PearsonCorrelation", X_train, y_train)    
    smart_corr = feature_selection.apply("SmartCorrelatedSelection", X_train, y_train)
    print("-------------------------------------------")
    print("Anova")
    print(Anova)
    print("-------------------------------------------")
    print("smart_corr")
    print(smart_corr)
    print("-------------------------------------------")
    print("mi")
    print(mi)
    print("-------------------------------------------")
    print("corr")
    print(corr)
    print("-------------------------------------------")
    
    
    path_sc = os.path.join(
            config['init_path'],
            config['reports']['tables'],
            f'corr_features_{pipeline_name}.jsonl')
    
    to_jsonl(smart_corr, path_sc, mode='append')
        
    path_ =  os.path.join(
        config['init_path'],
        config['reports']['plots'])
    
    Pearson_correlation(corr, title = f"corr_{pipeline_name}", path=path_) 
    
    Bar_plot(Anova, title = f"Anova_{pipeline_name}" , path=path_)
    
    Bar_plot(mi, title = f"Mutual_information_{pipeline_name}", path=path_)
    
if __name__ == "__main__":
    print("-------------------------------------------")
    Main_Feature_Selection(pipeline_name = "Pipeline3")
    print("-------------------------------------------")
    print("\n")
    Main_Feature_Selection(pipeline_name = "Pipeline2")
    print("-------------------------------------------")
    print("\n")
    Main_Feature_Selection(pipeline_name = "Pipeline1")