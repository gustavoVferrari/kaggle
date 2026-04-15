import os
import sys
import yaml
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

with open(os.path.join(project_root, "Classification/Titanic/config/config.yaml"), "r") as f:
        config = yaml.safe_load(f)

from functions.feature_analysis import (
    MissingData, 
    CardinalityAnalysis, 
    ColsTypeAnalysis
    )

if __name__ == "__main__":

    dataset_path = os.path.join(
        config['init_path'], 
        "data", 
        "processed", 
        "train_features.parquet"
        )
    plot_path = os.path.join(
        config['init_path'], 
        "reports", 
        "plots"
        )
    report_path = os.path.join(
        config['init_path'], 
        "reports", 
        "jsonl"
        )    
 
    # Create Features
    MissingData(dataset_path, plot_path)
    
    CardinalityAnalysis(dataset_path, report_path)
    
    ColsTypeAnalysis(dataset_path, report_path)
    
    print("Feature Analysis completed.")