import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.features.feature_analysis import (
    MissingData, 
    CardinalityAnalysis, 
    ColsTypeAnalysis
    )

if __name__ == "__main__":

    dataset_path = os.path.join(
        project_root, 
        "data", 
        "processed", 
        "train_features.parquet"
        )
    plot_path = os.path.join(
        project_root, 
        "reports", 
        "plots"
        )
    report_path = os.path.join(
        project_root, 
        "reports", 
        "jsonl"
        )    
 
    # Create Features
    MissingData(dataset_path, plot_path)
    
    CardinalityAnalysis(dataset_path, report_path)
    
    ColsTypeAnalysis(dataset_path, report_path)
    
    print("Feature Analysis completed.")