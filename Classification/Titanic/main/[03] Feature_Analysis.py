import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from utils.logs import setup_logging
from Classification.Titanic.src.utils.config import load_config
from functions.feature_analysis import (
    MissingData, 
    CardinalityAnalysis, 
    ColsTypeAnalysis
    )

if __name__ == "__main__":
    
    log_path = os.path.join(project_root, "Classification/Titanic")
    logger = setup_logging(log_path)    
    
    config = load_config(['config'])

    dataset_path = os.path.join(
        config['init_path'], 
        config['data']['processed'],       
        "train_features.parquet"
        )
    plot_path = os.path.join(
        config['init_path'], 
        config['reports']['plots']
        )
    report_path = os.path.join(
        config['init_path'], 
        config['reports']['tables']        
        )    
 
    # Create Features
    logger.info("Feature Analysis begin")

    MissingData(dataset_path, plot_path)
    logger.info("Missing Data infomation saved in : %s", dataset_path)
    logger.info("Missing Data plot saved in : %s", plot_path)
    
    CardinalityAnalysis(dataset_path, report_path)
    logger.info("Cardinality report saved in : %s", dataset_path)
    
    ColsTypeAnalysis(dataset_path, report_path)
    logger.info("Columns type analysis saved in : %s", report_path)
    
    logger.info("Feature Analysis processed")