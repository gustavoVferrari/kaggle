import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from Classification.Titanic.src.data.feature_creation import FeatureCreation
from Classification.Titanic.src.utils.config import load_config
from utils.logs import setup_logging

if __name__ == "__main__":
    
    log_path = os.path.join(project_root, "Classification/Titanic")
    logger = setup_logging(log_path)    
    
    try:
        logger.info("Starting feature creation")
        
        config = load_config(['config'])
        logger.info("Configuration loaded")
        
        dataset_path = os.path.join(
            config['init_path'], 
            config['data']['raw']
            )
        
        save_path = os.path.join(
            config['init_path'], 
            config['data']['processed']
            )
        
        logger.info("Raw dataset path: %s", dataset_path)
        logger.info("Processed dataset output path: %s", save_path)
        
        # Create Features
        logger.info("Creating training features")
        FeatureCreation(dataset_path, save_path)
        logger.info("Training features saved in: %s", save_path)

        logger.info("Creating test features")
        FeatureCreation(dataset_path, save_path, train=False)
        logger.info("Test features saved in: %s", save_path)
        
        logger.info("Feature creation completed")
    except Exception:
        logger.exception("Feature creation failed")
        raise
