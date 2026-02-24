import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.data.feature_creation import FeatureCreation

if __name__ == "__main__":

    dataset_path = os.path.join(
        project_root, 
        "data", 
        "raw"
        )
    save_path = os.path.join(
        project_root, 
        "data", 
        "processed"
        )
    
    # Create Features
    FeatureCreation(dataset_path, save_path)
    FeatureCreation(dataset_path, save_path, train=False)
    print("Data gathering and extraction completed.")