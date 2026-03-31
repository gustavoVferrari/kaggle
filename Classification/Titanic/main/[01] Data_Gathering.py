import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, project_root)

from src.data.data_gathering import DataGatheting, Unzipdata

if __name__ == "__main__":
    
    competition = "titanic"
    download_path = os.path.join(project_root, "data/raw")
    zip_file_path = f"{download_path}/{competition}.zip"
    extract_path = download_path

    # Download data
    DataGatheting(competition, download_path)

    # Unzip data
    Unzipdata(zip_file_path, extract_path)

    print("Data gathering and extraction completed.")