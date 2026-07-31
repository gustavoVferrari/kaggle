import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
sys.path.insert(0, project_root)

from Classification.Titanic.src.data.data_gathering import DataGatheting, Unzipdata
from utils.logs import setup_logging

if __name__ == "__main__":
    log_path = os.path.join(project_root, "Classification/Titanic")
    logger = setup_logging(log_path)
    
    competition = "titanic"
    download_path = os.path.join(project_root, "Classification/Titanic/data/raw")
    zip_file_path = f"{download_path}/{competition}.zip"
    extract_path = download_path

    try:
        logger.info("Starting data gathering for competition: %s", competition)
        logger.info("Download path: %s", download_path)
        logger.info("Zip file path: %s", zip_file_path)
        logger.info("Extract path: %s", extract_path)

        # Download data
        logger.info("Downloading data from Kaggle")
        DataGatheting(competition, download_path)

        # Unzip data
        logger.info("Extracting downloaded files")
        Unzipdata(zip_file_path, extract_path)

        logger.info("Data gathering and extraction completed")
    except Exception:
        logger.exception("Data gathering and extraction failed")
        raise
