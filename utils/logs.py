import os
import sys
import logging

def setup_logging(project_root, process_name=None):
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "data_gathering.log")
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if process_name is None:
        process_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]

    return logging.getLogger(process_name)
