import os
import logging
from datetime import datetime

def setup_logger(id: int, infer_file_path: str) -> logging.Logger:
    logger = logging.getLogger(f"infer_{id}")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(infer_file_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

def close_logger(logger: logging.Logger):
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

def log_status(res_file_path, status):
    log_dir = os.path.dirname(res_file_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(res_file_path, "a") as log_file:
        log_file.write(f"{timestamp}|!{status}")

def get_system_logger(log_path: str = "system.log") -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logger = logging.getLogger("system")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger