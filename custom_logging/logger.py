import logging
import os
import sys

def get_logger(name=__name__, log_dir=None, log_file="training.log"):
    """
    Sets up and returns a logger that writes to both the console and a file.
    
    Args:
        name (str): Name of the logger, usually __name__.
        log_dir (str, optional): Directory to save log files.
        log_file (str): Name of the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if len(logger.handlers) > 0:
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
