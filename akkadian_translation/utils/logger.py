import logging
import os
import sys

def get_logger(name=__name__, log_dir="logs", log_file="training.log"):
    """
    Sets up and returns a logger that writes to both the console and a file.
    
    Args:
        name (str): Name of the logger, usually __name__.
        log_dir (str): Directory where the log file will be saved.
        log_file (str): Name of the log file.
        
    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers are already configured to prevent duplication
    if len(logger.handlers) > 0:
        return logger
        
    # Format for logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional, if log_dir is provided)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path, mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
