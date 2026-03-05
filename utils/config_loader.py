import yaml
import os

def load_config(config_path):
    """
    Loads a YAML configuration file to drive the experiment.
    
    Args:
        config_path (str): Path to the YAML file.
        
    Returns:
        dict: Parsed configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config
