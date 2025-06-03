import yaml
import os

def load_config(config_path):
    """
    Load YAML configuration file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dict containing configuration settings
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"ERROR loading config from {config_path}: {e}")
        return None
        
def get_default_config_path(config_name):
    """
    Get the path to a default configuration file
    
    Args:
        config_name: Name of the configuration file
        
    Returns:
        Path to the configuration file
    """
    return os.path.join("analysis", "configs", config_name)
