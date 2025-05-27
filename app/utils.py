import pandas as pd
import yaml
from app.logger import get_logger

logger = get_logger()

def load_data(path):
    logger.info(f'Loading data from {path}')
    try:
        df = pd.read_csv(path)
        logger.info(f'Data shape: {df.shape}')
        return df
    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        raise

def load_config(path='config.yaml'):
    logger.info(f'Loading config from {path}')
    with open(path) as f:
        config = yaml.safe_load(f)
    return config
