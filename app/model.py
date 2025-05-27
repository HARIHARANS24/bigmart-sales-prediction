from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import json
from app.logger import get_logger

logger = get_logger()

def train_model(X, y, params):
    logger.info('Training RandomForestRegressor...')
    model = RandomForestRegressor(**params)
    model.fit(X, y)
    logger.info('Training complete.')
    return model

def evaluate_model(model, X_val, y_val):
    logger.info('Evaluating model...')
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    logger.info(f'MSE: {mse:.4f}, R2: {r2:.4f}')
    return {'mse': mse, 'r2': r2}

def save_model(model, metrics, model_path, metrics_path):
    dump(model, model_path)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    logger.info(f'Model saved to {model_path}')
    logger.info(f'Metrics saved to {metrics_path}')

def load_model(model_path):
    logger.info(f'Loading model from {model_path}')
    return load(model_path)
