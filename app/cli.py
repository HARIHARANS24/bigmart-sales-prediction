import click
import pandas as pd
from sklearn.model_selection import train_test_split
from app.utils import load_data, load_config
from app.preprocessing import preprocess_data
from app.model import train_model, evaluate_model
from app.logger import get_logger
import os
import joblib
import json

logger = get_logger()

@click.group()
def cli():
    pass

@cli.command()
@click.option('--train-path', default=None, help='Path to training data CSV')
def eda(train_path):
    train_path = train_path or load_config()['data']['train_path']
    df = load_data(train_path)
    click.echo(df.describe(include='all'))
    click.echo('\nMissing values:')
    click.echo(df.isnull().sum())

@cli.command()
@click.option('--train-path', default=None, help='Path to training data CSV')
@click.option('--test-size', default=0.2, help='Test size fraction')
def train(train_path, test_size):
    config = load_config()
    train_path = train_path or config['data']['train_path']
    df = load_data(train_path)
    df = preprocess_data(df)
    
    target = config['data']['target']
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model_params = config['model']['random_forest']
    model = train_model(X_train, y_train, model_params)
    
    metrics = evaluate_model(model, X_val, y_val)
    
    model_version = 'v1'
    model_path = f'models/model_{model_version}.pkl'
    metrics_path = f'models/model_{model_version}_metrics.json'

    # Use joblib to save the model
    joblib.dump(model, model_path)

    # Save metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    click.echo(f'Model trained and saved as {model_path}')
    click.echo(f'Metrics: {metrics}')

@cli.command()
@click.option('--model-path', default='models/model_v1.pkl', help='Path to saved model')
@click.option('--data', required=True, help='JSON string with input data')
def predict(model_path, data):
    from app.preprocessing import preprocess_data

    model = joblib.load(model_path)
    input_dict = json.loads(data)
    df = pd.DataFrame([input_dict])
    df_processed = preprocess_data(df, is_train=False)

    X = df_processed.drop(columns=['Item_Identifier', 'Outlet_Identifier'], errors='ignore')
    pred = model.predict(X)[0]
    click.echo(f'Predicted Sales: {pred:.2f}')

if __name__ == '__main__':
    cli()
