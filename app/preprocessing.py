import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
from app.logger import get_logger
import os

logger = get_logger()

ENCODER_PATH = 'models/label_encoders.pkl'  # Path to save/load encoders
FEATURES_PATH = 'models/feature_columns.pkl'  # Path to save/load feature columns

def preprocess_data(df, is_train=True):
    df = df.copy()

    # One-hot encode 'Item_Type'
    df = pd.get_dummies(df, columns=['Item_Type'])

    # Fill missing values safely
    if df['Item_Weight'].isnull().any():
        median_weight = df['Item_Weight'].median()
        df['Item_Weight'].fillna(median_weight, inplace=True)
        logger.info(f'Filled missing Item_Weight with median: {median_weight}')

    if df['Outlet_Size'].isnull().any():
        mode_size = df['Outlet_Size'].mode()
        if not mode_size.empty:
            mode_size_value = mode_size[0]
            df['Outlet_Size'].fillna(mode_size_value, inplace=True)
            logger.info(f'Filled missing Outlet_Size with mode: {mode_size_value}')
        else:
            logger.warning('Outlet_Size mode is empty, missing values not filled')

    # Normalize 'Item_Fat_Content'
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'
    })

    # Feature engineering: Create Item_Category from Item_Identifier
    df['Item_Category'] = df['Item_Identifier'].str[:2]

    categorical_cols = ['Item_Fat_Content', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type', 'Item_Category']

    if is_train:
        # Fit LabelEncoders and save them
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            logger.info(f'Fitted and encoded {col}')
        # Save encoders
        joblib.dump(encoders, ENCODER_PATH)
        logger.info(f'Saved LabelEncoders to {ENCODER_PATH}')
    else:
        # Load encoders and transform
        if os.path.exists(ENCODER_PATH):
            encoders = joblib.load(ENCODER_PATH)
            for col in categorical_cols:
                le = encoders.get(col)
                if le:
                    # Map unknown categories to -1
                    def safe_transform(val):
                        if val in le.classes_:
                            return le.transform([val])[0]
                        else:
                            return -1
                    df[col] = df[col].astype(str).map(safe_transform)
                    logger.info(f'Transformed {col} using saved encoder')
                else:
                    logger.warning(f'No encoder found for {col}, skipping encoding')
        else:
            logger.error(f'Encoder file not found at {ENCODER_PATH}')
            raise FileNotFoundError(f'Encoder file not found at {ENCODER_PATH}')

    # Feature interaction
    df['Visibility_by_Weight'] = df['Item_Visibility'] / df['Item_Weight']

    # Drop identifiers
    df = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'], errors='ignore')

    # Align columns during prediction
    if is_train:
        # Save feature columns for later use during prediction
        feature_cols = df.columns.tolist()
        joblib.dump(feature_cols, FEATURES_PATH)
        logger.info(f'Saved feature columns to {FEATURES_PATH}')
    else:
        # Load feature columns and reindex dataframe
        if os.path.exists(FEATURES_PATH):
            feature_cols = joblib.load(FEATURES_PATH)
            # Add any missing columns with default 0 (for one-hot features)
            for col in feature_cols:
                if col not in df.columns:
                    df[col] = 0
            # Reorder columns to match training
            df = df[feature_cols]
            logger.info('Aligned dataframe columns to feature columns used in training')
        else:
            logger.error(f'Feature columns file not found at {FEATURES_PATH}')
            raise FileNotFoundError(f'Feature columns file not found at {FEATURES_PATH}')

    # Convert all columns to numeric
    try:
        df = df.apply(pd.to_numeric, errors='raise')
        logger.info('Successfully converted all columns to numeric')
    except Exception as e:
        logger.error(f'Error converting columns to numeric: {e}')
        raise

    logger.info(f'Preprocessed data shape: {df.shape}')
    return df
