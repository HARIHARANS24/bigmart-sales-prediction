import sys
import os

# Add parent directory to sys.path so 'app' modules can be imported correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify
import pandas as pd
from app.preprocessing import preprocess_data
from app.logger import get_logger
import joblib

app = Flask(__name__)
logger = get_logger()

# Load model with joblib (make sure you saved it with joblib.dump)
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_v1.pkl')
model = joblib.load(model_path)

@app.route('/')
def index():
    return "Big Mart Sales Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Empty request body or invalid JSON'}), 400

        df = pd.DataFrame([data])
        logger.info(f"Received input for prediction: {df.to_dict(orient='records')}")

        # Preprocess the input data (pass is_train=False)
        processed = preprocess_data(df, is_train=False)

        # Drop ID columns if present
        X = processed.drop(columns=['Item_Identifier', 'Outlet_Identifier'], errors='ignore')

        # Drop target column if present to avoid errors
        if 'Item_Outlet_Sales' in X.columns:
            X = X.drop(columns=['Item_Outlet_Sales'])

        # Predict sales
        prediction = model.predict(X)

        return jsonify({'predicted_sales': float(prediction[0])})

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run app on all interfaces (0.0.0.0) on port 5000 with debug mode on
    app.run(host='0.0.0.0', debug=True, port=5000)
