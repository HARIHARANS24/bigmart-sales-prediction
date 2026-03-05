# BigMart Sales Prediction

🚀 End-to-end machine learning pipeline for BigMart sales prediction using Flask API, Streamlit UI, Docker, and modular Python architecture — includes EDA, model training, real-time prediction, and testing.

---

## 🗂️ Project Structure 
 
``` 
Directory structure:
└── hariharans24-bigmart-sales-prediction/
    ├── README.md
    ├── config.yaml
    ├── docker-compose.yml
    ├── Dockerfile
    ├── Dockerfile.streamlit
    ├── LICENSE
    ├── requirements.txt
    ├── run.txt
    ├── app/
    │   ├── __init__.py
    │   ├── api.py
    │   ├── cli.py
    │   ├── logger.py
    │   ├── model.py
    │   ├── preprocessing.py
    │   ├── streamlit_app.py
    │   ├── utils.py
    │   └── visualization.py
    ├── data/
    │   ├── test.csv
    │   └── train.csv
    ├── models/
    │   ├── feature_columns.pkl
    │   ├── label_encoders.pkl
    │   ├── model_v1.pkl
    │   └── model_v1_metrics.json
    ├── notebooks/
    │   ├── BigMart_EDA.ipynb
    │   └── Model_Training.ipynb
    └── tests/
        ├── test_preprocessing.py
        └── test_utils.py
```

---

## 📦 Features

- 🔍 **Exploratory Data Analysis**
  - Comprehensive data visualization
  - Statistical analysis
  - Feature correlation studies
  - Data quality assessment

- 🧠 **Machine Learning Pipeline**
  - Automated data preprocessing
  - Feature engineering
  - Model training and evaluation
  - Model versioning and metrics tracking

- 🔁 **API Services**
  - RESTful Flask API
  - Real-time prediction endpoints
  - Model inference service
  - Input validation and error handling

- 🌐 **User Interface**
  - Interactive Streamlit dashboard
  - Real-time predictions
  - Data visualization
  - User-friendly input forms

- 🐳 **Containerization**
  - Dockerized application
  - Multi-container setup with Docker Compose
  - Isolated environments
  - Easy deployment

- 🧪 **Testing**
  - Unit tests for core functionality
  - Preprocessing pipeline tests
  - Utility function tests
  - Test coverage reporting

---

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/HARIHARANS24/bigmart-sales-prediction.git
cd bigmart-sales-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run with Docker Compose**
```bash
docker compose up --build
```

4. **Access the applications**
- API: [http://localhost:5000](http://localhost:5000)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)

### Alternative: Running Locally

If you prefer to run the applications locally without Docker:

1. **Activate virtual environment (Windows PowerShell or CMD)**
```bash
.\venv\Scripts\Activate.ps1
```

2. **Run the CLI app for training**
```bash
python -m app.cli train
```

3. **Run the Flask API**
```bash
python -m app.api
```

4. **Run the Streamlit UI**
```bash
streamlit run app/streamlit_app.py
```

---

## 🛠 Tech Stack

- **Core**
  - Python 3.10
  - Pandas & NumPy
  - Scikit-learn
  - Seaborn & Matplotlib

- **Web Framework**
  - Flask
  - Streamlit

- **DevOps**
  - Docker
  - Docker Compose

- **Testing**
  - pytest
  - unittest

- **Configuration**
  - YAML
  - Environment variables

---

## 📊 Model Details

The project implements a machine learning pipeline for sales prediction with the following components:

1. **Data Preprocessing**
   - Missing value handling
   - Categorical encoding
   - Feature scaling
   - Feature engineering

2. **Model Training**
   - Regression algorithms
   - Cross-validation
   - Hyperparameter tuning
   - Model evaluation metrics

3. **Model Artifacts**
   - Serialized model
   - Feature columns
   - Label encoders
   - Performance metrics

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👥 Authors

- HARIHARAN S - Initial work

---

## 🙏 Acknowledgments

- BigMart dataset providers
- Open source community
- Contributors and maintainers
