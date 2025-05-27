
# BigMart Sales Prediction

🚀 End-to-end machine learning pipeline for BigMart sales prediction using Flask API, Streamlit UI, Docker, and modular Python architecture — includes EDA, model training, real-time prediction, and testing.

---

## 🗂️ Project Structure

```
bigmart-sales-prediction/
├── app/
│   ├── __init__.py
│   ├── cli.py                  # CLI commands
│   ├── api.py                  # Flask API
│   ├── streamlit_app.py        # Streamlit UI
│   ├── model.py                # Model training, evaluation, prediction logic
│   ├── preprocessing.py        # Data preprocessing and feature engineering
│   ├── visualization.py        # EDA plotting functions
│   ├── utils.py                # Helpers (logging, config, data loading)
│   ├── logger.py               # Logger setup
├── config.yaml                 # Config file for parameters
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── processed_train.csv     # After preprocessing (optional)
├── models/
│   ├── model_v1.pkl            # Saved models with versioning
│   ├── model_v1_metrics.json   # Metrics saved alongside model
├── tests/
│   ├── test_preprocessing.py   # Unit tests for preprocessing
│   ├── test_utils.py
├── notebooks/
│   ├── BigMart_EDA.ipynb       # Optional extended EDA notebook
│   ├── Model_Training.ipynb
├── requirements.txt
├── Dockerfile
├── Dockerfile.streamlit
├── docker-compose.yml
├── README.md
├── .env                        # Environment variables (e.g. API keys, secrets)
```

---

## 📦 Features

- 🔍 **EDA**: Visualizations for data distribution and relationships
- 🧠 **Model Training**: Scikit-learn regression pipeline with preprocessing
- 🔁 **API**: Flask REST API to serve predictions
- 🌐 **Streamlit UI**: User-friendly interface for analysis and prediction
- 🐳 **Dockerized**: Fully containerized with Docker Compose
- 🧪 **Testing**: Unit tests for critical modules
- 🔧 **Modular Code**: Separated concerns across reusable components

---

## 🚀 Quick Start

1. **Clone the repo**  
```bash
git clone https://github.com/your-username/bigmart-sales-prediction.git
cd bigmart-sales-prediction
```

2. **Build and run with Docker Compose**  
```bash
docker compose up --build
```

3. **Access the app**  
- API: [http://localhost:5000](http://localhost:5000)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)

---

## 🛠 Tech Stack

- Python 3.10
- Pandas, Scikit-learn, Seaborn, Matplotlib
- Flask, Streamlit
- Docker, Docker Compose
- YAML, Logging
- pytest

---

## 🤝 Contributing

Feel free to fork this project and submit PRs! Feedback is welcome.

---

## 📄 License

MIT License
