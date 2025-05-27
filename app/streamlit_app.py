import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from app.utils import load_data
from app.logger import get_logger

st.title('Big Mart Sales Analysis & Prediction')

# Load data for EDA (cached to speed up)
@st.cache_data
def load_train_data():
    return pd.read_csv('data/train.csv')

df = load_train_data()

# Sidebar for EDA options
st.sidebar.header('Exploratory Data Analysis')
eda_options = ['Sales Distribution', 'Sales by Outlet Type', 'Correlation Heatmap']
choice = st.sidebar.selectbox('Choose EDA Plot:', eda_options)

if choice == 'Sales Distribution':
    st.subheader('Distribution of Item Outlet Sales')
    fig, ax = plt.subplots()
    sns.histplot(df['Item_Outlet_Sales'], kde=True, ax=ax)
    st.pyplot(fig)

elif choice == 'Sales by Outlet Type':
    st.subheader('Average Sales by Outlet Type')
    avg_sales = df.groupby('Outlet_Type')['Item_Outlet_Sales'].mean().sort_values()
    st.bar_chart(avg_sales)

elif choice == 'Correlation Heatmap':
    st.subheader('Correlation Heatmap')
    corr = df.select_dtypes(include='number').corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Prediction section
st.header('Predict Sales for a New Item')

with st.form('prediction_form'):
    item_identifier = st.text_input('Item Identifier (e.g., FDA15)', 'FDA15')
    item_weight = st.number_input('Item Weight (kg)', 10.5)
    item_fat_content = st.selectbox('Item Fat Content', ['Low Fat', 'Regular'])
    item_visibility = st.number_input('Item Visibility', 0.05, 0.5, 0.07, 0.01)
    item_type = st.selectbox('Item Type', df['Item_Type'].unique())
    outlet_identifier = st.text_input('Outlet Identifier (e.g., OUT049)', 'OUT049')
    outlet_establishment_year = st.number_input('Outlet Establishment Year', 1985, 2010, 1999)
    outlet_size = st.selectbox('Outlet Size', ['Small', 'Medium', 'High'])
    outlet_location_type = st.selectbox('Outlet Location Type', ['Tier 1', 'Tier 2', 'Tier 3'])
    outlet_type = st.selectbox('Outlet Type', ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

    submitted = st.form_submit_button('Predict Sales')

if submitted:
    payload = {
        'Item_Identifier': item_identifier,
        'Item_Weight': item_weight,
        'Item_Fat_Content': item_fat_content,
        'Item_Visibility': item_visibility,
        'Item_Type': item_type,
        'Outlet_Identifier': outlet_identifier,
        'Outlet_Establishment_Year': outlet_establishment_year,
        'Outlet_Size': outlet_size,
        'Outlet_Location_Type': outlet_location_type,
        'Outlet_Type': outlet_type
    }
    try:
        # ✅ Use 'api' instead of 'localhost' inside Docker Compose
        response = requests.post('http://api:5000/predict', json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Sales: {result['predicted_sales']:.2f}")
        else:
            st.error('Prediction failed. Try again later.')
    except Exception as e:
        st.error(f"API connection error: {e}")
