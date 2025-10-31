import streamlit as st
import pandas as pd
import json
import os
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import base64
import io
from src.custom_ml_components import get_raw_text, VaderFeatureExtractor

# --- 1. DATA & METRICS LOADING ---

@st.cache_data
def load_analysis_data():
    """Loads the pre-processed CSV file with custom ML sentiment."""
    DATA_FILE = os.path.join("data", "custom_ml_analysis_dataset.csv")
    try:
        df = pd.read_csv(DATA_FILE)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except FileNotFoundError:
        st.error(f"Error: `custom_ml_analysis_dataset.csv` not found.")
        st.info("Please run the `preprocess_and_train_ml.py` script from your terminal first.")
        return pd.DataFrame()

@st.cache_data
def load_metrics():
    """Loads the saved custom_ml_metrics.json file."""
    METRICS_FILE = os.path.join("models", "custom_ml_metrics.json")
    try:
        with open(METRICS_FILE, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        st.error(f"Error: `custom_ml_metrics.json` not found.")
        st.info("Please run the `preprocess_and_train_ml.py` script from your terminal first.")
        return None

# --- 2. LIVE MODEL LOADING ---

@st.cache_resource
def load_custom_ml_model():
    """Loads the custom ML pipeline for live prediction."""
    MODEL_PATH = os.path.join("models", "custom_ml_model.pkl")
    try:
        # joblib automatically resolves the components via the imports above
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"Error: Custom ML Model not found at {MODEL_PATH}")
        st.info("Please run the `preprocess_and_train_ml.py` script first.")
        return None
    except Exception as e:
        # Final safety catch for Pickling/AttributeErrors
        st.error(f"Error loading Custom ML Model: {e}")
        st.info("Model loading failed due to component definition mismatch. Please ensure the `preprocess_and_train_ml.py` script was run and the custom components module is correct.")
        return None

def get_custom_ml_prediction(text_input, model):
    """Gets a live prediction from the custom ML model."""
    if model is None: return "Error", 0.0, "Custom ML model not loaded."
    if not isinstance(text_input, str) or len(text_input.strip()) < 5: return "N/A", 0.0, "Please enter more text."
    
    # Predict and get probability
    prediction = model.predict(pd.Series([text_input]))[0]
    # FIX: Convert numpy float to standard Python float for JSON safety
    confidence = float(max(model.predict_proba(pd.Series([text_input]))[0]))
    
    explanation = "Prediction based on Custom ML model (TF-IDF + VADER score input)."
    return prediction, confidence, explanation

# --- 3. FILTERING & DOWNLOAD ---

def get_filtered_data(df, date_range, platforms, verified_status, include_missing_dates, selected_ratings):
    """Filters the DataFrame based on sidebar controls."""
    if df.empty: return pd.DataFrame()
    if len(date_range) != 2: return df
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    date_mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    if include_missing_dates:
        missing_date_mask = df['date'].isna()
        df_filtered = df[date_mask | missing_date_mask]
    else:
        df_filtered = df[date_mask]
    if platforms:
        df_filtered = df_filtered[df_filtered['platform'].isin(platforms)]
    if verified_status:
        df_filtered = df_filtered[df_filtered['verified_purchase'].isin(verified_status)]
    if selected_ratings:
        df_filtered = df_filtered[df['rating'].isin(selected_ratings)]
    return df_filtered

@st.cache_data
def convert_df_to_csv(df):
    """Converts a DataFrame to a CSV string for downloading."""
    return df.to_csv(index=False).encode('utf-8')
