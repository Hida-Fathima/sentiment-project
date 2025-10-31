import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="AI-Powered Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load local CSS
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError: pass
css_file = os.path.join("assets", "style.css")
load_css(css_file)

# --- Main Page Content ---
st.title("📊 AI-Powered Sentiment Analysis for ChatGPT Reviews")
st.subheader("A Custom ML Project for Advanced EDA and Sentiment Insights")
st.markdown("---")

# --- Project Goals ---
st.header("About The Project")
st.markdown("""
This dashboard demonstrates a full NLP project using a robust custom-built Machine Learning model. 

The model is designed specifically for your small dataset by combining **TF-IDF text features** with the **VADER lexicon score** into a single Logistic Regression pipeline. This structure ensures a reliable sentiment prediction by leveraging both keyword importance (TF-IDF) and established lexicon rules (VADER).

The project correctly separates analysis into two key areas:
1.  **Traditional EDA:** Visualizing the raw dataset based on **user-provided 1-5 star ratings** (Page 1).
2.  **Sentiment Analysis:** Visualizing the data based on the **Custom ML Model's text-based predictions** (Page 2).
""")

with st.expander("Click to see details about the Dataset and Custom Model"):
    st.markdown("""
    ### Dataset
    - **Source:** `chatgpt_style_reviews_dataset.xlsx` (250 rows).
    - **Processed Data:** The app runs on `custom_ml_analysis_dataset.csv`. This file contains the original data *plus* the Custom ML Model's predictions and confidence scores.
    
    ### Model: Custom Logistic Regression Pipeline
    - **Architecture:** `FeatureUnion` (TF-IDF + VADER Lexicon Score) → `Logistic Regression`.
    - **Process:** The model was trained offline on **225 rows** and tested on **25 rows** to ensure metrics are reliable.
    - **Insight:** By including the VADER score as a numerical feature, the model learns not only *which words* are important but *how much* lexicon intensity contributes to the sentiment.
    """)
st.markdown("---")

# --- How to Use This Dashboard ---
st.header("How to Use This Dashboard")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader("🧭 1. EDA on Ratings")
    st.markdown("""
    Explore the **raw data**. This page answers your (📊 1-10) questions using the 1-5 star ratings, helpful votes, etc.
    """)

with col2:
    st.subheader("🤖 2. Sentiment Analysis")
    st.markdown("""
    Explore the **model's predictions**. This page answers your "Key Questions" using the sentiment predicted by the Custom ML Model.
    """)

with col3:
    st.subheader("📈 3. Model Performance")
    st.markdown("""
    See how the Custom ML Model performed on the 25-row test set. This page shows the **Accuracy & F1-Score**.
    """)

with col4:
    st.subheader("💬 4. Live Analyzer")
    st.markdown("""
    Type in a new review to get a live sentiment prediction.
    """)
