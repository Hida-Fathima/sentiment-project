import streamlit as st
import pandas as pd
from src.app_utils import load_metrics
import os

# --- Page Config ---
st.set_page_config(page_title="Model Performance", layout="wide")

# --- Load CSS ---
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError: pass
css_file = os.path.join("assets", "style.css")
load_css(css_file)

# --- Main Page ---
st.title("📈 Custom ML Model Performance")
st.markdown("This page shows the performance of the **Custom Logistic Regression Model** (trained with TF-IDF + VADER features).")
# --- Load Metrics ---
metrics = load_metrics()
if metrics is None: st.stop()

# --- Helper Function ---
def display_metrics(model_name, metrics_dict):
    st.subheader(f"{model_name} Performance")
    accuracy = metrics_dict.get('accuracy', 0.0)
    st.metric(label="Overall Agreement (Accuracy)", value=f"{accuracy:.2%}")
    st.markdown("**Classification Report**")
    report_df = pd.DataFrame(metrics_dict).transpose()
    report_df = report_df.round(3)
    st.dataframe(report_df, use_container_width=True)

# --- Display Metrics ---
if 'custom_ml' in metrics:
    display_metrics("Custom ML (TF-IDF + VADER)", metrics['custom_ml'])
else:
    st.error("Custom ML metrics not found in file.")

st.markdown("---")
st.info("""
**How to read this report:**
- **Accuracy:** The percentage of times the model's text-based prediction matched the sentiment from the star rating.
- **Precision:** Of all the times the model predicted "Positive", what percentage was correct?
- **Recall:** Of all the *actual* "Positive" reviews, what percentage did the model find?
- **F1-Score:** The balanced average of Precision and Recall.
""")
