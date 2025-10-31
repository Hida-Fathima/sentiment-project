import streamlit as st
import pandas as pd
from src.app_utils import (
    load_analysis_data, get_filtered_data,
)
from src.plotting_functions import (
    plot_sentiment_distribution, plot_sentiment_vs_rating, plot_sentiment_wordclouds,
    plot_sentiment_over_time, plot_sentiment_by_verified, plot_sentiment_by_length,
    plot_sentiment_by_location, plot_sentiment_by_platform,
    plot_sentiment_by_version, plot_top_words_in_negative
)
import os

# --- Page Config ---
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

# --- Load CSS ---
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError: pass
css_file = os.path.join("assets", "style.css")
load_css(css_file)

# --- Data Loading ---
df = load_analysis_data()
if df.empty: st.stop()

# --- Sidebar Filters ---
st.sidebar.header("📊 Sentiment Filters")
min_date = df.loc[df['date'].notna(), 'date'].min().date()
max_date = df.loc[df['date'].notna(), 'date'].max().date()
date_range = st.sidebar.date_input(
    "Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
platform_options = df['platform'].unique().tolist()
selected_platforms = st.sidebar.multiselect(
    "Select Platform(s)", options=platform_options, default=platform_options
)
verified_options = df['verified_purchase'].unique().tolist()
selected_verified = st.sidebar.multiselect(
    "Select Verified Status", options=verified_options, default=verified_options
)
rating_options = sorted(df['rating'].unique().tolist())
selected_ratings = st.sidebar.multiselect(
    "Select Star Rating(s)", options=rating_options, default=rating_options
)
include_missing_dates = st.sidebar.checkbox("Include reviews with missing dates", value=True)

# --- Filter Data ---
df_filtered = get_filtered_data(
    df, date_range, selected_platforms, selected_verified, include_missing_dates, selected_ratings
)

if df_filtered.empty:
    st.warning("No data matches the current filters. Please adjust your selection.")
    st.stop()

# --- Main Page ---
st.title("🤖 Sentiment Analysis Results (Custom ML-Powered)")
st.markdown("This page answers your **Key Questions** using the sentiment **predicted by your custom TF-IDF + VADER model**.")
st.markdown("---")

# --- Top-Level Metrics ---
st.header("Key Metrics (Based on Custom ML's Predictions)")
col1, col2, col3, col4 = st.columns(4) 
col1.metric("Total Reviews Analyzed", f"{len(df_filtered):,}")
sentiment_counts = df_filtered['custom_ml_prediction'].value_counts()
col2.metric("Positive Reviews (by Model)", f"{sentiment_counts.get('Positive', 0):,}")
col3.metric("Negative Reviews (by Model)", f"{sentiment_counts.get('Negative', 0):,}")
col4.metric("Neutral Reviews (by Model)", f"{sentiment_counts.get('Neutral', 0):,}")
st.markdown("---")


# --- Visualization Selector ---
st.header("Sentiment Analysis Visualizations")
viz_option = st.selectbox(
    "Select a Key Question to Visualize:",
    options=[
        "1. What is the overall sentiment of user reviews?",
        "2. How does sentiment vary by star rating?",
        "3. Which keywords are in each sentiment class?",
        "4. How has sentiment changed over time?",
        "5. Do verified users leave more positive/negative reviews?",
        "6. Are longer reviews more likely to be negative/positive?",
        "7. Which locations show the most positive/negative sentiment?",
        "8. Is there a difference in sentiment across platforms?",
        "9. Which versions are associated with higher/lower sentiment?",
        "10. What are the most common negative feedback themes?"
    ]
)

# --- Display Selected Visualization ---
if viz_option == "1. What is the overall sentiment of user reviews?":
    fig, insight = plot_sentiment_distribution(df_filtered)
elif viz_option == "2. How does sentiment vary by star rating?":
    fig, insight = plot_sentiment_vs_rating(df_filtered)
elif viz_option == "3. Which keywords are in each sentiment class?":
    fig, insight = plot_sentiment_wordclouds(df_filtered)
elif viz_option == "4. How has sentiment changed over time?":
    freq = st.radio("Select Time Frequency:", ('M', 'W'), format_func=lambda x: 'Monthly' if x == 'M' else 'Weekly', horizontal=True)
    fig, insight = plot_sentiment_over_time(df_filtered, freq=freq)
elif viz_option == "5. Do verified users leave more positive/negative reviews?":
    fig, insight = plot_sentiment_by_verified(df_filtered)
elif viz_option == "6. Are longer reviews more likely to be negative/positive?":
    fig, insight = plot_sentiment_by_length(df_filtered)
elif viz_option == "7. Which locations show the most positive/negative sentiment?":
    fig, insight = plot_sentiment_by_location(df_filtered, top_n=15)
elif viz_option == "8. Is there a difference in sentiment across platforms?":
    fig, insight = plot_sentiment_by_platform(df_filtered)
elif viz_option == "9. Which versions are associated with higher/lower sentiment?":
    fig, insight = plot_sentiment_by_version(df_filtered)
elif viz_option == "10. What are the most common negative feedback themes?":
    fig, insight = plot_top_words_in_negative(df_filtered)

st.plotly_chart(fig, use_container_width=True)
st.markdown(f"**Insight:** {insight}")
