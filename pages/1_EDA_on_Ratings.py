import streamlit as st
import pandas as pd
from src.app_utils import (
    load_analysis_data, get_filtered_data, convert_df_to_csv,
)
from src.plotting_functions import (
    plot_rating_distribution, plot_helpful_reviews, plot_rating_wordclouds,
    plot_avg_rating_over_time, plot_ratings_by_location, plot_ratings_by_platform,
    plot_ratings_by_verified, plot_review_length_by_rating,
    plot_top_words_in_1_star, plot_version_vs_rating
)
import os

# --- Page Config ---
st.set_page_config(page_title="EDA on Ratings", layout="wide")

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

# Get max helpful votes in current data (for context)
max_helpful_votes_in_data = int(df['helpful_votes'].max())

# --- Sidebar Filters ---
st.sidebar.header("📊 EDA Filters")
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

# --- Updated Slider Label ---
helpful_threshold = st.sidebar.slider(
    "Minimum Helpful Votes Required (Max in Data: {})".format(max_helpful_votes_in_data), 
    min_value=0, 
    max_value=200, 
    value=10, 
    step=1
)
# --------------------

include_missing_dates = st.sidebar.checkbox("Include reviews with missing dates", value=True)

# --- Filter Data ---
df_filtered = get_filtered_data(
    df, date_range, selected_platforms, selected_verified, include_missing_dates, selected_ratings
)

if df_filtered.empty:
    st.warning("No data matches the current filters. Please adjust your selection.")
    st.stop()

# --- Main Page ---
st.title("🧭 Exploratory Data Analysis (EDA on Ratings)")
st.markdown("This page visualizes the **raw dataset** using the **1-5 star ratings**, `helpful_votes`, and other features.")
st.markdown("---")

# --- Download Button ---
st.header("Download AI-Processed Data")
st.markdown("This is the fully cleaned CSV file, including the Custom ML-generated sentiment labels and confidence scores.")
csv_data = convert_df_to_csv(df_filtered)
st.download_button(
    label="📥 Download Filtered Data (CSV)",
    data=csv_data,
    file_name="filtered_custom_ml_analysis.csv",
    mime="text/csv",
)
st.markdown("---")

# --- Visualization Selector ---
st.header("Exploratory Visualizations (on Raw Data)")
viz_option = st.selectbox(
    "Select a Visualization to Display:",
    options=[
        "1. Distribution of User Star Ratings (1-5)",
        "2. Helpful Reviews (by Vote Count)",
        "3. Keywords in 4-5 Star vs 1-2 Star Reviews",
        "4. Average User Star Rating Over Time",
        "5. Average User Star Rating by Location",
        "6. Average User Star Rating by Platform",
        "7. Average User Star Rating by Verified Status",
        "8. Review Length by User Star Rating",
        "9. Top Words in 1-Star Reviews",
        "10. Average User Star Rating by App Version"
    ]
)

# --- Display Selected Visualization ---
if viz_option == "1. Distribution of User Star Ratings (1-5)":
    fig, insight = plot_rating_distribution(df_filtered)
elif viz_option == "2. Helpful Reviews (by Vote Count)":
    # Pass the slider value to the plotting function
    fig, insight = plot_helpful_reviews(df_filtered, helpful_threshold)
elif viz_option == "3. Keywords in 4-5 Star vs 1-2 Star Reviews":
    fig, insight = plot_rating_wordclouds(df_filtered)
elif viz_option == "4. Average User Star Rating Over Time":
    freq = st.radio("Select Time Frequency:", ('M', 'W'), format_func=lambda x: 'Monthly' if x == 'M' else 'Weekly', horizontal=True)
    fig, insight = plot_avg_rating_over_time(df_filtered, freq=freq)
elif viz_option == "5. Average User Star Rating by Location":
    fig, insight = plot_ratings_by_location(df_filtered, top_n=15)
elif viz_option == "6. Average User Star Rating by Platform":
    fig, insight = plot_ratings_by_platform(df_filtered)
elif viz_option == "7. Average User Star Rating by Verified Status":
    fig, insight = plot_ratings_by_verified(df_filtered)
elif viz_option == "8. Review Length by User Star Rating":
    fig, insight = plot_review_length_by_rating(df_filtered)
elif viz_option == "9. Top Words in 1-Star Reviews":
    fig, insight = plot_top_words_in_1_star(df_filtered)
elif viz_option == "10. Average User Star Rating by App Version":
    fig, insight = plot_version_vs_rating(df_filtered)

st.plotly_chart(fig, use_container_width=True)
st.markdown(f"**Insight:** {insight}")