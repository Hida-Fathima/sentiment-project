# AI-Powered Sentiment Analysis for ChatGPT Reviews
This project implements an end-to-end Natural Language Processing (NLP) solution to analyze user reviews of a ChatGPT application. It leverages a custom-built, robust Machine Learning pipeline to classify reviews into Positive, Neutral, or Negative sentiment categories based on the text provided.

The key deliverable is an interactive, professional Streamlit Web Dashboard that provides deep Exploratory Data Analysis (EDA) on the original user ratings and advanced sentiment insights based on the ML model's predictions.
💡 The Core Innovation: Custom ML Pipeline
The most critical component of this project is the Custom Logistic Regression Pipeline, designed for high reliability on a small, domain-specific dataset.

Why a Custom Pipeline?
Traditional models often fail to capture all nuances. Our pipeline, saved via joblib, combines two powerful feature extraction techniques:

TF-IDF Vectorization: Captures the importance of unique keywords and phrases (n-grams) within the dataset. It learns which words are statistically relevant to a specific sentiment class.

VADER Lexicon Score: Uses the pre-trained VADER (Valence Aware Dictionary and sEntiment Reasoner) lexicon to extract a Compound Polarity Score (a numerical measure of overall emotional intensity).
Business Value & Key Use Cases
The Streamlit dashboard is separated into two pages to provide clear, actionable business intelligence:

🧭 EDA on Ratings: Provides insights based on the raw 1-5 star user ratings. This addresses traditional questions about user satisfaction, such as:

What is the distribution of user-given stars?

How does the average rating change over time or across app versions?

🤖 Sentiment Analysis: Provides deeper insights based on the ML Model's predicted sentiment. This addresses strategic questions, such as:

What are the actual keywords driving Negative sentiment (e.g., "slow," "bug," "paywall")?

Do verified/paying users express significantly different text sentiment compared to non-verified users?

Is there a divergence between the numeric star rating and the text sentiment? (e.g., a 4-star review with highly negative text).
Getting Started
Follow these steps to set up and run the project locally.

Prerequisites
You must have Python 3.8+ installed.

1. Clone the Repository

git clone <YOUR_REPO_URL>
cd ai-powered-sentiment-analysis
2. Install Dependencies
The project uses the following libraries. Install them using the requirements.txt file:


pip install -r requirements.txt
3. Run the Preprocessing & Model Training Script
This script loads the raw data, performs cleaning, trains the Custom ML Pipeline on 225 rows (testing on 25), saves the final model, and creates the processed dataset (custom_ml_analysis_dataset.csv) for the dashboard.

python preprocess_and_train_ml.py
This will output:

✅ data/custom_ml_analysis_dataset.csv (The dashboard source data)

✅ models/custom_ml_model.pkl (The saved ML Pipeline)

✅ models/custom_ml_metrics.json (The model's performance report)

4. Launch the Streamlit Dashboard
Start the interactive web application:

streamlit run app.py
The application will automatically open in your web browser, allowing you to explore the EDA, Sentiment Analysis, Model Performance, and Live Analyzer pages.
