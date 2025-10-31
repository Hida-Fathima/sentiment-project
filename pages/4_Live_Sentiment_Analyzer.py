import streamlit as st
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# --- Page Config ---
st.set_page_config(page_title="Live Analyzer", layout="wide")

# --- Load CSS ---
def load_css(file_name):
    try:
        with open(file_name, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError: pass
css_file = os.path.join("assets", "style.css")
load_css(css_file)

# --- VADER LOGIC ---

@st.cache_resource
def load_vader_analyzer():
    """Loads and caches the VADER analyzer."""
    return SentimentIntensityAnalyzer()

def get_vader_prediction(text_input, analyzer):
    """Performs VADER prediction using generic naming."""
    if not isinstance(text_input, str) or len(text_input.strip()) < 5: 
        return "N/A", 0.0, "Please enter more text."
    
    scores = analyzer.polarity_scores(text_input)
    compound = scores['compound']
    
    # VADER Logic for Prediction
    if compound >= 0.05: prediction = "Positive"
    elif compound <= -0.05: prediction = "Negative"
    else: prediction = "Neutral"
    
    # Use compound score as confidence
    confidence = abs(compound)
    if prediction == "Neutral": confidence = 0.5 
    
    explanation = f"Prediction based on AI Model's Lexicon Score. Score: {compound:.4f}"
    return prediction, confidence, explanation

# --- Title ---
st.title("💬 Live Sentiment Analyzer")
st.markdown("Test the **AI Model** in real-time. Enter a review below to get an instant prediction and confidence score.")
st.markdown("---")

# --- Load Model ---
with st.spinner("Loading AI Model (Lexicon Analyzer)..."):
    # Load the VADER analyzer instance directly
    analyzer = load_vader_analyzer() 

if analyzer is None:
    st.error("AI Model could not be loaded. Please check system files.")
    st.stop()

# --- Prediction Interface ---
text_input = st.text_area("Enter your review text here:", height=150, placeholder="This app is...")

if st.button("Analyze Sentiment"):
    if not text_input or len(text_input.strip()) < 5:
        st.warning("Please enter some text to analyze (at least 5 characters).")
    else:
        # Use the locally defined VADER function
        prediction, confidence, explanation = get_vader_prediction(text_input, analyzer)
        
        st.markdown("---")
        st.header("Analysis Result")
        
        if prediction == "Positive": st.success(f"**Sentiment: {prediction}**")
        elif prediction == "Negative": st.error(f"**Sentiment: {prediction}**")
        elif prediction == "Neutral": st.warning(f"**Sentiment: {prediction}**")
        else: st.info(f"**Sentiment: {prediction}**")
            
        # --- FIXED CONFIDENCE DISPLAY ---
        st.subheader("Model Confidence")
        progress_val = abs(confidence)
        st.progress(progress_val)
        st.markdown(f"Confidence Score: **{abs(confidence):.4f}**")
        
        st.subheader("Explanation")
        st.info(explanation)
