import pandas as pd
import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# --- 0. Setup ---
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. Define File Paths ---
RAW_DATA_FILE = os.path.join("data", "chatgpt_style_reviews_dataset.xlsx")
OUTPUT_DATA_FILE = os.path.join("data", "custom_ml_analysis_dataset.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "custom_ml_model.pkl")
METRICS_FILE = os.path.join(MODEL_DIR, "custom_ml_metrics.json")

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# --- 2. Define Helper Functions (Needed for Global Pickling) ---

def label_from_rating(rating):
    rating = int(rating)
    if rating >= 4: return "Positive"
    elif rating == 3: return "Neutral"
    else: return "Negative"

def clean_text_for_ml(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def clean_text_for_eda(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in text.split() if token not in stopwords_set]
    return " ".join(tokens)

# --- 3. ML Pipeline Components (Essential for joblib) ---

# Component 1: TF-IDF Text Extraction
def get_raw_text(X):
    return X

# Component 2: VADER Score Extraction
analyzer_instance = SentimentIntensityAnalyzer()
class VaderFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, analyzer=analyzer_instance):
        self.analyzer = analyzer
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        scores = X.apply(lambda text: self.analyzer.polarity_scores(text)['compound'])
        return scores.values.reshape(-1, 1)

# ------------------------------------------------------------------
# --- MAIN SCRIPT EXECUTION ---
# ------------------------------------------------------------------
print("--- Starting Custom ML Model Training Script ---")

# --- 4. Load & Prepare Data ---
df = pd.read_excel(RAW_DATA_FILE, na_values=['########'])
df = df.dropna(subset=['review', 'rating'])
df['rating_sentiment'] = df['rating'].apply(label_from_rating)
df['clean_review_ml'] = df['review'].apply(clean_text_for_ml)
df['clean_text_eda'] = df['review'].apply(clean_text_for_eda)
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['review_length'] = df['review_length'].fillna(0)
df['helpful_votes'] = df['helpful_votes'].fillna(0) # Ensure helpful votes is clean
df['vader_compound_score'] = df['review'].apply(lambda x: analyzer_instance.polarity_scores(x)['compound'])

# --- 5. Implement Train/Test Split (225/25) ---
X = df['review']
y = df['rating_sentiment']

X_train_data, X_test_data, y_train, y_test = train_test_split(
    X, y, test_size=25, random_state=42, stratify=y
)
print(f"Split data: Training on {len(X_train_data)} rows, Testing on {len(X_test_data)} rows.")

# --- 6. Define and Train ML Pipeline ---
print("Defining and training Custom ML Model (Logistic Regression)...")

tfidf_pipeline = Pipeline([
    ('selector', FunctionTransformer(get_raw_text, validate=False)),
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000))
])

feature_union = FeatureUnion(transformer_list=[
    ('text_features', tfidf_pipeline),
    ('vader_feature', VaderFeatureExtractor(analyzer=analyzer_instance)) 
])

custom_ml_pipeline = Pipeline([
    ('features', feature_union),
    ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
])

custom_ml_pipeline.fit(X_train_data, y_train)
print("Custom ML Model trained successfully.")

# --- 7. Generate Predictions, Metrics, and Save ---
print("Generating predictions and metrics...")

# Predict on the TEST SET (25 rows) for realistic evaluation
y_pred_test = custom_ml_pipeline.predict(X_test_data)

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred_test)
report = classification_report(y_test, y_pred_test, labels=custom_ml_pipeline.classes_, output_dict=True, zero_division=0)
report['accuracy'] = accuracy

# Save Metrics
with open(METRICS_FILE, 'w') as f:
    json.dump({'custom_ml': report}, f, indent=4)
print(f"Metrics saved to {METRICS_FILE}")
print(f"  -> Test Set Accuracy: {accuracy:.2%}")

# Predict on the FULL SET for EDA/Visualization
df['custom_ml_prediction'] = custom_ml_pipeline.predict(df['review'])
df['custom_ml_confidence'] = [float(max(p)) for p in custom_ml_pipeline.predict_proba(df['review'])]

# Save the Pipeline
joblib.dump(custom_ml_pipeline, MODEL_PATH)
print(f"Custom ML Pipeline saved to {MODEL_PATH}")

# Save Final Dataset
df.to_csv(OUTPUT_DATA_FILE, index=False)
print(f"Final analysis dataset saved to {OUTPUT_DATA_FILE}")
print("\n--- ✅ SCRIPT COMPLETE ---")