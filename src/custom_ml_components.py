import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

# --- Essential for pickling/unpickling custom components ---

# 1. Component used for TF-IDF Text Extraction
def get_raw_text(X):
    """Returns the input text Series unmodified. (Used by FeatureUnion)"""
    return X

# 2. Component used for VADER Score Extraction
analyzer_instance = SentimentIntensityAnalyzer()
class VaderFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, analyzer=analyzer_instance):
        self.analyzer = analyzer
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        scores = X.apply(lambda text: self.analyzer.polarity_scores(text)['compound'])
        return scores.values.reshape(-1, 1)
