import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import base64
import io

# --- Configuration ---
COLORS = {'Positive': '#10A37F', 'Neutral': '#FFD700', 'Negative': '#F04A4A',
          'Background': '#0E1117', 'Text': '#FAFAFA', 'Grid': '#262730'}

def create_plotly_layout(fig, title):
    """Helper function to apply standard dark theme layout."""
    fig.update_layout(
        title=title, plot_bgcolor=COLORS['Background'], paper_bgcolor=COLORS['Background'],
        font_color=COLORS['Text'],
        xaxis=dict(gridcolor=COLORS['Grid']), yaxis=dict(gridcolor=COLORS['Grid']),
        hovermode="x unified"
    )
    return fig

# --- SET 1: EDA Charts ---

def plot_rating_distribution(df):
    """(📊 1) Bar chart of user star ratings (1-5 stars)."""
    rating_counts = df['rating'].value_counts().sort_index()
    fig = px.bar(rating_counts, x=rating_counts.index, y=rating_counts.values,
                 labels={'x': 'Rating (1-5 Stars)', 'y': 'Number of Reviews'},
                 color=rating_counts.index, color_continuous_scale=px.colors.sequential.Bluyl)
    fig = create_plotly_layout(fig, "Distribution of User Star Ratings (1-5)")
    fig.update_layout(coloraxis_showscale=False)
    insight = "This shows the original 1-5 star ratings given by users."
    return fig, insight

def plot_helpful_reviews(df, threshold=10):
    """(📊 2) Pie chart of helpful vs. not-helpful reviews."""
    df['is_helpful'] = df['helpful_votes'] >= threshold
    helpful_counts = df['is_helpful'].value_counts()
    helpful_counts.index = helpful_counts.index.map({True: f'Helpful (≥{threshold} votes)', False: f'Not Helpful (<{threshold} votes)'})
    fig = px.pie(helpful_counts, values=helpful_counts.values, names=helpful_counts.index,
                 color=helpful_counts.index,
                 color_discrete_map={f'Helpful (≥{threshold} votes)': COLORS['Positive'], f'Not Helpful (<{threshold} votes)': COLORS['Grid']})
    fig = create_plotly_layout(fig, f"Proportion of Reviews with ≥ {threshold} Helpful Votes")
    insight = f"{helpful_counts.get(f'Helpful (≥{threshold} votes)', 0)} reviews are considered helpful by {threshold}+ users."
    return fig, insight

def plot_rating_wordclouds(df):
    """(📊 3) Two word clouds for positive vs. negative *ratings*."""
    pos_text = " ".join(df[df['rating_sentiment'] == 'Positive']['clean_text_eda'])
    neg_text = " ".join(df[df['rating_sentiment'] == 'Negative']['clean_text_eda'])
    if not pos_text: pos_text = "no positive words"
    if not neg_text: neg_text = "no negative words"
    wc_pos = WordCloud(width=400, height=200, background_color=None, mode='RGBA', colormap='Greens').generate(pos_text)
    wc_neg = WordCloud(width=400, height=200, background_color=None, mode='RGBA', colormap='Reds').generate(neg_text)
    def wc_to_base64(wc):
        img_pil = wc.to_image(); img_io = io.BytesIO(); img_pil.save(img_io, 'PNG'); return base64.b64encode(img_io.getvalue()).decode()
    fig = go.Figure()
    fig.add_layout_image(x=0, y=1, sizex=0.5, sizey=1, xanchor="left", yanchor="top", source='data:image/png;base64,{}'.format(wc_to_base64(wc_pos)))
    fig.add_layout_image(x=0.5, y=1, sizex=0.5, sizey=1, xanchor="left", yanchor="top", source='data:image/png;base64,{}'.format(wc_to_base64(wc_neg)))
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]), plot_bgcolor=COLORS['Background'], paper_bgcolor=COLORS['Background'],
        margin=dict(t=50, b=50, l=10, r=10), # Add bottom margin to make space for text
        annotations=[
            # Annotation 1: Positive Ratings
            dict(x=0.25, y=-0.10, text="<b>Keywords in 4-5 Star Reviews</b>", showarrow=False, xref="paper", yref="paper", font=dict(color=COLORS['Positive'], size=14)),
            # Annotation 2: Negative Ratings
            dict(x=0.75, y=-0.10, text="<b>Keywords in 1-2 Star Reviews</b>", showarrow=False, xref="paper", yref="paper", font=dict(color=COLORS['Negative'], size=14))
        ]
    )
    insight = "These are the most frequent words in reviews that users *rated* as Positive (4-5) or Negative (1-2)."
    return fig, insight

def plot_avg_rating_over_time(df, freq='M'):
    """(📊 4) Line chart of average *star rating* over time."""
    df_time = df.dropna(subset=['date'])
    if df_time.empty: return go.Figure(), "No data with valid dates available."
    if df_time['date'].dt.tz is None: df_time['date'] = df_time['date'].dt.tz_localize('UTC')
    df_time = df_time.set_index('date').resample(freq)['rating'].mean().reset_index()
    fig = px.line(df_time, x='date', y='rating', labels={'date': 'Date', 'rating': 'Average Rating (1-5)'}, markers=True)
    fig = create_plotly_layout(fig, f"Average User Star Rating Over Time")
    insight = "This chart tracks the original 1-5 star user ratings over time."
    return fig, insight

def plot_ratings_by_location(df, top_n=15):
    """(📊 5) Bar chart of average *star ratings* by location."""
    location_ratings = df.groupby('location')['rating'].agg(['mean', 'count']).reset_index()
    location_ratings = location_ratings[location_ratings['count'] > 2].nlargest(top_n, 'count').sort_values('mean', ascending=False)
    if location_ratings.empty: return go.Figure(), "Not enough data for any location (min. 3 reviews)."
    fig = px.bar(location_ratings, x='location', y='mean', color='mean',
                 color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[1, 5],
                 labels={'location': 'Location', 'mean': 'Average Star Rating'}, hover_data={'count': True})
    fig = create_plotly_layout(fig, f"Average User Star Rating by Location (Min. 3 Reviews)")
    insight = "This chart highlights regional differences in user star ratings."
    return fig, insight

def plot_ratings_by_platform(df):
    """(📊 6) Grouped bar chart of average *star ratings* by platform."""
    platform_ratings = df.groupby('platform')['rating'].mean().reset_index().sort_values('rating', ascending=False)
    if platform_ratings.empty: return go.Figure(), "No data to show."
    fig = px.bar(platform_ratings, x='platform', y='rating', color='rating',
                 color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[1, 5],
                 labels={'platform': 'Platform', 'rating': 'Average Star Rating'})
    fig = create_plotly_layout(fig, "Average User Star Rating by Platform")
    insight = "This compares the average 1-5 star rating across different platforms."
    return fig, insight

def plot_ratings_by_verified(df):
    """(📊 7) Grouped bar chart of average *star ratings* by verified status."""
    verified_ratings = df.groupby('verified_purchase')['rating'].mean().reset_index().sort_values('rating', ascending=False)
    if verified_ratings.empty: return go.Figure(), "No data to show."
    fig = px.bar(verified_ratings, x='verified_purchase', y='rating', color='rating',
                 color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[1, 5],
                 labels={'verified_purchase': 'Verified Purchase', 'rating': 'Average Star Rating'})
    fig = create_plotly_layout(fig, "Average User Star Rating: Verified vs. Non-Verified")
    insight = "This shows if verified (paying) users give different star ratings than non-verified users."
    return fig, insight

def plot_review_length_by_rating(df):
    """(📊 8) Box plot of review length per *star rating* category."""
    if df.empty: return go.Figure(), "No data to show."
    max_len = df['review_length'].quantile(0.99)
    df_filtered = df[df['review_length'] < max_len]
    fig = px.box(df_filtered, x='rating', y='review_length', color='rating',
                 labels={'rating': 'Rating (1-5 Stars)', 'review_length': 'Length of Raw Review (Chars)'},
                 color_discrete_map={r:c for r, c in zip(range(1, 6), px.colors.sequential.Bluyl)})
    fig = create_plotly_layout(fig, "Review Length by User Star Rating")
    insight = "This box plot reveals patterns in review length based on the original 1-5 star rating."
    return fig, insight

def plot_top_words_in_1_star(df, top_n=20):
    """(📊 9) Bar chart of top terms in *1-star reviews*."""
    one_star_text = " ".join(df[df['rating'] == 1]['clean_text_eda'])
    if not one_star_text: return go.Figure(), "No 1-star reviews with text found."
    word_counts = Counter(one_star_text.split()).most_common(top_n)
    df_words = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
    fig = px.bar(df_words.sort_values('Frequency', ascending=True), x='Frequency', y='Word', # Set ascending=True to ensure correct bar order
                 orientation='h', color='Frequency', color_continuous_scale=px.colors.sequential.Reds)
    fig = create_plotly_layout(fig, "Top 20 Words in 1-Star Reviews")
    insight = "This chart identifies key pain points from reviews that users *rated* as 1-star."
    return fig, insight

def plot_version_vs_rating(df):
    """(📊 10) Bar chart of app version vs. average *star rating*."""
    version_ratings = df.groupby('version')['rating'].agg(['mean', 'count']).reset_index()
    version_ratings = version_ratings[version_ratings['count'] > 2].sort_values('mean', ascending=False)
    if version_ratings.empty: return go.Figure(), "Not enough data for any app version (min. 3 reviews)."
    
  
    fig = px.bar(version_ratings, x='version', y='mean', color='mean',
                 color_continuous_scale=px.colors.diverging.RdYlGn, range_color=[1, 5],
                 labels={'version': 'App Version', 'mean': 'Average Star Rating'}, hover_data={'count': True})
   
    
    fig = create_plotly_layout(fig, "Average User Star Rating by App Version (Min. 3 Reviews)")
    insight = "This chart evaluates if new app versions are receiving better *star ratings*."
    return fig, insight

# --- SET 2: Sentiment Analysis Charts (Based on Custom ML Predictions) ---

def plot_sentiment_distribution(df):
    """(Key Q 1) Pie chart of *Custom ML's* sentiment predictions."""
    sentiment_counts = df['custom_ml_prediction'].value_counts()
    fig = px.pie(sentiment_counts, values=sentiment_counts.values, names=sentiment_counts.index,
                 color=sentiment_counts.index, color_discrete_map=COLORS)
    fig = create_plotly_layout(fig, f"Custom ML-Predicted Sentiment Distribution")
    insight = "This shows the overall sentiment *as predicted by the custom ML model* from the review text."
    return fig, insight

def plot_sentiment_vs_rating(df):
    """(Key Q 2) How does *Custom ML sentiment* vary by *star rating*?"""
    sentiment_vs_rating = df.groupby('rating')['custom_ml_prediction'].value_counts(normalize=True).unstack(fill_value=0)
    sentiment_vs_rating = sentiment_vs_rating.mul(100).reset_index()
    sentiment_vs_rating = sentiment_vs_rating.melt('rating', var_name='Sentiment', value_name='Percentage')
    fig = px.bar(sentiment_vs_rating, x='rating', y='Percentage', color='Sentiment',
                 labels={'rating': 'User Star Rating', 'Percentage': '% of Reviews'},
                 color_discrete_map=COLORS, barmode='stack')
    fig = create_plotly_layout(fig, f"Custom ML Sentiment vs. User Star Rating")
    insight = "This chart shows how the custom ML model's prediction aligns with the original 1-5 star rating."
    return fig, insight

def plot_sentiment_wordclouds(df):
    """(Key Q 3) Two word clouds for positive vs. negative *Custom ML sentiment*."""
    pos_text = " ".join(df[df['custom_ml_prediction'] == 'Positive']['clean_text_eda'])
    neg_text = " ".join(df[df['custom_ml_prediction'] == 'Negative']['clean_text_eda'])
    if not pos_text: pos_text = "no positive words"
    if not neg_text: neg_text = "no negative words"
    wc_pos = WordCloud(width=400, height=200, background_color=None, mode='RGBA', colormap='Greens').generate(pos_text)
    wc_neg = WordCloud(width=400, height=200, background_color=None, mode='RGBA', colormap='Reds').generate(neg_text)
    def wc_to_base64(wc):
        img_pil = wc.to_image(); img_io = io.BytesIO(); img_pil.save(img_io, 'PNG'); return base64.b64encode(img_io.getvalue()).decode()
    fig = go.Figure()
    fig.add_layout_image(x=0, y=1, sizex=0.5, sizey=1, xanchor="left", yanchor="top", source='data:image/png;base64,{}'.format(wc_to_base64(wc_pos)))
    fig.add_layout_image(x=0.5, y=1, sizex=0.5, sizey=1, xanchor="left", yanchor="top", source='data:image/png;base64,{}'.format(wc_to_base64(wc_neg)))
    fig.update_layout(
        xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False, range=[0, 1]), plot_bgcolor=COLORS['Background'], paper_bgcolor=COLORS['Background'],
        margin=dict(t=50, b=50, l=10, r=10), # Add bottom margin to make space for text
        annotations=[
            # Annotation 1: Positive Sentiment
            dict(x=0.25, y=-0.10, text="<b>Keywords in Positive Sentiment</b>", showarrow=False, xref="paper", yref="paper", font=dict(color=COLORS['Positive'], size=14)),
            # Annotation 2: Negative Sentiment
            dict(x=0.75, y=-0.10, text="<b>Keywords in Negative Sentiment</b>", showarrow=False, xref="paper", yref="paper", font=dict(color=COLORS['Negative'], size=14))
        ]
    )
    insight = f"These are the most frequent words in reviews that the *Custom ML model* classified as Positive or Negative."
    return fig, insight

def plot_sentiment_over_time(df, freq='M'):
    """(Key Q 4) How has *Custom ML sentiment* changed over time?"""
    df_time = df.dropna(subset=['date'])
    if df_time.empty: return go.Figure(), "No data with valid dates available."
    if df_time['date'].dt.tz is None: df_time['date'] = df_time['date'].dt.tz_localize('UTC')
    df_time_sentiment = df_time.set_index('date').resample(freq)['custom_ml_prediction'].value_counts(normalize=True).unstack(fill_value=0)
    df_time_sentiment = df_time_sentiment.mul(100).reset_index()
    df_time_sentiment = df_time_sentiment.melt('date', var_name='Sentiment', value_name='Percentage')
    fig = px.line(df_time_sentiment, x='date', y='Percentage', color='Sentiment',
                  labels={'date': 'Date', 'Percentage': '% of Reviews'},
                  color_discrete_map=COLORS)
    fig = create_plotly_layout(fig, f"Sentiment Trend Over Time (Custom ML)")
    insight = "This tracks the percentage of Positive, Neutral, and Negative reviews (by Custom ML) over time."
    return fig, insight

def plot_sentiment_by_verified(df):
    """(Key Q 5) *Custom ML sentiment* distribution for verified vs. non-verified."""
    verified_sentiment = df.groupby('verified_purchase')['custom_ml_prediction'].value_counts(normalize=True).unstack(fill_value=0)
    verified_sentiment = verified_sentiment.mul(100).reset_index()
    verified_sentiment = verified_sentiment.melt('verified_purchase', var_name='Sentiment', value_name='Percentage')
    fig = px.bar(verified_sentiment, x='verified_purchase', y='Percentage', color='Sentiment',
                 labels={'verified_purchase': 'Verified Purchase', 'Percentage': '% of Reviews'},
                 color_discrete_map=COLORS, barmode='group')
    fig = create_plotly_layout(fig, f"Sentiment by Verified Status (Custom ML)")
    insight = "This compares the sentiment distribution (by Custom ML) for verified vs. non-verified users."
    return fig, insight

def plot_sentiment_by_length(df):
    """(Key Q 6) Are longer reviews more *sentiment-positive/negative*?"""
    try:
        df['review_length_group'] = pd.qcut(df['review_length'], q=5, duplicates='drop')
        df['review_length_group'] = df['review_length_group'].astype(str)
    except ValueError:
        return go.Figure(), "Not enough review length diversity to create groups."
        
    length_sentiment = df.groupby('review_length_group')['custom_ml_prediction'].value_counts(normalize=True).unstack(fill_value=0)
    length_sentiment = length_sentiment.mul(100).reset_index()
    length_sentiment = length_sentiment.melt('review_length_group', var_name='Sentiment', value_name='Percentage')
    
    fig = px.bar(length_sentiment, x='review_length_group', y='Percentage', color='Sentiment',
                 labels={'review_length_group': 'Review Length (Quantile Group)', 'Percentage': '% of Reviews'},
                 color_discrete_map=COLORS)
    fig = create_plotly_layout(fig, f"Sentiment vs. Review Length (Custom ML)")
    insight = "This shows if longer reviews (grouped by length) tend to be more positive or negative."
    return fig, insight

def plot_sentiment_by_location(df, top_n=15):
    """(Key Q 7) Which locations show the most positive/negative *Custom ML sentiment*?"""
    location_sentiment = df.groupby('location')['custom_ml_prediction'].value_counts(normalize=True).unstack(fill_value=0)
    location_counts = df['location'].value_counts()
    location_sentiment = location_sentiment[location_counts > 2]
    if location_sentiment.empty: return go.Figure(), "Not enough data for any location (min. 3 reviews)."
    top_locations = location_counts[location_counts > 2].nlargest(top_n).index
    location_sentiment = location_sentiment.loc[top_locations]
    location_sentiment = location_sentiment.sort_values(by='Positive', ascending=False)
    fig = px.bar(location_sentiment, x=location_sentiment.index, y=['Positive', 'Neutral', 'Negative'],
                 color_discrete_map=COLORS, barmode='stack',
                 labels={'value': 'Percentage of Reviews', 'index': 'Location'})
    fig = create_plotly_layout(fig, f"Sentiment by Location (Top {top_n} by Volume, Min. 3 Reviews)")
    insight = "This shows the distribution of *Custom ML-predicted sentiment* for different locations."
    return fig, insight

def plot_sentiment_by_platform(df):
    """(Key Q 8) Is there a difference in *Custom ML sentiment* across platforms?"""
    platform_sentiment = df.groupby('platform')['custom_ml_prediction'].value_counts(normalize=True).unstack(fill_value=0)
    platform_sentiment = platform_sentiment.mul(100).reset_index()
    platform_sentiment = platform_sentiment.melt('platform', var_name='Sentiment', value_name='Percentage')
    fig = px.bar(platform_sentiment, x='platform', y='Percentage', color='Sentiment',
                 labels={'platform': 'Platform', 'Percentage': '% of Reviews'},
                 color_discrete_map=COLORS, barmode='group')
    fig = create_plotly_layout(fig, f"Sentiment by Platform (Custom ML)")
    insight = "This compares the sentiment distribution (by Custom ML) across platforms."
    return fig, insight

def plot_sentiment_by_version(df):
    """(Key Q 9) Which versions are associated with higher/lower *Custom ML sentiment*?"""
    version_sentiment = df.groupby('version')['custom_ml_prediction'].value_counts(normalize=True).unstack(fill_value=0)
    version_counts = df['version'].value_counts()
    version_sentiment = version_sentiment[version_counts > 2]
    if version_sentiment.empty: return go.Figure(), "Not enough data for any app version (min. 3 reviews)."
    version_sentiment = version_sentiment.sort_values(by='Positive', ascending=False)
    fig = px.bar(version_sentiment, x=version_sentiment.index, y=['Positive', 'Neutral', 'Negative'],
                 color_discrete_map=COLORS, barmode='stack',
                 labels={'value': 'Percentage of Reviews', 'index': 'App Version'})
    fig = create_plotly_layout(fig, f"Sentiment by App Version (Min. 3 Reviews)")
    insight = "This chart shows if newer app versions are receiving more positive *text sentiment* from the Custom ML model."
    return fig, insight

def plot_top_words_in_negative(df, top_n=20):
    """(Key Q 10) What are the most common negative feedback themes?"""
    neg_text = " ".join(df[df['custom_ml_prediction'] == 'Negative']['clean_text_eda'])
    if not neg_text: return go.Figure(), "No reviews classified as Negative by the model."
    word_counts = Counter(neg_text.split()).most_common(top_n)
    df_words = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
    fig = px.bar(df_words.sort_values('Frequency', ascending=True), x='Frequency', y='Word', # Set ascending=True to ensure correct bar order
                 orientation='h', color='Frequency', color_continuous_scale=px.colors.sequential.Reds)
    fig = create_plotly_layout(fig, f"Top 20 Words in Custom ML-Predicted Negative Reviews")
    insight = "This chart identifies key pain points from reviews the *Custom ML model* understood as negative."
    return fig, insight
