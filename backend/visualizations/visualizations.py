import os
import base64
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pandas as pd
import logging
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure the visualizations folder exists
VISUALIZATIONS_FOLDER = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(VISUALIZATIONS_FOLDER, exist_ok=True)

def generate_word_cloud(text, filename):
    """Generate a word cloud and save it as an image."""
    try:
        plt.switch_backend('Agg')  # Use non-interactive backend
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            max_words=100,
            min_font_size=10,
            max_font_size=50
        ).generate(text)
        
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        
        output_path = os.path.join(VISUALIZATIONS_FOLDER, filename)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        logger.info(f"Successfully generated word cloud: {filename}")
        
    except Exception as e:
        logger.error(f"Error generating word cloud: {str(e)}")
        raise

def generate_time_series(posts_df, filename):
    """Generate a time series chart and save it as an image."""
    try:
        plt.switch_backend('Agg')  # Use non-interactive backend
        plt.figure(figsize=(10, 5))
        
        # Convert to datetime and set as index
        posts_df['created_utc'] = pd.to_datetime(posts_df['created_utc'])
        posts_df.set_index('created_utc', inplace=True)
        
        # Create the time series plot
        sns.set_style("whitegrid")
        posts_per_hour = posts_df.resample('H').size()
        
        plt.plot(posts_per_hour.index, posts_per_hour.values, 
                marker='o', linestyle='-', linewidth=2, markersize=6)
        
        plt.title('Disaster-Related Posts Frequency Over Time', pad=20)
        plt.xlabel('Time (UTC)')
        plt.ylabel('Number of Posts')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        output_path = os.path.join(VISUALIZATIONS_FOLDER, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Successfully generated time series plot: {filename}")
        
    except Exception as e:
        logger.error(f"Error generating time series plot: {str(e)}")
        raise

def image_to_base64(filename):
    """Convert an image to base64."""
    try:
        image_path = os.path.join(VISUALIZATIONS_FOLDER, filename)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        with open(image_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            logger.info(f"Successfully converted image to base64: {filename}")
            return encoded
            
    except Exception as e:
        logger.error(f"Error converting image to base64: {str(e)}")
        raise

def generate_sentiment_analysis(posts_df, filename):
    """Generate a sentiment analysis visualization and save it as an image."""
    try:
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))
        
        # Calculate sentiment for each post
        def get_sentiment(text):
            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0:
                return 'Positive'
            elif analysis.sentiment.polarity < 0:
                return 'Negative'
            return 'Neutral'
        
        posts_df['sentiment'] = posts_df['text'].apply(get_sentiment)
        sentiment_counts = posts_df['sentiment'].value_counts()
        
        # Create pie chart
        colors = ['#4CAF50', '#F44336', '#2196F3']  # Green for positive, Red for negative, Blue for neutral
        plt.pie(sentiment_counts, labels=sentiment_counts.index, colors=colors,
                autopct='%1.1f%%', startangle=90)
        plt.title('Sentiment Distribution in Disaster-Related Posts')
        
        output_path = os.path.join(VISUALIZATIONS_FOLDER, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Successfully generated sentiment analysis plot: {filename}")
        
    except Exception as e:
        logger.error(f"Error generating sentiment analysis plot: {str(e)}")
        raise

def generate_subreddit_distribution(posts_df, filename):
    """Generate a subreddit distribution visualization and save it as an image."""
    try:
        plt.switch_backend('Agg')
        plt.figure(figsize=(12, 6))
        
        # Get top 10 subreddits by post count
        subreddit_counts = posts_df['subreddit'].value_counts().head(10)
        
        # Create horizontal bar chart
        sns.set_style("whitegrid")
        ax = sns.barplot(x=subreddit_counts.values, y=subreddit_counts.index, 
                        palette='viridis')
        
        plt.title('Top 10 Subreddits with Disaster-Related Posts')
        plt.xlabel('Number of Posts')
        plt.ylabel('Subreddit')
        
        # Add value labels on the bars
        for i, v in enumerate(subreddit_counts.values):
            ax.text(v, i, f' {v}', va='center')
        
        output_path = os.path.join(VISUALIZATIONS_FOLDER, filename)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        logger.info(f"Successfully generated subreddit distribution plot: {filename}")
        
    except Exception as e:
        logger.error(f"Error generating subreddit distribution plot: {str(e)}")
        raise