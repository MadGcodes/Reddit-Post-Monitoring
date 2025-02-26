from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch import nn
import json
import os
import praw
import pandas as pd
from datetime import datetime, timedelta
import logging
import numpy as np
from visualizations.visualizations import (
    generate_word_cloud, 
    generate_time_series, 
    generate_sentiment_analysis,
    generate_subreddit_distribution,
    image_to_base64
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model configuration
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
with open(os.path.join(MODEL_DIR, 'config.json'), 'r') as f:
    model_config = json.load(f)

# Reddit API Credentials - Should be moved to environment variables
reddit_credentials = {
    'client_id': os.getenv('REDDIT_CLIENT_ID', '1jMVeE7ePBXVJdDS9mgwaA'),
    'client_secret': os.getenv('REDDIT_CLIENT_SECRET', 'X7nvjX3RYbNGtbhNbVA4g_VF-plJjA'),
    'user_agent': 'DisasterMonitor/1.0'
}

class DisasterDataset(Dataset):
    def __init__(self, texts, max_length=128):
        self.texts = texts
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

class DisasterClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class ModelEnsemble:
    def __init__(self, model_dir, config):
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load all model folds
        for i in range(config['num_models']):
            model = DisasterClassifier()
            model.load_state_dict(torch.load(
                os.path.join(model_dir, f'model_fold_{i}.pt'),
                map_location=self.device
            ))
            model.eval()
            model.to(self.device)
            self.models.append(model)
        
        logger.info(f"Loaded {len(self.models)} models successfully")
        
    def predict(self, texts, batch_size=16):
        dataset = DisasterDataset(texts, max_length=model_config['max_seq_length'])
        dataloader = DataLoader(dataset, batch_size=batch_size)
        
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch_preds = []
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get predictions from all models
                for model in self.models:
                    outputs = model(input_ids, attention_mask)
                    batch_preds.append(outputs.cpu().numpy())
                
                # Average predictions from all models
                avg_preds = np.mean(batch_preds, axis=0)
                all_predictions.extend(avg_preds)
        
        return np.array(all_predictions)

# Initialize model ensemble
try:
    model_ensemble = ModelEnsemble(MODEL_DIR, model_config)
    logger.info("Model ensemble initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model ensemble: {str(e)}")
    model_ensemble = None

class RedditDisasterMonitor:
    def __init__(self, reddit_credentials):
        try:
            self.reddit = praw.Reddit(**reddit_credentials)
            logger.info("Reddit Monitor Initialized Successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Reddit Monitor: {str(e)}")
            raise

    def fetch_reddit_posts(self, subreddits, keywords, hours_back=24, post_limit=100):
        all_posts = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                posts = subreddit.new(limit=post_limit * 2)  # Fetch more posts for filtering
                
                for post in posts:
                    post_time = datetime.fromtimestamp(post.created_utc)
                    if post_time < cutoff_time:
                        continue
                        
                    full_text = f"{post.title} {post.selftext}"
                    # First filter by keywords
                    if any(keyword.lower() in full_text.lower() for keyword in keywords):
                        all_posts.append({
                            'subreddit': subreddit_name,
                            'title': post.title,
                            'text': post.selftext,
                            'full_text': full_text,
                            'created_utc': post_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'score': post.score,
                            'num_comments': post.num_comments,
                            'url': post.url
                        })
                        
                logger.info(f"Successfully fetched posts from r/{subreddit_name}")
            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit_name}: {str(e)}")
                continue
        
        if not all_posts:
            return pd.DataFrame()
            
        # Convert to DataFrame
        posts_df = pd.DataFrame(all_posts)
        
        if model_ensemble:
            # Use model ensemble to filter posts
            predictions = model_ensemble.predict(posts_df['full_text'].values)
            posts_df['disaster_score'] = predictions
            
            # Filter posts with high disaster probability (threshold: 0.5)
            posts_df = posts_df[posts_df['disaster_score'] >= 0.5].sort_values(
                'disaster_score', ascending=False
            ).head(post_limit)
            
            # Drop the full_text and disaster_score columns
            posts_df = posts_df.drop(['full_text', 'disaster_score'], axis=1)
        
        return posts_df

try:
    monitor = RedditDisasterMonitor(reddit_credentials=reddit_credentials)
except Exception as e:
    logger.error(f"Failed to initialize RedditDisasterMonitor: {str(e)}")
    monitor = None

@app.route('/api/fetch-and-analyze', methods=['POST'])
def fetch_and_analyze():
    try:
        if not monitor:
            return jsonify({'success': False, 'error': 'Reddit Monitor not initialized'}), 500

        data = request.json
        keywords = data.get('keywords', [])
        hours_back = data.get('hours_back', 24)

        if not keywords:
            return jsonify({'success': False, 'error': 'No keywords provided'}), 400

        subreddits = ['news', 'worldnews', 'weather', 'environment', 'climate', 'naturaldisaster']
        posts_df = monitor.fetch_reddit_posts(subreddits, keywords, hours_back=hours_back)
        
        if posts_df.empty:
            return jsonify({'success': False, 'error': 'No posts found'}), 404

        # Generate visualizations
        all_text = ' '.join(posts_df['title'] + ' ' + posts_df['text'])
        
        # Generate and save visualizations
        word_cloud_file = 'word_cloud.png'
        time_series_file = 'time_series.png'
        sentiment_file = 'sentiment_analysis.png'
        subreddit_file = 'subreddit_distribution.png'
        
        generate_word_cloud(all_text, word_cloud_file)
        generate_time_series(posts_df, time_series_file)
        generate_sentiment_analysis(posts_df, sentiment_file)
        generate_subreddit_distribution(posts_df, subreddit_file)

        # Convert visualizations to base64
        visualizations = {
            'word_cloud': image_to_base64(word_cloud_file),
            'time_series': image_to_base64(time_series_file),
            'sentiment_analysis': image_to_base64(sentiment_file),
            'subreddit_distribution': image_to_base64(subreddit_file)
        }

        return jsonify({
            'success': True,
            'posts': posts_df.to_dict(orient='records'),
            'visualizations': visualizations
        })

    except Exception as e:
        logger.error(f"Error in fetch_and_analyze: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
