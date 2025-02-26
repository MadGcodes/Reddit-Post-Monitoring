# Disaster Monitor

A real-time disaster monitoring system that uses Reddit data and machine learning to track and visualize disaster-related information.

## Features

- Real-time monitoring of disaster-related posts from multiple subreddits
- BERT-based machine learning model ensemble for accurate disaster detection
- Interactive visualizations including:
  - Word clouds of trending terms
  - Time series analysis of post frequency
  - Sentiment analysis of disaster-related discussions
  - Subreddit distribution analysis
- Keyword-based filtering with ML-enhanced relevance scoring

## Project Structure

```
.
├── backend/
│   ├── app.py                 # Flask backend server
│   ├── models/               # Trained BERT model ensemble
│   └── visualizations/       # Visualization generation modules
└── disaster-monitor-frontend/
    └── src/                  # React frontend application
```

## Setup

### Backend

1. Install Python dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export REDDIT_CLIENT_ID='your_client_id'
export REDDIT_CLIENT_SECRET='your_client_secret'
```

3. Run the backend server:
```bash
python app.py
```

### Frontend

1. Install Node.js dependencies:
```bash
cd disaster-monitor-frontend
npm install
```

2. Run the frontend development server:
```bash
npm start
```

## Model Information

The system uses an ensemble of 5 BERT-based models trained on disaster-related text data. Each model was trained using different data folds to ensure robust predictions.

## API Endpoints

- POST `/api/fetch-and-analyze`
  - Fetches and analyzes disaster-related posts
  - Parameters:
    - `keywords`: List of keywords to filter posts
    - `hours_back`: Number of hours to look back (default: 24)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 