from flask import Blueprint, request, jsonify
from sentiment_analyzer import TripAdvisorSentimentAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create blueprint
sentiment_bp = Blueprint('sentiment', __name__)

# Initialize sentiment analyzer
analyzer = TripAdvisorSentimentAnalyzer()

# Load the trained model
try:
    analyzer.load_model()
    logger.info("Sentiment analysis model loaded successfully")
except Exception as e:
    logger.error(f"Error loading sentiment analysis model: {str(e)}")

@sentiment_bp.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of a given review text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing review text'
            }), 400
        
        review_text = data['text']
        
        # Get sentiment prediction and confidence
        sentiment, confidence = analyzer.predict_sentiment(review_text)
        
        return jsonify({
            'text': review_text,
            'sentiment': sentiment,
            'confidence': float(confidence),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return jsonify({
            'error': 'Error analyzing sentiment',
            'details': str(e)
        }), 500

@sentiment_bp.route('/batch-analyze', methods=['POST'])
def batch_analyze_sentiment():
    """Analyze sentiment for multiple reviews"""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data or not isinstance(data['texts'], list):
            return jsonify({
                'error': 'Missing or invalid review texts'
            }), 400
        
        results = []
        for text in data['texts']:
            sentiment, confidence = analyzer.predict_sentiment(text)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': float(confidence)
            })
        
        return jsonify({
            'results': results,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error analyzing sentiments: {str(e)}")
        return jsonify({
            'error': 'Error analyzing sentiments',
            'details': str(e)
        }), 500
