from flask import Blueprint, request, jsonify
from .sentiment_analyzer import TripAdvisorSentimentAnalyzer

nlp_bp = Blueprint('nlp', __name__)
sentiment_analyzer = TripAdvisorSentimentAnalyzer()

@nlp_bp.route('/analyze', methods=['POST'])
def analyze_text():
    """Analyze text using NLP models"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
            
        # Analyze sentiment
        sentiment_score = sentiment_analyzer.analyze_sentiment(text)
        
        return jsonify({
            'sentiment': sentiment_score,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
