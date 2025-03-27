"""
Sentiment analyzer for travel-related text
"""
import os
import nltk
import logging
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TripAdvisorSentimentAnalyzer:
    def __init__(self):
        """Initialize sentiment analyzer with positive and negative word dictionaries"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load positive and negative word dictionaries
        self.positive_words = {
            'amazing', 'awesome', 'beautiful', 'best', 'comfortable', 'excellent', 'fantastic',
            'friendly', 'great', 'happy', 'helpful', 'impressive', 'lovely', 'nice', 'perfect',
            'pleasant', 'recommend', 'relaxing', 'wonderful', 'worth'
        }
        
        self.negative_words = {
            'awful', 'bad', 'disappointing', 'dirty', 'expensive', 'horrible', 'noisy',
            'poor', 'terrible', 'uncomfortable', 'unfriendly', 'unpleasant', 'worst'
        }
        
        logger.info("Model loaded successfully from word dictionaries")
        
    def preprocess_text(self, text: str) -> list:
        """Preprocess text by tokenizing, removing stop words, and lemmatizing"""
        # Tokenize
        tokens = word_tokenize(text.lower())
        
        # Remove stop words and lemmatize
        tokens = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalnum() and token not in self.stop_words
        ]
        
        return tokens
        
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text and return score between 0 and 1"""
        try:
            tokens = self.preprocess_text(text)
            
            if not tokens:
                return 0.5  # Neutral sentiment for empty text
            
            # Count positive and negative words
            positive_count = sum(1 for token in tokens if token in self.positive_words)
            negative_count = sum(1 for token in tokens if token in self.negative_words)
            
            # Calculate sentiment score
            total_count = positive_count + negative_count
            if total_count == 0:
                return 0.5  # Neutral sentiment if no sentiment words found
            
            sentiment_score = positive_count / (positive_count + negative_count)
            
            # Apply min/max thresholds
            sentiment_score = max(0.1, min(0.9, sentiment_score))
            
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.5  # Return neutral sentiment on error
