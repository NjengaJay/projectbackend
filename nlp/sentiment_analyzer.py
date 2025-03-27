"""
Enhanced sentiment analyzer for TripAdvisor reviews
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, NamedTuple
from textblob import TextBlob
import spacy

logger = logging.getLogger(__name__)

class SentimentScore(NamedTuple):
    """Container for sentiment analysis results"""
    score: float  # Overall sentiment score (-1 to 1)
    confidence: float  # Confidence in the sentiment score (0 to 1)
    aspect_scores: Dict[str, float]  # Aspect-specific sentiment scores

class TripAdvisorSentimentAnalyzer:
    """Analyzes sentiment in travel-related text using an enhanced approach"""
    
    def __init__(self, model_path: str = "models"):
        """Initialize sentiment analyzer."""
        try:
            # Load sentiment words
            self.sentiment_words = self._load_sentiment_words(model_path)
            
            # Set default thresholds
            self.positive_threshold = 0.6
            self.negative_threshold = -0.6
            
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("Could not load spaCy model. Using basic tokenization.")
                self.nlp = None
                
            # Load sentiment lexicons
            self.positive_words = {
                # General positive terms
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'perfect', 'best', 'awesome', 'outstanding', 'superb', 'lovely',
                'beautiful', 'brilliant', 'exceptional', 'marvelous', 'delightful',
                
                # Travel-specific positive terms
                'comfortable', 'clean', 'spacious', 'convenient', 'helpful',
                'friendly', 'welcoming', 'scenic', 'peaceful', 'relaxing',
                'authentic', 'charming', 'cozy', 'luxurious', 'modern',
                
                # Service-related positive terms
                'attentive', 'professional', 'efficient', 'prompt', 'courteous',
                'accommodating', 'knowledgeable', 'responsive', 'reliable'
            }
            
            self.negative_words = {
                # General negative terms
                'bad', 'poor', 'terrible', 'awful', 'horrible', 'disappointing',
                'worst', 'unacceptable', 'mediocre', 'subpar', 'inadequate',
                
                # Travel-specific negative terms
                'dirty', 'uncomfortable', 'noisy', 'crowded', 'expensive',
                'inconvenient', 'broken', 'outdated', 'smelly', 'unsafe',
                'unclean', 'cramped', 'rundown', 'overpriced',
                
                # Service-related negative terms
                'rude', 'unhelpful', 'slow', 'unresponsive', 'unprofessional',
                'incompetent', 'negligent', 'careless', 'unfriendly'
            }
            
            # Aspect categories and their related terms
            self.aspects = {
                'service': {
                    'service', 'staff', 'employee', 'waiter', 'receptionist',
                    'manager', 'host', 'hospitality', 'assistance', 'help'
                },
                'cleanliness': {
                    'clean', 'dirty', 'hygiene', 'sanitary', 'spotless',
                    'pristine', 'mess', 'dust', 'stain', 'maintenance'
                },
                'location': {
                    'location', 'area', 'neighborhood', 'distance', 'central',
                    'accessible', 'nearby', 'convenient', 'close', 'far'
                },
                'comfort': {
                    'comfort', 'comfortable', 'bed', 'quiet', 'noise',
                    'spacious', 'cramped', 'cozy', 'temperature', 'amenities'
                },
                'value': {
                    'value', 'price', 'worth', 'expensive', 'cheap',
                    'reasonable', 'cost', 'affordable', 'overpriced', 'bargain'
                }
            }
            
            # Intensity modifiers and their weights
            self.intensity_modifiers = {
                'very': 1.5,
                'really': 1.5,
                'extremely': 2.0,
                'incredibly': 2.0,
                'super': 1.5,
                'quite': 1.2,
                'somewhat': 0.8,
                'slightly': 0.6,
                'not': -1.0,
                "n't": -1.0,
                'never': -1.0
            }
            
            logger.info("Successfully initialized sentiment analyzer")
            
        except Exception as e:
            logger.error(f"Error initializing sentiment analyzer: {str(e)}", exc_info=True)
            raise
            
    def _load_sentiment_words(self, model_path: str) -> Dict[str, float]:
        """Load sentiment words from file or use defaults."""
        try:
            # Try to load from file
            sentiment_path = os.path.join(model_path, "sentiment_words.json")
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'r') as f:
                    return json.load(f)
            
            # Use default sentiment words
            return {
                # Positive words
                "amazing": 1.0,
                "awesome": 1.0,
                "excellent": 1.0,
                "fantastic": 1.0,
                "great": 0.8,
                "good": 0.6,
                "nice": 0.6,
                "wonderful": 0.8,
                "beautiful": 0.8,
                "enjoyed": 0.7,
                "recommend": 0.7,
                "worth": 0.6,
                "perfect": 0.9,
                "love": 0.9,
                "best": 0.9,
                
                # Negative words
                "terrible": -1.0,
                "awful": -1.0,
                "horrible": -1.0,
                "bad": -0.6,
                "poor": -0.6,
                "disappointing": -0.7,
                "waste": -0.8,
                "avoid": -0.8,
                "dirty": -0.6,
                "expensive": -0.5,
                "crowded": -0.4,
                "overrated": -0.7,
                "boring": -0.6,
                "hate": -0.9,
                "worst": -0.9
            }
            
        except Exception as e:
            logger.error(f"Error loading sentiment words: {str(e)}", exc_info=True)
            raise
            
    def save_model(self, model_path: str):
        """
        Save model data to disk
        
        Args:
            model_path: Directory to save model files
        """
        try:
            # Create model directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            
            # Save sentiment words and aspects
            sentiment_words_path = os.path.join(model_path, "sentiment_words.json")
            with open(sentiment_words_path, 'w') as f:
                json.dump({
                    "positive_words": list(self.positive_words),
                    "negative_words": list(self.negative_words),
                    "aspects": {k: list(v) for k, v in self.aspects.items()},
                    "intensity_modifiers": self.intensity_modifiers
                }, f)
                
            logger.info(f"Successfully saved model to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}", exc_info=True)
            raise
            
    def load_model(self, model_path: str):
        """
        Load model data from disk
        
        Args:
            model_path: Directory containing model files
        """
        try:
            # Load sentiment words and aspects
            sentiment_words_path = os.path.join(model_path, "sentiment_words.json")
            with open(sentiment_words_path, 'r') as f:
                data = json.load(f)
                self.positive_words = set(data["positive_words"])
                self.negative_words = set(data["negative_words"])
                self.aspects = {k: set(v) for k, v in data["aspects"].items()}
                self.intensity_modifiers = data["intensity_modifiers"]
                
            logger.info(f"Successfully loaded model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            logger.warning("Using default sentiment words and aspects")
            
    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score between -1 (negative) and 1 (positive)
        """
        try:
            # For testing, return mock sentiment based on positive/negative word count
            positive_words = {
                "amazing", "great", "excellent", "good", "wonderful", 
                "fantastic", "awesome", "love", "enjoy", "beautiful",
                "impressive", "helpful", "worth", "recommended"
            }
            
            negative_words = {
                "bad", "poor", "terrible", "awful", "horrible",
                "disappointing", "waste", "avoid", "crowded", "expensive",
                "rude", "dirty", "broken", "slow", "boring"
            }
            
            # Normalize text
            text = text.lower()
            words = set(text.split())
            
            # Count positive and negative words
            pos_count = len(words.intersection(positive_words))
            neg_count = len(words.intersection(negative_words))
            
            # Calculate sentiment score between -1 and 1
            total = pos_count + neg_count
            if total == 0:
                return 0.0
                
            sentiment = (pos_count - neg_count) / total
            return max(min(sentiment, 1.0), -1.0)  # Clamp between -1 and 1
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}", exc_info=True)
            return 0.0  # Neutral sentiment on error

    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze sentiment in the given text using enhanced approach.
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentScore with overall score, confidence, and aspect scores
        """
        try:
            # Use TextBlob for base sentiment
            blob = TextBlob(text)
            base_score = blob.sentiment.polarity
            base_confidence = abs(base_score)  # Higher magnitude = higher confidence
            
            # Tokenize text
            if self.nlp:
                doc = self.nlp(text.lower())
                tokens = [token.text for token in doc]
                lemmas = [token.lemma_ for token in doc]
            else:
                tokens = text.lower().split()
                lemmas = tokens
            
            # Count sentiment words and calculate score
            pos_count = sum(1 for word in lemmas if word in self.positive_words)
            neg_count = sum(1 for word in lemmas if word in self.negative_words)
            
            total_count = pos_count + neg_count
            if total_count > 0:
                # Calculate weighted score (-1 to 1)
                lexicon_score = (pos_count - neg_count) / total_count
                
                # Apply intensity modifiers
                for i, token in enumerate(tokens):
                    if token in self.intensity_modifiers:
                        # Check next word for sentiment
                        if i + 1 < len(lemmas):
                            next_word = lemmas[i + 1]
                            modifier = self.intensity_modifiers[token]
                            if next_word in self.positive_words:
                                lexicon_score *= modifier
                            elif next_word in self.negative_words:
                                lexicon_score *= -modifier
                
                # Combine TextBlob and lexicon scores
                final_score = (base_score + lexicon_score) / 2
                
                # Calculate confidence based on agreement and magnitude
                score_diff = abs(base_score - lexicon_score)
                agreement_factor = 1 - (score_diff / 2)  # 1 = perfect agreement
                confidence = min((base_confidence + (total_count / 10)) * agreement_factor, 1.0)
            else:
                # Fall back to TextBlob if no lexicon matches
                final_score = base_score
                confidence = base_confidence
            
            # Analyze aspect-specific sentiment
            aspect_scores = {}
            for aspect, terms in self.aspects.items():
                aspect_mentions = []
                for i, lemma in enumerate(lemmas):
                    if lemma in terms:
                        # Look for sentiment words in context window
                        start = max(0, i - 3)
                        end = min(len(lemmas), i + 4)
                        context = lemmas[start:end]
                        
                        # Calculate sentiment in context
                        pos = sum(1 for w in context if w in self.positive_words)
                        neg = sum(1 for w in context if w in self.negative_words)
                        
                        if pos + neg > 0:
                            score = (pos - neg) / (pos + neg)
                            aspect_mentions.append(score)
                
                if aspect_mentions:
                    aspect_scores[aspect] = sum(aspect_mentions) / len(aspect_mentions)
            
            return SentimentScore(
                score=max(-1.0, min(1.0, final_score)),  # Clamp between -1 and 1
                confidence=confidence,
                aspect_scores=aspect_scores
            )
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
            return SentimentScore(score=0.0, confidence=0.0, aspect_scores={})
