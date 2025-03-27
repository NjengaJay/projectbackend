from transformers import pipeline
from nltk.tokenize import sent_tokenize
import nltk
from typing import Dict, List, Tuple

class SentimentAnalyzer:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize the sentiment analysis pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True
        )

    def analyze_review(self, text: str) -> Dict:
        """
        Analyze the sentiment of a review text and extract key aspects
        
        Args:
            text (str): The review text to analyze
            
        Returns:
            Dict containing sentiment score and extracted aspects
        """
        # Split text into sentences
        sentences = sent_tokenize(text)
        
        # Analyze sentiment for each sentence
        sentiments = self.sentiment_pipeline(sentences)
        
        # Calculate overall sentiment score
        overall_score = self._calculate_overall_score(sentiments)
        
        # Extract aspects and their sentiments
        aspects = self._extract_aspects(sentences, sentiments)
        
        return {
            'overall_score': overall_score,
            'aspects': aspects,
            'sentence_sentiments': [
                {
                    'sentence': sent,
                    'sentiment': sent_result['label'],
                    'score': sent_result['score']
                }
                for sent, sent_result in zip(sentences, sentiments)
            ]
        }

    def _calculate_overall_score(self, sentiments: List[Dict]) -> float:
        """
        Calculate overall sentiment score from individual sentence sentiments
        """
        scores = []
        for sent in sentiments:
            # Convert POSITIVE/NEGATIVE to numeric score
            score = sent['score'] if sent['label'] == 'POSITIVE' else 1 - sent['score']
            scores.append(score)
        
        # Return average score normalized to [-1, 1] range
        return (2 * (sum(scores) / len(scores))) - 1 if scores else 0

    def _extract_aspects(self, sentences: List[str], sentiments: List[Dict]) -> List[Dict]:
        """
        Extract key aspects and their associated sentiments from the review
        
        This is a simple implementation that could be enhanced with aspect-based
        sentiment analysis models or more sophisticated NLP techniques
        """
        aspects = []
        aspect_keywords = {
            'location': ['location', 'place', 'area', 'neighborhood'],
            'accessibility': ['accessible', 'wheelchair', 'disability', 'ramp'],
            'activities': ['activity', 'tour', 'experience', 'attraction'],
            'transport': ['transport', 'bus', 'train', 'taxi'],
            'accommodation': ['hotel', 'hostel', 'apartment', 'room'],
            'food': ['restaurant', 'food', 'meal', 'dining']
        }
        
        for sentence, sentiment in zip(sentences, sentiments):
            sentence_lower = sentence.lower()
            
            # Check for each aspect
            for aspect, keywords in aspect_keywords.items():
                if any(keyword in sentence_lower for keyword in keywords):
                    aspects.append({
                        'aspect': aspect,
                        'sentiment': sentiment['label'],
                        'score': sentiment['score'],
                        'text': sentence
                    })
        
        return aspects

    def batch_analyze_reviews(self, reviews: List[str]) -> List[Dict]:
        """
        Analyze multiple reviews in batch
        """
        return [self.analyze_review(review) for review in reviews]
