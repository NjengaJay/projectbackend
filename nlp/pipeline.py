from typing import Dict, List, Optional, Tuple, Union
import re
from .sentiment_analyzer import TripAdvisorSentimentAnalyzer
from .intent_classifier import IntentClassifier
from .entity_recognizer import EntityRecognizer
from .models import Entity, NLPResponse
from .text_processor import TextProcessor

class DutchTravelNLPPipeline:
    """NLP Pipeline for Dutch Travel Assistant"""
    
    def __init__(self):
        """Initialize the NLP pipeline components"""
        self.text_processor = TextProcessor()
        self.intent_classifier = IntentClassifier(self.text_processor)
        self.entity_recognizer = EntityRecognizer()
        self.sentiment_analyzer = TripAdvisorSentimentAnalyzer()
        
        # Load models and resources
        self._load_resources()
        
    def _load_resources(self):
        """Load necessary models and resources"""
        # Ensure models are loaded
        self.intent_classifier.load_model()
        self.entity_recognizer.load_patterns()
        
    def process(self, query: str) -> NLPResponse:
        """Process a user query through the NLP pipeline
        
        Args:
            query: User's input query
            
        Returns:
            NLPResponse containing intent, entities, and sentiment analysis if applicable
        """
        # Preserve original text for entity extraction
        original_query = query
        
        # Preprocess query
        query = self._preprocess_query(query)
        
        # Extract entities from original query to preserve exact matches
        entities = self.entity_recognizer.extract_entities(original_query)
        
        # Classify intent
        intent, confidence = self.intent_classifier.classify(query)
        
        # Initialize response
        response = NLPResponse(
            intent=intent,
            confidence=confidence,
            entities=entities
        )
        
        # If intent is get_review, perform sentiment analysis
        if intent == "get_review":
            # Find attraction entity
            attraction = next(
                (entity.value for entity in entities if entity.type == "attraction"),
                None
            )
            if attraction:
                sentiment = self.sentiment_analyzer.analyze(original_query)
                response.sentiment_analysis = sentiment
                
        return response
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the input query
        
        Args:
            query: Raw input query
            
        Returns:
            Cleaned and normalized query
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = " ".join(query.split())
        
        # Remove punctuation except question marks
        query = re.sub(r'[^\w\s?]', ' ', query)
        
        # Normalize common variations
        replacements = {
            "museum": ["museums", "gallery", "galleries"],
            "train": ["trains", "rail", "railway"],
            "bus": ["buses", "coach", "coaches"],
            "tram": ["trams", "streetcar", "trolley"],
            "ticket": ["tickets", "fare", "fares"],
            "cost": ["price", "prices", "fee", "fees"],
            "accessible": ["accessibility", "wheelchair", "disabled"],
            "route": ["way", "path", "direction", "directions"]
        }
        
        for standard, variants in replacements.items():
            for variant in variants:
                query = re.sub(r'\b{}\b'.format(variant), standard, query)
                
        return query
