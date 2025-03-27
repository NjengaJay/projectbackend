"""
Intent classification module for the travel assistant
"""

import logging
import os
import re
from typing import Dict, List, Tuple, Optional
import json
import spacy

logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class IntentClassifier:
    """Simple rule-based intent classifier."""
    
    def __init__(self):
        """Initialize the intent classifier with default patterns."""
        self.patterns = {
            'route_query': [
                'how do i get', 'how to get', 'route', 'way', 'path', 'travel', 'journey',
                'directions', 'navigate'
            ],
            'accessibility_query': [
                'wheelchair', 'accessible', 'disability', 'disabled', 'mobility',
                'step-free', 'assistance', 'accessibility'
            ],
            'cost_query': [
                'cost', 'price', 'fare', 'ticket', 'cheap', 'expensive', 'budget',
                'how much'
            ],
            'time_query': [
                'time', 'duration', 'long', 'fast', 'quick', 'schedule', 'when',
                'how long'
            ],
            'hotel_query': [
                'hotel', 'stay', 'accommodation', 'place to stay', 'lodging', 'room',
                'where to stay', 'book a room'
            ],
            'attraction_query': [
                'tourist', 'attraction', 'visit', 'see', 'sightseeing', 'museum',
                'landmark', 'places to go', 'places to see', 'things to do'
            ],
            'info_query': [
                'info', 'information', 'about', 'tell', 'what', 'explain',
                'describe', 'details', 'where', 'which'
            ]
        }
        self.dutch_cities = [
            'amsterdam', 'rotterdam', 'utrecht', 'den haag', 'eindhoven',
            'groningen', 'tilburg', 'almere', 'breda', 'nijmegen'
        ]
        
    def classify(self, text: str) -> Tuple[str, float]:
        """Classify the intent of a text query."""
        text = text.lower()
        
        # Define patterns for each intent
        patterns = {
            "review_query": [
                r"(what|how).*(people|travelers|tourists).*(think|say|rate)",
                r"(review|rating|opinion|feedback|experience).*",
                r".*\b(good|bad|rated|reviewed)\b.*",
                r".*\b(worth|recommend|suggest)\b.*",
                r".*\bthink about\b.*",
                r".*what.*think.*",  # Catch "What do people think about X"
                r".*how.*like.*",    # Catch "How do people like X"
                r".*opinion.*on.*"   # Catch "Opinion on X"
            ],
            "cost_query": [
                r"(how much|cost|price|fare).*(from|to|between)",
                r"(ticket|travel|journey).*(cost|price|fare)",
                r"(cost|price|fare).*(ticket|travel|journey)",
                r".*\bhow much\b.*\b(from|to)\b.*",
                r".*\bcost\b.*\b(from|to)\b.*"
            ],
            "route_query": [
                r"(how|way|route|path).*(to get|to travel|to go)",
                r"(from|between).*(to|and).*(route|path|direction)",
                r"(find|show|get).*(route|path|direction|way)",
                r".*\bhow (do|can|should) (i|we) get\b.*",
                r".*\bhow (do|can|should) (i|we) (travel|go)\b.*",
                r".*\b(from|between) .* (to|and)\b.*"
            ],
            "attraction_query": [
                r"(what|where|which|show me|find|suggest).*(museum|gallery|park|garden|restaurant|landmark|monument|statue|tower|cathedral|church|attraction|place)",
                r"(museum|gallery|park|garden|restaurant|landmark|monument|statue|tower|cathedral|church|attraction|place).*(to (visit|see|explore|go to))",
                r"(best|top|popular|recommended).*(museum|gallery|park|garden|restaurant|landmark|monument|statue|tower|cathedral|church|attraction|place)"
            ]
        }
        
        # Check each pattern
        max_confidence = 0.0
        best_intent = "unknown"
        
        # Debug logging
        logger.info(f"Classifying text: {text}")
        
        for intent, intent_patterns in patterns.items():
            for pattern in intent_patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    # Calculate confidence based on match length and position
                    match_length = match.end() - match.start()
                    match_position = match.start()
                    confidence = (match_length / len(text)) * (1 - match_position / len(text))
                    
                    # Give higher confidence to cost queries when they match
                    if intent == "cost_query":
                        confidence *= 1.2
                    
                    # Give higher confidence to review queries with "think about" pattern
                    if intent == "review_query" and "think about" in text:
                        confidence *= 1.3
                        
                    # Debug logging
                    logger.info(f"Pattern {pattern} matched with confidence {confidence} for intent {intent}")
                    
                    if confidence > max_confidence:
                        max_confidence = confidence
                        best_intent = intent
                        # Debug logging
                        logger.info(f"New best intent: {best_intent} with confidence {max_confidence}")
        
        # If we have a very low confidence, return unknown
        if max_confidence < 0.1:
            return "unknown", 0.0
            
        logger.info(f"Final intent: {best_intent} with confidence {max_confidence}")
        return best_intent, max_confidence

    def extract_entities(self, query: str) -> dict:
        """Extract entities from the query."""
        entities = {}
        
        # Convert query to lowercase for easier matching
        query_lower = query.lower()
        
        # Extract cities for origin/destination
        cities_found = []
        for city in self.dutch_cities:
            if city in query_lower:
                cities_found.append(city)
        
        if len(cities_found) >= 2:
            entities['origin'] = cities_found[0]
            entities['destination'] = cities_found[1]
        elif len(cities_found) == 1:
            if 'from' in query_lower:
                entities['origin'] = cities_found[0]
            else:
                entities['destination'] = cities_found[0]
        
        # Extract location for attractions
        if 'in' in query_lower:
            parts = query_lower.split('in')
            if len(parts) > 1:
                location = parts[1].strip()
                # Check if the location contains a Dutch city
                for city in self.dutch_cities:
                    if city in location:
                        entities['location'] = city
                        break
        
        # Extract attraction type
        attraction_types = ['museum', 'park', 'restaurant', 'cafe', 'theater', 'market']
        for type_ in attraction_types:
            if type_ in query_lower or type_ + 's' in query_lower:
                entities['attraction_type'] = type_
                break
        
        # Extract subject for reviews
        dutch_attractions = [
            'rijksmuseum', 'van gogh museum', 'anne frank house',
            'efteling', 'keukenhof', 'mauritshuis', 'kinderdijk'
        ]
        
        for attraction in dutch_attractions:
            if attraction in query_lower:
                entities['subject'] = attraction
                break
        
        return entities
