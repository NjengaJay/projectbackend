"""
Entity recognition module for travel-related queries
"""

import logging
from typing import Dict, List, Optional, Any
import spacy
import re
import os
import json

logger = logging.getLogger(__name__)

class EntityRecognizer:
    """Recognizes travel-related entities in text"""
    
    def __init__(self, model_path: str = "models"):
        """Initialize entity recognizer with default patterns."""
        try:
            # Initialize spacy
            self.nlp = spacy.load("en_core_web_sm")
            
            # Add entity ruler to pipeline if not present
            if "entity_ruler" not in self.nlp.pipe_names:
                ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            else:
                ruler = self.nlp.get_pipe("entity_ruler")
            
            # Location patterns
            ruler.add_patterns([
                {"label": "LOCATION", "pattern": "London"},
                {"label": "LOCATION", "pattern": "Paris"},
                {"label": "LOCATION", "pattern": "Berlin"},
                {"label": "LOCATION", "pattern": "Rome"},
                {"label": "LOCATION", "pattern": "Madrid"},
                {"label": "LOCATION", "pattern": "Amsterdam"}
            ])
            
            # Attraction type patterns
            ruler.add_patterns([
                {"label": "ATTRACTION_TYPE", "pattern": "museum"},
                {"label": "ATTRACTION_TYPE", "pattern": "museums"},
                {"label": "ATTRACTION_TYPE", "pattern": "park"},
                {"label": "ATTRACTION_TYPE", "pattern": "parks"},
                {"label": "ATTRACTION_TYPE", "pattern": "gallery"},
                {"label": "ATTRACTION_TYPE", "pattern": "galleries"},
                {"label": "ATTRACTION_TYPE", "pattern": "restaurant"},
                {"label": "ATTRACTION_TYPE", "pattern": "restaurants"},
                {"label": "ATTRACTION_TYPE", "pattern": "cafe"},
                {"label": "ATTRACTION_TYPE", "pattern": "cafes"},
                {"label": "ATTRACTION_TYPE", "pattern": "tourist spot"},
                {"label": "ATTRACTION_TYPE", "pattern": "tourist spots"},
                {"label": "ATTRACTION_TYPE", "pattern": "landmark"},
                {"label": "ATTRACTION_TYPE", "pattern": "landmarks"}
            ])
            
            # Attraction patterns
            ruler.add_patterns([
                {"label": "ATTRACTION", "pattern": "Louvre"},
                {"label": "ATTRACTION", "pattern": "British Museum"},
                {"label": "ATTRACTION", "pattern": "Eiffel Tower"},
                {"label": "ATTRACTION", "pattern": "Tower Bridge"},
                {"label": "ATTRACTION", "pattern": "Colosseum"},
                {"label": "ATTRACTION", "pattern": "Rijksmuseum"},
                {"label": "ATTRACTION", "pattern": "Prado Museum"}
            ])
            
            # Transport mode patterns
            ruler.add_patterns([
                {"label": "TRANSPORT_MODE", "pattern": "train"},
                {"label": "TRANSPORT_MODE", "pattern": "bus"},
                {"label": "TRANSPORT_MODE", "pattern": "plane"},
                {"label": "TRANSPORT_MODE", "pattern": "car"},
                {"label": "TRANSPORT_MODE", "pattern": "taxi"},
                {"label": "TRANSPORT_MODE", "pattern": "subway"},
                {"label": "TRANSPORT_MODE", "pattern": "metro"}
            ])
            
            logger.info("Successfully initialized entity recognizer")
            
        except Exception as e:
            logger.error(f"Error initializing entity recognizer: {str(e)}", exc_info=True)
            raise
            
    def _load_custom_model(self, model_path: str):
        """Load custom NER model if available"""
        try:
            custom_ner = spacy.load(model_path)
            self.nlp.add_pipe("custom_ner", source=custom_ner)
            logger.info(f"Loaded custom NER model from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load custom NER model: {str(e)}")
            
    def _extract_money(self, text: str) -> Optional[float]:
        """Extract monetary amounts from text"""
        money_pattern = r'\$?\d+(?:\.\d{2})?'
        matches = re.findall(money_pattern, text)
        if matches:
            # Remove $ and convert to float
            amount = float(matches[0].replace('$', ''))
            return amount
        return None
        
    def _extract_locations(self, doc) -> Dict[str, str]:
        """Extract location entities from text"""
        locations = {}
        
        # Look for GPE (cities, countries) and LOC (geographical locations)
        location_ents = [ent for ent in doc.ents if ent.label_ in ["GPE", "LOC", "LOCATION"]]
        
        if len(location_ents) >= 2:
            # Assume first is start, last is end if multiple locations
            locations["start_location"] = location_ents[0].text
            locations["end_location"] = location_ents[-1].text
        elif len(location_ents) == 1:
            # For single location, check context to determine if it's start, end, or just location
            text = doc.text.lower()
            if "from" in text and location_ents[0].start > text.index("from"):
                locations["start_location"] = location_ents[0].text
            elif "to" in text and location_ents[0].start > text.index("to"):
                locations["end_location"] = location_ents[0].text
            else:
                locations["location"] = location_ents[0].text
        
        return locations
        
    def _extract_transport_mode(self, doc) -> Optional[str]:
        """Extract transport mode from text"""
        transport_mode_ents = [ent for ent in doc.ents if ent.label_ == "TRANSPORT_MODE"]
        if transport_mode_ents:
            return transport_mode_ents[0].text
        return None
        
    def _extract_attraction_type(self, doc) -> Optional[str]:
        """Extract attraction type from text"""
        attraction_type_ents = [ent for ent in doc.ents if ent.label_ == "ATTRACTION_TYPE"]
        if attraction_type_ents:
            return attraction_type_ents[0].text
        return None
        
    def _extract_attraction(self, doc) -> Optional[str]:
        """Extract attraction from text"""
        attraction_ents = [ent for ent in doc.ents if ent.label_ == "ATTRACTION"]
        if attraction_ents:
            return attraction_ents[0].text
        return None
        
    def _extract_accessibility(self, doc) -> List[str]:
        """Extract accessibility requirements from text"""
        text_lower = doc.text.lower()
        accessibility = []
        
        # Check custom patterns
        accessibility_patterns = ["wheelchair", "elevator", "ramp", "accessible", "disability", "disabled", "handicap", "lift"]
        for feature in accessibility_patterns:
            if feature in text_lower:
                accessibility.append(feature)
                
        return accessibility
        
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extract travel-related entities from text."""
        try:
            # Process text
            doc = self.nlp(text.lower())
            
            # Extract entities
            entities = {}
            
            # Extract locations
            locations = self._extract_locations(doc)
            if locations:
                if locations.get("start_location"):
                    entities["start_location"] = locations["start_location"]
                if locations.get("end_location"):
                    entities["end_location"] = locations["end_location"]
                if not locations.get("start_location") and not locations.get("end_location"):
                    entities["location"] = next(iter(locations.values()))
            
            # Extract transport mode
            mode = self._extract_transport_mode(doc)
            if mode:
                entities["transport_mode"] = mode
            
            # Extract attraction type
            attraction_type = self._extract_attraction_type(doc)
            if attraction_type:
                entities["attraction_type"] = attraction_type
            
            # Extract attraction name
            attraction = self._extract_attraction(doc)
            if attraction:
                entities["attraction"] = attraction
            
            # Extract monetary amounts
            max_cost = self._extract_money(text)
            if max_cost:
                entities["max_cost"] = max_cost
                
            # Extract accessibility features
            accessibility = self._extract_accessibility(doc)
            if accessibility:
                entities["accessibility_features"] = accessibility
                
            logger.info(f"Extracted entities: {entities}")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return {}
            
    def save_model(self, model_path: str):
        """
        Save model data to disk
        
        Args:
            model_path: Directory to save model files
        """
        try:
            # Create model directory if it doesn't exist
            os.makedirs(model_path, exist_ok=True)
            
            # Save patterns
            patterns_path = os.path.join(model_path, "patterns.json")
            with open(patterns_path, 'w') as f:
                json.dump(self.nlp.get_pipe("entity_ruler").patterns, f)
                
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
            # Load patterns
            patterns_path = os.path.join(model_path, "patterns.json")
            with open(patterns_path, 'r') as f:
                patterns = json.load(f)
                
            # Update ruler with loaded patterns
            self.nlp.get_pipe("entity_ruler").add_patterns(patterns)
                
            logger.info(f"Successfully loaded model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            logger.warning("Using default patterns")
