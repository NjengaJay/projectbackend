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
            
            # Location patterns for Dutch cities
            ruler.add_patterns([
                {"label": "LOCATION", "pattern": "Amsterdam"},
                {"label": "LOCATION", "pattern": "Rotterdam"},
                {"label": "LOCATION", "pattern": "Utrecht"},
                {"label": "LOCATION", "pattern": "Den Haag"},
                {"label": "LOCATION", "pattern": "The Hague"},
                {"label": "LOCATION", "pattern": "Eindhoven"},
                {"label": "LOCATION", "pattern": "Groningen"},
                {"label": "LOCATION", "pattern": "Tilburg"},
                {"label": "LOCATION", "pattern": "Almere"},
                {"label": "LOCATION", "pattern": "Breda"},
                {"label": "LOCATION", "pattern": "Nijmegen"}
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
            
            # Dutch attraction patterns
            ruler.add_patterns([
                {"label": "ATTRACTION", "pattern": "Rijksmuseum"},
                {"label": "ATTRACTION", "pattern": "Van Gogh Museum"},
                {"label": "ATTRACTION", "pattern": "Anne Frank House"},
                {"label": "ATTRACTION", "pattern": "Royal Palace"},
                {"label": "ATTRACTION", "pattern": "NEMO Science Museum"},
                {"label": "ATTRACTION", "pattern": "Artis Zoo"},
                {"label": "ATTRACTION", "pattern": "Vondelpark"},
                {"label": "ATTRACTION", "pattern": "Maritime Museum"},
                {"label": "ATTRACTION", "pattern": "Keukenhof Gardens"},
                {"label": "ATTRACTION", "pattern": "Efteling"},
                {"label": "ATTRACTION", "pattern": "Madurodam"}
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
        
    def extract_entities(self, text: str) -> Dict[str, str]:
        """Extract entities from text."""
        doc = self.nlp(text)
        entities = {}
        
        # Extract locations (looking for two locations for route queries)
        locations = [ent.text for ent in doc.ents if ent.label_ == "LOCATION"]
        if len(locations) >= 2:
            entities["origin"] = locations[0]
            entities["destination"] = locations[1]
        elif len(locations) == 1:
            # For attraction queries, single location is the target
            entities["location"] = locations[0]
        
        # Extract attraction types
        attraction_types = [ent.text for ent in doc.ents if ent.label_ == "ATTRACTION_TYPE"]
        if attraction_types:
            entities["attraction_type"] = attraction_types[0]
        
        # Extract specific attractions
        attractions = [ent.text for ent in doc.ents if ent.label_ == "ATTRACTION"]
        if attractions:
            entities["attraction"] = attractions[0]
        
        # Extract transport modes
        transport_modes = [ent.text for ent in doc.ents if ent.label_ == "TRANSPORT_MODE"]
        if transport_modes:
            entities["transport_mode"] = transport_modes[0]
            
        return entities
            
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
