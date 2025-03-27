"""
Advanced text processing with synonym handling and spelling correction
"""
from typing import Dict, List, Optional, Set, Tuple
import nltk
from nltk.corpus import wordnet
from rapidfuzz import fuzz, process
import spacy
from pathlib import Path
import json
import logging
import requests
from spellchecker import SpellChecker

class TextProcessor:
    def __init__(self):
        """Initialize text processor with enhanced synonym handling"""
        self.nlp = spacy.load("en_core_web_sm")
        
        # Common location abbreviations and synonyms
        self.location_synonyms = {
            "AMS": "amsterdam",
            "RTM": "rotterdam",
            "UTR": "utrecht",
            "GRQ": "groningen",
            "EIN": "eindhoven",
            "MST": "maastricht",
            "the hague": "den haag",
            "den haag": "den haag",
            "schiphol": "amsterdam airport schiphol",
            "amsterdam airport": "amsterdam airport schiphol",
            "rotterdam airport": "rotterdam the hague airport",
            "eindhoven airport": "eindhoven airport",
            "utrecht cs": "utrecht centraal",
            "utrecht central": "utrecht centraal",
            "utrecht station": "utrecht centraal",
            "utrecht central station": "utrecht centraal",
            "amsterdam cs": "amsterdam centraal",
            "amsterdam central": "amsterdam centraal",
            "amsterdam station": "amsterdam centraal",
            "amsterdam central station": "amsterdam centraal",
            "rotterdam cs": "rotterdam centraal",
            "rotterdam central": "rotterdam centraal",
            "rotterdam station": "rotterdam centraal",
            "rotterdam central station": "rotterdam centraal",
            "the dam": "amsterdam",
            "adam": "amsterdam",
            "rdam": "rotterdam",
            "dhg": "den haag"
        }
        
        # Transport synonyms
        self.transport_synonyms = {
            "train": ["ns", "rail", "railway", "trains"],
            "tram": ["streetcar", "trolley", "trams"],
            "bus": ["coach", "buses"],
            "metro": ["subway", "underground"],
            "bike": ["bicycle", "cycling", "bikes"],
            "ferry": ["boat", "waterbus"],
            "taxi": ["cab", "uber", "lyft"]
        }
        
        # Initialize spelling checker
        self.spell = SpellChecker()
        
        # Add domain-specific words to spell checker
        domain_words = (
            set(self.location_synonyms.keys()) |
            set(self.location_synonyms.values()) |
            {word for synonyms in self.transport_synonyms.values() for word in synonyms} |
            {"gvb", "ns", "ov", "chipkaart", "rijksmuseum", "keukenhof", "efteling"}
        )
        self.spell.word_frequency.load_words(list(domain_words))
        
        # Initialize fuzzy matching thresholds
        self.MIN_SIMILARITY = 85  # Increased threshold for better accuracy
        
        # Load custom location data
        self.load_location_data()
        
        # Create a combined synonym dictionary
        self.synonym_dict = {**self.location_synonyms, **{k: ' '.join(v) for k, v in self.transport_synonyms.items()}}
    
    def load_location_data(self):
        """Load location data from various sources"""
        try:
            # Load custom location database
            db_path = Path(__file__).parent / "data" / "location_database.json"
            if db_path.exists():
                with open(db_path, "r", encoding="utf-8") as f:
                    self.location_database = json.load(f)
            else:
                self.location_database = {}
                
            # Expand synonyms with database entries
            for entry in self.location_database.values():
                if "aliases" in entry:
                    for alias in entry["aliases"]:
                        self.location_synonyms[alias.lower()] = entry["name"].lower()
                        
        except Exception as e:
            logging.error(f"Error loading location data: {e}")
            self.location_database = {}
    
    def expand_synonyms(self, text: str) -> str:
        """Expand synonyms in text"""
        # Debug logging
        print(f"\nExpanding synonyms in: {text}")
        
        # First check for multi-word phrases
        phrases = {
            'wheelchair accessible': ['wheelchair accessible'],
            'wheelchair access': ['wheelchair access'],
            'disabled access': ['disabled access', 'handicap access'],
            'disability access': ['disability access', 'handicap access'],
            'show me museums': ['show me museums', 'find museums'],
            'museums in': ['museums in', 'museums at'],
            'tourist attractions in': ['tourist attractions in', 'places to visit in']
        }
        
        # Process text word by word to preserve phrases
        words = text.split()
        result = []
        i = 0
        
        while i < len(words):
            matched_phrase = False
            
            # Try to match phrases first
            for phrase, synonyms in phrases.items():
                phrase_words = phrase.split()
                if (i + len(phrase_words) <= len(words) and 
                    ' '.join(words[i:i+len(phrase_words)]).lower() == phrase):
                    print(f"Checking phrase: {' '.join(words[i:i+len(phrase_words)])}")
                    result.extend(phrase_words)
                    i += len(phrase_words)
                    matched_phrase = True
                    break
            
            if not matched_phrase:
                # Handle single word
                word = words[i].lower()
                print(f"Checking word: {word}")
                
                # Add word-level synonyms here if needed
                result.append(word)
                i += 1
        
        # Join words back together
        first_pass = ' '.join(result)
        print(f"First pass result: {first_pass}")
        
        # Do a second pass to catch any remaining phrases
        final_result = first_pass
        for phrase, synonyms in phrases.items():
            if phrase in final_result.lower():
                continue
            for synonym in synonyms:
                if synonym in final_result.lower():
                    final_result = final_result.replace(synonym, phrase)
                    break
        
        print(f"Final result: {final_result}")
        return final_result
    
    def correct_spelling(self, word: str) -> str:
        """Correct spelling using fuzzy matching and domain-specific dictionary"""
        if not word:
            return ""
            
        # Don't correct numbers or special characters
        if any(c.isdigit() for c in word) or not word.isalnum():
            return word
        
        # Check domain-specific spell checker
        misspelled = self.spell.unknown([word])
        if misspelled:
            correction = self.spell.correction(word)
            return correction if correction else word
        
        return word
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text by correcting spelling and expanding synonyms"""
        if not text:
            return ""
        
        # Debug logging
        print(f"\nPreprocessing text: {text}")
        
        # Remove punctuation except apostrophes and normalize whitespace
        text = ' '.join(''.join(c for c in text if c.isalnum() or c.isspace() or c == "'").split())
        
        # Expand synonyms
        text = self.expand_synonyms(text)
        print(f"After synonym expansion: {text}")
        
        # Correct spelling and remove duplicates while preserving phrases
        words = []
        seen = set()
        phrases = {
            'wheelchair accessible', 'wheelchair access', 'disabled access',
            'disability access', 'handicap access', 'wheelchair',
            'what do people say about', 'what do people think about',
            'what do others say about', 'what do others think about',
            'show me museums', 'museums in', 'tourist attractions in',
            'van gogh museum', 'rijksmuseum', 'anne frank house'
        }
        
        i = 0
        words_list = text.split()
        while i < len(words_list):
            matched_phrase = False
            
            # Try to match phrases first
            for phrase in phrases:
                phrase_words = phrase.split()
                if (i + len(phrase_words) <= len(words_list) and 
                    ' '.join(words_list[i:i+len(phrase_words)]).lower() == phrase.lower()):
                    words.extend(phrase_words)
                    seen.update(w.lower() for w in phrase_words)
                    i += len(phrase_words)
                    matched_phrase = True
                    break
            
            if not matched_phrase:
                # Handle single word
                word = self.correct_spelling(words_list[i])
                if word.lower() not in seen:
                    words.append(word)
                    seen.add(word.lower())
                i += 1
        
        text = ' '.join(words)
        print(f"After spelling correction: {text}")
        
        return text
    
    def extract_entities_with_spacy(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using spaCy NER"""
        doc = self.nlp(text)
        entities = {
            "location": [],
            "attraction": [],
            "transport": [],
            "time": [],
            "date": [],
            "organization": []
        }
        
        for ent in doc.ents:
            if ent.label_ in ["GPE", "LOC"]:
                entities["location"].append(ent.text)
            elif ent.label_ in ["ORG", "FAC"]:
                entities["attraction"].append(ent.text)
            elif ent.label_ in ["DATE", "TIME"]:
                if any(time_word in ent.text.lower() 
                      for time_word in ["am", "pm", "hour", "minute"]):
                    entities["time"].append(ent.text)
                else:
                    entities["date"].append(ent.text)
        
        # Add transport entities from our synonym dictionary
        words = set(text.lower().split())
        for transport, synonyms in self.transport_synonyms.items():
            if transport in words or any(syn in words for syn in synonyms):
                entities["transport"].append(transport)
        
        return entities
    
    def update_location_database(self, location_name: str, data: dict):
        """Update the location database with new information"""
        if location_name.lower() not in self.location_database:
            self.location_database[location_name.lower()] = data
            self.save_location_data()
    
    def save_location_data(self):
        """Save location database to file"""
        try:
            db_path = Path(__file__).parent / "data" / "location_database.json"
            db_path.parent.mkdir(exist_ok=True)
            with open(db_path, "w", encoding="utf-8") as f:
                json.dump(self.location_database, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving location data: {e}")
