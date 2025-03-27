"""
Data models for NLP components
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class Entity:
    """Class to represent an extracted entity"""
    type: str  # location, transport_mode, attraction, cost, accessibility
    value: str
    start: int
    end: int

@dataclass
class NLPResponse:
    """Class to represent the structured response from the NLP pipeline"""
    intent: str
    confidence: float
    entities: List[Entity]
    sentiment_analysis: Optional[Dict] = None
