"""
Generate sample training data for the Travel Assistant NLP model
"""
import json
from typing import List, Tuple, Dict

def generate_sample_data() -> Tuple[List[str], List[str]]:
    """Generate sample training data for intent classification"""
    data = {
        "book_flight": [
            "I need to book a first-class flight to Tokyo for next week",
            "Book me a round-trip flight to Barcelona",
            "Need to book a business class flight to Tokyo next Monday",
            "I'd like to book a flight from London to Paris",
            "Can you help me book a flight to New York?",
            "Looking to book an economy flight to Dubai",
            "Book a direct flight to Singapore for tomorrow",
            "I want to book a morning flight to San Francisco",
            "Help me book a return flight from Berlin to Rome",
            "Need to book an urgent flight to Mumbai",
            "Book me on the next available flight to Seoul",
            "I'd like to book a late evening flight to Bangkok",
            "Need to book a connecting flight through Dubai",
            "Book a first-class ticket to Los Angeles",
            "Looking to book an international flight to Tokyo"
        ],
        "hotel_booking": [
            "Book a room at the Hilton in Paris",
            "Need a 5-star hotel in Dubai with a pool",
            "Looking for a luxury hotel in Dubai Marina with spa",
            "Book me a suite at the Ritz Carlton",
            "Need a hotel room near Times Square",
            "Book a beachfront resort in Maldives",
            "Looking for a boutique hotel in Rome",
            "Need accommodation near the Eiffel Tower",
            "Book a family room at Marriott London",
            "Need a hotel with conference facilities in Singapore",
            "Looking for a ski resort in the Swiss Alps",
            "Book a penthouse suite with ocean view",
            "Need a pet-friendly hotel in Amsterdam",
            "Looking for an all-inclusive resort in Cancun",
            "Book a room with mountain view in Queenstown"
        ],
        "restaurant_recommendation": [
            "Recommend Italian restaurants in Rome",
            "Looking for vegetarian restaurants near the Eiffel Tower",
            "Best sushi places in Tokyo",
            "Family-friendly restaurants in London",
            "Recommend a romantic restaurant in Venice",
            "Looking for Michelin-starred restaurants in Paris",
            "Best local food spots in Bangkok",
            "Recommend seafood restaurants in Sydney",
            "Halal restaurants in Dubai Mall",
            "Traditional tapas bars in Barcelona",
            "Fine dining options in Singapore",
            "Best brunch spots in New York",
            "Authentic Mexican restaurants in Mexico City",
            "Rooftop restaurants with city views",
            "Kid-friendly cafes in Amsterdam"
        ],
        "tourist_attraction": [
            "What are the must-visit places in Paris?",
            "Popular attractions in Tokyo",
            "Historical sites in Rome",
            "Best beaches in Bali",
            "Tourist spots in New York City",
            "Famous landmarks in London",
            "Cultural attractions in Kyoto",
            "Must-see places in Barcelona",
            "Top tourist destinations in Dubai",
            "Historical monuments in Athens",
            "Natural wonders in New Zealand",
            "Art galleries in Florence",
            "Ancient temples in Cambodia",
            "Safari parks in Kenya",
            "UNESCO sites in Egypt"
        ],
        "transportation": [
            "How to get from JFK airport to Manhattan",
            "Best way to travel between London and Paris",
            "Public transport options in Tokyo",
            "Getting from Barcelona Airport to Sagrada Familia",
            "Train schedule from Rome to Florence",
            "Airport shuttle service in Singapore",
            "Metro system in Dubai",
            "Bus routes in Amsterdam",
            "Taxi services in Hong Kong",
            "Ferry schedule to Greek Islands",
            "Getting around in Venice",
            "Airport transfers in Bangkok",
            "Car rental options in Los Angeles",
            "Train passes in Switzerland",
            "Local transportation in Istanbul"
        ]
    }
    
    # Generate training data
    texts = []
    labels = []
    for intent, examples in data.items():
        for example in examples:
            texts.append(example)
            labels.append(intent)
    
    return texts, labels

def save_sample_data(output_file: str = "travel_training_data.json"):
    """Save sample data to a JSON file"""
    texts, labels = generate_sample_data()
    
    data = {
        "data": [
            {"text": text, "label": label} for text, label in zip(texts, labels)
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    save_sample_data()
