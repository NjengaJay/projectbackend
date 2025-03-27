"""
Comprehensive test script for the Travel Assistant NLP model
Testing intent classification, NER, and sentiment analysis
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.model import TravelAssistantNLP
import json
from typing import Dict, Any, List
from tabulate import tabulate

def run_comprehensive_test(model: TravelAssistantNLP):
    """Run comprehensive tests on the model"""
    test_cases = {
        "Intent Classification": [
            "I need to book a first-class flight to Tokyo for next week",
            "Can you recommend a 5-star hotel in Dubai with a pool?",
            "What are some authentic Italian restaurants in Rome?",
            "What are the must-visit attractions in Paris?",
            "How do I get from JFK airport to Manhattan?",
            "Need a vegetarian-friendly restaurant near the Eiffel Tower",
            "Looking for budget hotels in Amsterdam city center",
            "What's the best way to travel between London and Paris?",
            "Book me a round-trip flight to Barcelona",
            "Show me historical landmarks in Athens"
        ],
        "Sentiment Analysis": [
            "This hotel was absolutely amazing, the staff was incredibly helpful!",
            "Terrible experience, worst flight I've ever taken.",
            "The restaurant was okay, nothing special but decent food.",
            "I'm really excited about my upcoming trip to Hawaii!",
            "The tour guide was knowledgeable but the tour was too rushed.",
            "Fantastic service and beautiful views from my room!",
            "The airport shuttle was late and the driver was rude.",
            "Great value for money, would definitely recommend!",
            "The food was mediocre and overpriced.",
            "Perfect location, friendly staff, and clean rooms!"
        ],
        "Complex Queries": [
            "I had a terrible experience at the Hilton in Paris, but the Eiffel Tower view was amazing",
            "Need to book a business class flight to Tokyo next Monday, preferably in the morning",
            "Looking for family-friendly restaurants near the British Museum in London for this weekend",
            "Can you recommend a luxury hotel in Dubai Marina with a spa for my honeymoon in December?",
            "What's the fastest way to get from Barcelona Airport to Sagrada Familia on Tuesday afternoon?"
        ]
    }
    
    results = {
        "Intent Classification": [],
        "Sentiment Analysis": [],
        "Complex Queries": []
    }
    
    print("\n🔍 Running Comprehensive Tests...\n")
    
    # Test Intent Classification
    print("1️⃣ Testing Intent Classification")
    print("-" * 80)
    for query in test_cases["Intent Classification"]:
        result = model.predict(query)
        results["Intent Classification"].append({
            "Query": query,
            "Predicted Intent": result["intent"],
            "Confidence": f"{result['confidence']:.2f}",
            "Entities": result["entities"]
        })
        print(f"Query: {query}")
        print(f"Intent: {result['intent']} (Confidence: {result['confidence']:.2f})")
        print(f"Entities: {result['entities']}")
        print("-" * 80)
    
    # Test Sentiment Analysis
    print("\n2️⃣ Testing Sentiment Analysis")
    print("-" * 80)
    for query in test_cases["Sentiment Analysis"]:
        result = model.predict(query)
        results["Sentiment Analysis"].append({
            "Query": query,
            "Sentiment": result["sentiment"],
            "Entities": result["entities"]
        })
        print(f"Query: {query}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Entities: {result['entities']}")
        print("-" * 80)
    
    # Test Complex Queries
    print("\n3️⃣ Testing Complex Queries")
    print("-" * 80)
    for query in test_cases["Complex Queries"]:
        result = model.predict(query)
        results["Complex Queries"].append({
            "Query": query,
            "Intent": result["intent"],
            "Sentiment": result["sentiment"],
            "Entities": result["entities"]
        })
        print(f"Query: {query}")
        print(f"Intent: {result['intent']} (Confidence: {result['confidence']:.2f})")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Entities: {result['entities']}")
        print("-" * 80)
    
    # Save results
    with open("comprehensive_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n📊 Test Results Summary")
    print("Results have been saved to comprehensive_test_results.json")

if __name__ == "__main__":
    # Initialize model
    print("Initializing model...")
    model = TravelAssistantNLP()
    
    # Load trained model
    model.load_model("models/intent_classifier")
    
    # Run tests
    run_comprehensive_test(model)
