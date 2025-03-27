"""
Test script for the Travel Assistant NLP model
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nlp.model import TravelAssistantNLP
from nlp.sample_data import generate_sample_data
from sentiment_analyzer import TripAdvisorSentimentAnalyzer
import pandas as pd
import json
import torch

def initialize_sentiment_analyzer():
    """Initialize and train the sentiment analyzer"""
    print("Initializing sentiment analyzer...")
    analyzer = TripAdvisorSentimentAnalyzer()
    
    # Create sample sentiment data
    sample_reviews = pd.DataFrame({
        'review_text': [
            "The hotel was amazing and the staff was very friendly!",
            "Terrible experience, would not recommend.",
            "The food was decent but the service was slow.",
            "Great location and beautiful views.",
            "The room was dirty and the AC didn't work.",
            "Absolutely loved my stay here!",
            "Not worth the money at all.",
            "Pretty average experience overall.",
            "Beautiful hotel with excellent service!",
            "Worst hotel I've ever stayed in."
        ],
        'rating': [5, 1, 3, 5, 1, 5, 1, 3, 5, 1],
        'sentiment': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0]
    })
    
    try:
        # Train sentiment analyzer
        sample_reviews = analyzer.prepare_data(sample_reviews)
        X_train = sample_reviews['processed_text']
        y_train = sample_reviews['sentiment']
        analyzer.train_model(X_train, y_train)
        analyzer.save_model()
        print("Sentiment analyzer trained and saved successfully")
    except Exception as e:
        print(f"Error training sentiment analyzer: {str(e)}")
        print("Loading pre-trained model instead...")
        analyzer.load_model()
    
    return analyzer

def test_model():
    # Initialize sentiment analyzer
    sentiment_analyzer = initialize_sentiment_analyzer()
    
    # Initialize NLP model
    print("\nInitializing NLP model...")
    nlp_model = TravelAssistantNLP()
    nlp_model.sentiment_analyzer = sentiment_analyzer
    
    # Generate and split training data
    print("Preparing training data...")
    texts, labels = generate_sample_data()
    train_size = int(0.8 * len(texts))
    
    train_texts = texts[:train_size]
    train_labels = labels[:train_size]
    eval_texts = texts[train_size:]
    eval_labels = labels[train_size:]
    
    # Train the model
    print("Training model...")
    nlp_model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        output_dir='models/intent_classifier'
    )
    
    # Test queries
    test_queries = [
        "I'm really excited to book a first-class flight to Paris next month!",
        "This hotel in London was terrible, the service was awful",
        "Can you recommend some good Italian restaurants near the Eiffel Tower?",
        "I'd love to visit the ancient temples in Kyoto",
        "How do I get from Heathrow Airport to central London?",
        "The subway system in Tokyo is so efficient and clean!",
        "Looking for a luxury beach resort in Maldives with spa services",
        "The tour guide in Rome was very knowledgeable and friendly",
    ]
    
    print("\nTesting model with various queries...\n")
    results = []
    
    for query in test_queries:
        print(f"Query: {query}")
        result = nlp_model.predict(query)
        
        print(f"Intent: {result['intent']} (Confidence: {result['confidence']:.2f})")
        print(f"Sentiment: {result['sentiment']}")
        if result['entities']:
            print("Entities:", result['entities'])
        print("-" * 80)
        
        results.append({
            "query": query,
            "analysis": result
        })
    
    # Save results
    with open("nlp_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\nTest results have been saved to nlp_test_results.json")

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Create necessary directories
    os.makedirs("models/intent_classifier", exist_ok=True)
    os.makedirs("models/sentiment_analyzer", exist_ok=True)
    
    test_model()
