"""
Train and test the Travel Assistant NLP model
"""
import os
from sklearn.model_selection import train_test_split
from .model import TravelAssistantNLP
from .sample_data import generate_sample_data

def train_model():
    # Create model directory if it doesn't exist
    model_dir = "travel_assistant_model"
    os.makedirs(model_dir, exist_ok=True)

    # Initialize model
    nlp_model = TravelAssistantNLP()

    # Get sample data
    texts, labels = generate_sample_data()

    # Split data
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # Train model
    nlp_model.train(
        train_texts=train_texts,
        train_labels=train_labels,
        eval_texts=eval_texts,
        eval_labels=eval_labels,
        output_dir=model_dir
    )

    # Test some sample queries
    test_queries = [
        "I want to visit the Louvre museum",
        "Find me a flight to Miami next week",
        "What's a good restaurant in Tokyo?",
        "How can I get from JFK to Manhattan?"
    ]

    print("\nTesting model with sample queries:")
    for query in test_queries:
        result = nlp_model.predict(query)
        print(f"\nQuery: {query}")
        print(f"Predicted Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.2f}")
        if result['entities']:
            print("Detected Entities:", result['entities'])

if __name__ == "__main__":
    train_model()
