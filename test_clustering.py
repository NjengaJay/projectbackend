import pandas as pd
from tourist_clustering import TouristClusteringModel

def test_clustering():
    # Initialize and load the trained model
    model = TouristClusteringModel()
    model.load_model()
    
    # Test cases representing different tourist profiles
    test_cases = [
        {
            'Age': 25,
            'Preferred Tour Duration': 3,
            'Tour Duration': 2,
            'Tourist Rating': 4.5,
            'System Response Time': 2.5,
            'Recommendation Accuracy': 95,
            'VR Experience Quality': 4.8,
            'Satisfaction': 5,
            'Accessibility': 1,
            'Site_Name': 0  # Assuming 0 represents a specific site
        },
        {
            'Age': 65,
            'Preferred Tour Duration': 7,
            'Tour Duration': 6,
            'Tourist Rating': 3.0,
            'System Response Time': 3.5,
            'Recommendation Accuracy': 85,
            'VR Experience Quality': 4.0,
            'Satisfaction': 3,
            'Accessibility': 0,
            'Site_Name': 2  # Assuming 2 represents a different site
        }
    ]
    
    # Test predictions
    for i, case in enumerate(test_cases):
        prediction = model.predict_cluster(case)
        print(f"\nTest Case {i+1}:")
        print(f"Tourist Profile: {case}")
        print(f"Assigned Cluster: {prediction['cluster']}")
        print(f"Confidence: {prediction['confidence']:.2f}")

if __name__ == "__main__":
    test_clustering()
