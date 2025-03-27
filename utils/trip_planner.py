from datetime import datetime, timedelta
import random
from typing import Dict, List
import numpy as np
from transformers import pipeline
from .recommendation import RecommendationEngine
from models.models import Destination
from models.trip_plan import TripPlan

class TripPlanner:
    def __init__(self):
        self.recommendation_engine = RecommendationEngine()
        self.text_generator = pipeline("text-generation", model="gpt2")
        
    def generate_itinerary(self, trip_plan: TripPlan, destinations: List[Destination]) -> Dict:
        """Generate a complete itinerary based on trip plan parameters"""
        # Train recommendation engine if needed
        self.recommendation_engine.train_clusters(destinations)
        
        # Calculate trip duration
        duration = (trip_plan.end_date - trip_plan.start_date).days + 1
        daily_budget = trip_plan.budget / duration
        
        itinerary = {
            'summary': self._generate_trip_summary(trip_plan),
            'daily_plans': self._generate_daily_plans(
                trip_plan,
                destinations,
                duration,
                daily_budget
            )
        }
        
        return itinerary
    
    def _generate_trip_summary(self, trip_plan: TripPlan) -> str:
        """Generate a natural language summary of the trip"""
        prompt = f"A {(trip_plan.end_date - trip_plan.start_date).days + 1}-day trip to {trip_plan.destination}"
        summary = self.text_generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
        return summary
    
    def _generate_daily_plans(
        self,
        trip_plan: TripPlan,
        destinations: List[Destination],
        duration: int,
        daily_budget: float
    ) -> List[Dict]:
        """Generate detailed plans for each day of the trip"""
        daily_plans = []
        current_date = trip_plan.start_date
        
        for day in range(duration):
            # Get recommended attractions for the day
            recommended_places = self.recommendation_engine.get_similar_destinations(
                destinations[0].id,  # Use first destination as reference
                n_recommendations=3
            )
            
            # Generate meal suggestions
            meals = self._generate_meal_suggestions(trip_plan.preferences)
            
            daily_plan = {
                'date': current_date.strftime('%Y-%m-%d'),
                'schedule': [
                    {
                        'time': '09:00',
                        'activity': 'Breakfast',
                        'location': meals['breakfast'],
                        'estimated_cost': 20.0
                    },
                    {
                        'time': '10:00',
                        'activity': 'Morning Activity',
                        'location': recommended_places[0]['name'],
                        'estimated_cost': 30.0
                    },
                    {
                        'time': '13:00',
                        'activity': 'Lunch',
                        'location': meals['lunch'],
                        'estimated_cost': 25.0
                    },
                    {
                        'time': '14:30',
                        'activity': 'Afternoon Activity',
                        'location': recommended_places[1]['name'],
                        'estimated_cost': 35.0
                    },
                    {
                        'time': '19:00',
                        'activity': 'Dinner',
                        'location': meals['dinner'],
                        'estimated_cost': 40.0
                    }
                ],
                'daily_total': 150.0,  # Sum of estimated costs
                'accessibility_notes': self._generate_accessibility_notes(
                    trip_plan.accessibility_requirements,
                    recommended_places
                )
            }
            
            daily_plans.append(daily_plan)
            current_date += timedelta(days=1)
        
        return daily_plans
    
    def _generate_meal_suggestions(self, preferences: Dict) -> Dict:
        """Generate meal suggestions based on user preferences"""
        # This would ideally use a more sophisticated recommendation system
        return {
            'breakfast': 'Local Cafe',
            'lunch': 'Traditional Restaurant',
            'dinner': 'Gourmet Dining'
        }
    
    def _generate_accessibility_notes(
        self,
        requirements: Dict,
        places: List[Dict]
    ) -> List[str]:
        """Generate accessibility notes for recommended places"""
        if not requirements:
            return []
            
        notes = []
        for place in places:
            if 'accessibility_features' in place:
                notes.append(f"{place['name']}: {', '.join(place['accessibility_features'])}")
        return notes
