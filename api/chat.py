from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required
from datetime import datetime

chat_bp = Blueprint('chat', __name__)

# Mock responses for different types of queries
MOCK_RESPONSES = {
    'default': "I'm here to help you with your travel plans! You can ask me about destinations, trip planning, or travel tips.",
    'greeting': "Hello! How can I assist you with your travel plans today?",
    'destination': "I can help you find great destinations based on your preferences. What type of trip are you interested in?",
    'budget': "I can suggest destinations that match your budget. What's your approximate budget range?",
    'weather': "The weather varies by destination and season. Which destination are you interested in?",
    'activities': "There are many activities available depending on the destination. What types of activities do you enjoy?",
}

def get_mock_response(message):
    message = message.lower()
    if any(word in message for word in ['hi', 'hello', 'hey']):
        return MOCK_RESPONSES['greeting']
    elif any(word in message for word in ['destination', 'place', 'where']):
        return MOCK_RESPONSES['destination']
    elif any(word in message for word in ['budget', 'cost', 'price']):
        return MOCK_RESPONSES['budget']
    elif any(word in message for word in ['weather', 'climate']):
        return MOCK_RESPONSES['weather']
    elif any(word in message for word in ['activity', 'activities', 'do']):
        return MOCK_RESPONSES['activities']
    return MOCK_RESPONSES['default']

@chat_bp.route('/api/chat', methods=['POST', 'OPTIONS'])
@jwt_required()
def chat():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400

        user_message = data['message']
        response = get_mock_response(user_message)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
