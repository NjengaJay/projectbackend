from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from datetime import datetime
import uuid
import json
import os
from travel_assistant import TravelAssistant, TravelResponse
from travel_planner import TravelPreferences, RoutePreference

chat_bp = Blueprint('chat', __name__)

# Initialize travel assistant
travel_assistant = TravelAssistant()

# Store user sessions
user_sessions = {}

def get_or_create_session(user_id):
    """Get existing session or create new one for user"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'session_id': str(uuid.uuid4()),
            'created_at': datetime.utcnow(),
            'context': {}
        }
    return user_sessions[user_id]

@chat_bp.route('/api/chat', methods=['POST', 'OPTIONS'])
@jwt_required()
def chat():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400

        # Get or create user session
        session = get_or_create_session(user_id)
        
        # Process message through travel assistant
        user_message = data['message']
        travel_response = travel_assistant.process_query(user_message)
        
        # Extract route preferences if present
        preferences = None
        if 'preferences' in data:
            pref_data = data['preferences']
            preferences = TravelPreferences(
                route_preference=RoutePreference(pref_data.get('route_preference', 'time')),
                accessibility_required=pref_data.get('accessibility_required', False),
                max_cost=pref_data.get('max_cost'),
                scenic_priority=pref_data.get('scenic_priority', 0.0)
            )
        
        # Format response
        response_data = {
            'session_id': session['session_id'],
            'response': travel_response.to_dict(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Add quick replies if available
        if travel_response.intent in ['route_query', 'attraction_query']:
            response_data['quick_replies'] = [
                "Show me the scenic route",
                "What accessibility features are available?",
                "Tell me about attractions nearby",
                "What's the fastest route?",
                "What's the cheapest option?"
            ]
        
        # Update session context
        session['context'].update({
            'last_intent': travel_response.intent,
            'last_entities': travel_response.entities,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@chat_bp.route('/api/chat/history', methods=['GET'])
@jwt_required()
def get_chat_history():
    """Get chat history for current user"""
    try:
        user_id = get_jwt_identity()
        session = get_or_create_session(user_id)
        
        # In a real implementation, this would fetch from database
        # For now, return session context
        return jsonify({
            'session_id': session['session_id'],
            'context': session['context']
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
