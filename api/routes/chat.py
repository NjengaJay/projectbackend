from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from flask_cors import cross_origin
import logging
import traceback
from ..chatbot_handler import ChatbotHandler

logger = logging.getLogger(__name__)
chat_bp = Blueprint('chat', __name__)

chatbot = ChatbotHandler()

@chat_bp.route('', methods=['POST', 'OPTIONS'])  
@cross_origin(origins=["http://localhost:3000"], 
             methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             allow_headers=["Content-Type", "Authorization"])
def chat():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    try:
        # Only require JWT for non-OPTIONS requests
        if request.method != 'OPTIONS':
            token = request.headers.get('Authorization')
            if not token:
                return jsonify({'error': 'No authorization token provided'}), 401

        data = request.get_json()
        message = data.get('message')
        preferences = data.get('preferences', {})
        user_id = 'test_user'  # For now, use a test user ID

        logger.info(f"Received message: {message}")
        logger.info(f"With preferences: {preferences}")

        if not message:
            return jsonify({
                'error': 'Message is required'
            }), 400

        # Process message using chatbot handler
        result = chatbot.process_message(message, {'preferences': preferences})
        logger.info(f"Generated response: {result}")

        return jsonify({
            'response': result['response']
        }), 200

    except Exception as e:
        error_msg = f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return jsonify({
            'response': {
                'text': "I apologize, but I'm having trouble processing your request. Please try again.",
                'error': str(e)
            }
        }), 500
