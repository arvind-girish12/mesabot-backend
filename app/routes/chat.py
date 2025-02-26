from flask import Blueprint, request, jsonify
from app.services import query_documents, run_rag
import traceback

chat_bp = Blueprint("chat", __name__)

@chat_bp.route("/", methods=["POST"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight OK"}), 200
    """Handles user queries, retrieves relevant documents, formats prompt, and generates response."""
    data = request.json
    user_query = data.get("message")
    chat_history = data.get("memory", "")  # Get chat history from request

    if not user_query:
        return jsonify({"error": "Message is required"}), 400

    try:
        # Run RAG process with chat history context
        response = run_rag(user_query, chat_history)

        return jsonify({"response": response}), 200

    except Exception as e:
        print(traceback.format_exc())  # Debugging
        return jsonify({"error": str(e)}), 500
