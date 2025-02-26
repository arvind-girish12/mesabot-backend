from flask import Blueprint, jsonify
from app.services import generate_questions_from_recent

prompt_bp = Blueprint("prompt", __name__)

@prompt_bp.route("/prompts", methods=["GET"])
def get_prompts():
    """Retrieves 3 suggested questions based on recently added documents."""
    try:
        questions = generate_questions_from_recent()
        return jsonify({"prompts": questions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
