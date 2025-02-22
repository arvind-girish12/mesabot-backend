import uuid
from flask import Blueprint, request, jsonify
from app.services import upsert_document

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/", methods=["POST"])
def admin():
    """Handles document storage from text input (FormData)."""
    subject = request.form.get("subject")
    text = request.form.get("text")

    if not text:
        return jsonify({"error": "Subject and text are required"}), 400

    # Append subject and text with a line break
    processed_text = f"{subject}\n\n{text}"

    try:
        # Generate a unique ID for the document
        doc_id = str(uuid.uuid4())

        # Store in Pinecone
        upsert_document(doc_id, processed_text, {"subject": subject, "source": "text_input"})

        return jsonify({"message": "Data stored successfully", "doc_id": doc_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
