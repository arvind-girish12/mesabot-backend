from flask import Blueprint, jsonify
from datetime import datetime
from app.services import query_documents, run_rag

deadline_bp = Blueprint("deadline", __name__)

@deadline_bp.route("/", methods=["GET"])
def deadline():
    """Fetches all submission deadlines for today and generates a response."""
    try:
        # Define the query to match relevant deadlines
        query = "Submissions due today OR documents containing 'submit by', 'upload by', 'deadline', 'final date'"

        # Retrieve relevant documents from Pinecone
        retrieved_docs = query_documents(query, top_k=5)

        # Filter documents that contain deadline-related words
        filtered_docs = [
            doc.page_content for doc in retrieved_docs
            if any(keyword in doc.page_content.lower() for keyword in ["submit by", "upload by", "deadline", "final date"])
        ]

        # If no relevant docs found, return a message
        if not filtered_docs:
            return jsonify({"response": "No submissions are due today."}), 200

        # Combine relevant document text
        context = "\n\n".join(filtered_docs)

        # Generate a response using LLM
        response = run_rag(f"List all submissions due today. Here is the context:\n{context}")

        return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
