import uuid
import os
import re
from flask import Blueprint, request, jsonify
from app.services import upsert_document, process_tabular_data
from langchain.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import json
from app.services.vectorstore import index

admin_bp = Blueprint("admin", __name__)

@admin_bp.route("/", methods=["POST"])
def admin():
    """Handles document storage from text input and file uploads (FormData)."""
    subject = request.form.get("subject")
    text = request.form.get("text")
    file = request.files.get("file")

    if not (text or file):
        return jsonify({"error": "Either text or file is required"}), 400

    try:
        doc_id = str(uuid.uuid4())
        
        # Process file if provided
        if file:
            filename = file.filename
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.pdf':
                # Save PDF temporarily
                temp_path = f"/tmp/{filename}"
                file.save(temp_path)
                
                # Load and split PDF
                loader = PyPDFLoader(temp_path)
                documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=60,
                    chunk_overlap=20
                )
                chunks = text_splitter.split_documents(documents)
                
                # Store each non-empty chunk
                for i, chunk in enumerate(chunks):
                    if chunk.page_content.strip():  # Only store if content is not empty
                        chunk_id = f"{doc_id}_chunk_{i}"
                        upsert_document(
                            chunk_id, 
                            chunk.page_content,
                            {
                                "subject": subject,
                                "source": "pdf",
                                "parent_id": doc_id,
                                "chunk_index": i
                            }
                        )
                
                os.remove(temp_path)
                
            elif file_ext in ['.csv', '.xlsx']:
                # Save file temporarily
                temp_path = f"/tmp/{filename}"
                file.save(temp_path)
                
                if file_ext == '.xlsx':
                    # Load Excel file
                    xls = pd.ExcelFile(temp_path)
                    
                    # Process each sheet
                    for sheet_name in xls.sheet_names:
                        df = xls.parse(sheet_name)
                        # Clean empty rows and columns first
                        df = df.dropna(how='all', axis=0)  # Drop empty rows
                        df = df.dropna(how='all', axis=1)  # Drop empty columns
                        process_tabular_data(df, doc_id, subject, "xlsx", sheet_name)
                else:
                    # Load CSV file
                    df = pd.read_csv(temp_path)
                    # Clean empty rows and columns first
                    df = df.dropna(how='all', axis=0)  # Drop empty rows
                    df = df.dropna(how='all', axis=1)  # Drop empty columns
                    process_tabular_data(df, doc_id, subject, "csv")
                
                os.remove(temp_path)

        # Process text if provided
        if text:
            processed_text = f"{subject}\n\n{text}"
            if processed_text.strip():  # Only store if content is not empty
                text_doc_id = f"{doc_id}_text" if file else doc_id
                upsert_document(
                    text_doc_id,
                    processed_text,
                    {"subject": subject, "source": "text_input"}
                )

        return jsonify({"message": "Data stored successfully", "doc_id": doc_id}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@admin_bp.route("/whatsapp", methods=["POST"])
def process_whatsapp():
    """Process WhatsApp chat export text file."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
        
    file = request.files['file']
    if file.filename == '' or not file.filename.endswith('.txt'):
        return jsonify({"error": "Invalid file format. Please upload a .txt file"}), 400

    try:
        # Read the file content
        content = file.read().decode('utf-8')
        
        # Store the timestamp pattern used for splitting
        timestamp_pattern = r'\[\d{2}/\d{2}/\d{2},\s\d{2}:\d{2}:\d{2}\s[AP]M\]'
        messages = re.split(timestamp_pattern, content)
        timestamps = re.findall(timestamp_pattern, content)
        
        doc_id = str(uuid.uuid4())
        
        # Process each message with its timestamp
        for i, (message, timestamp) in enumerate(zip(messages[1:], timestamps)):
            if message.strip():  # Skip empty messages
                chunk_id = f"{doc_id}_msg_{i}"
                full_message = f"{timestamp}{message}\n\n"
                
                metadata = {
                    "source": "whatsapp",
                    "parent_id": doc_id,
                    "chunk_index": i,
                    "timestamp": timestamp,
                    "timestamp_pattern": timestamp_pattern
                }
                
                upsert_document(
                    chunk_id,
                    full_message,
                    metadata
                )
                
        return jsonify({
            "message": "WhatsApp chat processed successfully",
            "doc_id": doc_id
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@admin_bp.route("/clear", methods=["DELETE"])
def clear_vectorstore():
    """Deletes all vectors from the vectorstore."""
    try:
        # Delete all vectors from the index
        index.delete(delete_all=True)
        return jsonify({"message": "Vectorstore cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
