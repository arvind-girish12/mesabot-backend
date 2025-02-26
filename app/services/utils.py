import json
from .vectorstore import upsert_document
from .rag import classify_data_structure

def process_tabular_data(df, doc_id, subject, source_prefix, sheet_name=None):
    """Process tabular data from CSV or Excel files."""
    # Set column names from first row and remove that row
    df.columns = df.columns.astype(str)  # Convert column names to strings
    df = df.iloc[1:].reset_index(drop=True)
    
    # Take first 10 rows if more than 10 rows exist
    sample_df = df.head(10) if len(df) > 10 else df
        
    # Convert sample to string for classification
    sample_str = sample_df.to_string()
    
    # Classify data structure
    data_type = classify_data_structure(sample_str)
    
    if data_type == 'STRUCTURED':
        # Process as structured data
        records = df.to_dict(orient='records')
        for i, record in enumerate(records):
            if record:  # Skip empty records
                chunk_id = f"{doc_id}_{sheet_name}_chunk_{i}" if sheet_name else f"{doc_id}_chunk_{i}"
                metadata = {
                    "subject": subject,
                    "source": f"{source_prefix}_structured",
                    "parent_id": doc_id,
                    "chunk_index": i
                }
                if sheet_name:
                    record["sheet_name"] = sheet_name
                upsert_document(
                    chunk_id,
                    json.dumps(record),
                    metadata
                )
    else:
        # Process as unstructured data
        text_content = df.to_string()
        if text_content.strip():
            chunk_id = f"{doc_id}_{sheet_name}" if sheet_name else f"{doc_id}_text"
            if sheet_name:
                text_content = f"Sheet: {sheet_name}\n\n{text_content}"
            metadata = {
                "subject": subject,
                "source": f"{source_prefix}_unstructured",
                "parent_id": doc_id
            }
            upsert_document(
                chunk_id,
                text_content,
                metadata
            )