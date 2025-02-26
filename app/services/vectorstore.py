from langchain.vectorstores import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from app.config import Config
from langchain_pinecone import PineconeVectorStore
# Initialize Hugging Face embeddings model (No API key required)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Pinecone client
pc = Pinecone(api_key=Config.PINECONE_API_KEY)
index = pc.Index(Config.PINECONE_INDEX)

# Create LangChain-compatible vector store
vector_store = PineconeVectorStore(embedding=embedding_model, index=index)

def upsert_document(doc_id, text, metadata={}):
    """Generate an embedding for the given text and store it in Pinecone."""
    vector_store.add_texts(texts=[text], metadatas=[metadata], ids=[doc_id])

def query_documents(query, top_k=3):
    """Retrieve the top_k most similar documents from Pinecone using LangChain."""
    results = vector_store.similarity_search(query, k=top_k)
    return results

def get_recent_documents(limit=3):
    """Retrieve the most recently added documents from Pinecone."""
    # Fetch all vectors and sort by timestamp in metadata
    results = index.query(
        vector=[0] * embedding_model.embedding_dimension,  # Dummy vector
        top_k=100,  # Fetch enough to find recent ones
        include_metadata=True
    )
    
    # Sort by timestamp if available in metadata
    sorted_results = sorted(
        results.matches,
        key=lambda x: x.metadata.get('timestamp', 0),
        reverse=True
    )
    
    # Convert to LangChain document format
    documents = []
    for match in sorted_results[:limit]:
        doc = vector_store.similarity_search(
            match.id,
            k=1
        )[0]
        documents.append(doc)
        
    return documents
