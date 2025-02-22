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
