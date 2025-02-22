from langchain_huggingface import HuggingFaceEmbeddings
# Initialize Hugging Face embeddings model (No API key required)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def get_nvidia_embedding(text):
    """Generate an embedding for the input text using NVIDIA's LLaMA model."""
    return embedding_model
