import os
from dotenv import load_dotenv

# Explicitly load the latest .env file
load_dotenv(override=True)

class Config:
    """Configuration class for storing API keys and settings."""

    # NVIDIA API for generating embeddings
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    NVIDIA_MODEL = "NV-Embed-QA"  # Change if using a different NVIDIA embedding model
    NVIDIA_EMBEDDINGS_URL = "https://api.nvidia.com/llama-embed"

    # Pinecone configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX = os.getenv("PINECONE_INDEX")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
    PINECONE_API_URL = os.getenv("PINECONE_API_URL")

    # Google Gemini configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL = "gemini-1.5-pro"  # Google's Gemini Pro model

    @staticmethod
    def check_config():
        """Ensure all required environment variables are set."""
        required_vars = [
            "NVIDIA_API_KEY", "PINECONE_API_KEY", "PINECONE_ENVIRONMENT", "PINECONE_INDEX", "GOOGLE_API_KEY"
        ]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Validate configuration on startup
Config.check_config()
