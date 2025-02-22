import requests
from app.config import Config
from app.services.vectorstore import query_documents
from app.services.embeddings import get_nvidia_embedding
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Initialize LangChain's Groq LLM model
llm = ChatGroq(model=Config.GROQ_MODEL, groq_api_key=Config.GROQ_API_KEY)

# Define a structured prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are an AI assistant that provides answers based on the given context.\n"
        "If the context is empty or irrelevant, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "User Question:\n{query}\n\n"
        "Concise, informative response:"
    ),
)

def run_rag(query):
    """Retrieve relevant documents from Pinecone and generate a response using Groq's LLM."""
    # Convert query into an embedding using NVIDIA
    query_embedding = get_nvidia_embedding(query)

    # Retrieve relevant documents from Pinecone
    retrieved_docs = query_documents(query, top_k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Send the context and query to Groq's LLM
    return generate_response(query, context)

def generate_response(query, context):
    """Generate a response using LangChain's ChatGroq model with a structured prompt."""
    
    # Format the prompt using LangChain's template
    formatted_prompt = prompt_template.format(context=context, query=query)

    # Invoke LLM with the formatted prompt
    response = llm.invoke(formatted_prompt)
    
    return response.content
