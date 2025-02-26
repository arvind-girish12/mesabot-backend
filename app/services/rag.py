import requests
from app.config import Config
from app.services.vectorstore import query_documents, get_recent_documents
from app.services.embeddings import get_nvidia_embedding
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# Initialize LangChain's Gemini LLM model
llm = ChatGoogleGenerativeAI(model=Config.GEMINI_MODEL, google_api_key=Config.GOOGLE_API_KEY)

# Define a structured prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        "You are an AI assistant that provides answers based on the given context.\n"
        "If the context is empty or irrelevant, say 'I don't know'.\n\n"
        "Context:\n{context}\n\n"
        "Chat History:\n{chat_history}\n\n"
        "User Question:\n{query}\n\n"
        "Please provide your response in markdown format.\n"
        "Concise, informative response:"
    ),
)

# Define prompt template for generating questions
questions_prompt_template = PromptTemplate(
    input_variables=["documents"],
    template=(
        "Based on these documents, generate 3 relevant questions that could be asked:\n\n"
        "Documents:\n{documents}\n\n"
        "Generate exactly 3 questions in a list format."
    )
)

# Define prompt template for data structure classification
data_structure_prompt = PromptTemplate(
    input_variables=["text"],
    template=(
        "Analyze this data and determine if it represents STRUCTURED or UNSTRUCTURED data.\n"
        "STRUCTURED data must have:\n"
        "- All column headers appear together in one section at the start\n"
        "- Data rows follow the column headers\n"
        "- No additional column headers or instructional text mixed in with data\n\n"
        "UNSTRUCTURED data includes:\n"
        "- Multiple sections of column headers\n"
        "- Instructional or descriptive text mixed in\n"
        "- Any non-tabular content\n\n"
        "Data:\n{text}\n\n"
        "Respond with only one word - either 'STRUCTURED' or 'UNSTRUCTURED'."
    )
)

def run_rag(query, chat_history= ""):
    """Retrieve relevant documents from Pinecone and generate a response using Gemini's LLM."""
    # Convert query into an embedding using NVIDIA
    query_embedding = get_nvidia_embedding(query)

    # Retrieve relevant documents from Pinecone
    retrieved_docs = query_documents(query, top_k=3)
    context = " ".join([doc.page_content for doc in retrieved_docs])

    # Send the context and query to Gemini's LLM
    return generate_response(query, context, chat_history)

def generate_response(query, context, chat_history):
    """Generate a response using LangChain's ChatGoogleGenerativeAI model with a structured prompt."""
    
    # Format the prompt using LangChain's template
    formatted_prompt = prompt_template.format(context=context, query=query, chat_history=chat_history)

    # Invoke LLM with the formatted prompt
    response = llm.invoke(formatted_prompt)
    
    return response.content

def generate_questions_from_recent():
    """Generate 3 questions based on the most recently added documents."""
    # Get the 3 most recent documents
    recent_docs = get_recent_documents(limit=3)
    
    # Combine document contents
    docs_content = "\n\n".join([doc.page_content for doc in recent_docs])
    
    # Format prompt with documents
    formatted_prompt = questions_prompt_template.format(documents=docs_content)
    
    # Generate questions using LLM
    response = llm.invoke(formatted_prompt)
    
    return response.content

def classify_data_structure(text):
    """Classify if data represents structured or unstructured format."""
    # Format the prompt with the input text
    formatted_prompt = data_structure_prompt.format(text=text)
    
    # Get classification from LLM
    response = llm.invoke(formatted_prompt)
    
    return response.content.strip()
