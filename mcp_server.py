# my_server.py
from fastmcp import FastMCP
import asyncio # We'll need this later for the client

# Instantiate the server, giving it a name
mcp = FastMCP(name="My First MCP Server")

print("FastMCP server object created.")

@mcp.tool()
def greet(name: str) -> str:
    """Returns a simple greeting."""
    return f"Hello, {name}!"

@mcp.tool()
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b

print("Tools 'greet' and 'add' added.")

APP_CONFIG = {"theme": "dark", "version": "1.1", "feature_flags": ["new_dashboard"]}

@mcp.resource("data://config")
def get_config() -> dict:
    """Provides the application configuration."""
    return APP_CONFIG

print("Resource 'data://config' added.")

USER_PROFILES = {
    101: {"name": "Alice", "status": "active"},
    102: {"name": "Bob", "status": "inactive"},
}

@mcp.resource("users://{user_id}/profile")
def get_user_profile(user_id: int) -> dict:
    """Retrieves a user's profile by their ID."""
    # The {user_id} from the URI is automatically passed as an argument
    return USER_PROFILES.get(user_id, {"error": "User not found"})

print("Resource template 'users://{user_id}/profile' added.")

@mcp.prompt("summarize")
async def summarize_prompt(text: str) -> list[dict]:
    """Generates a prompt to summarize the provided text."""
    return [
        {"role": "system", "content": "You are a helpful assistant skilled at summarization."},
        {"role": "user", "content": f"Please summarize the following text:\n\n{text}"}
    ]

print("Prompt 'summarize' added.")


    
# Import necessary libraries for NLP tasks
from textblob import TextBlob
from transformers import pipeline

# Sentiment Analysis Tool
@mcp.tool()
def analyze_sentiment(text: str) -> dict:
    """Analyzes the sentiment of the given text."""
    blob = TextBlob(text)
    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity
    }

# Text Summarization Tool removed
print("Tool 'analyze_sentiment' added.")
# Import necessary libraries for RAG tool
import requests
import openai
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import os

load_dotenv()  # Load variables from .env
# Set your Azure OpenAI endpoint and API key
endpoint = os.getenv("ENDPOINT")  # Azure OpenAI endpoint
model_name = os.getenv("model_name")  # Model name
deployment = os.getenv("deployment")  # Deployment name
subscription_key = os.getenv("SECRET_KEY") 
api_version = os.getenv("api_version")  # API version
# RAG Tool
print("Tool 'build_website_vector_db' is being invoked...")
@mcp.tool()
def build_website_vector_db(url: str) -> str:
    """Scrapes a website, creates a vector database, and returns the path to the database."""
    try:
        print(f"Building vector database for URL: {url}")
        # Scrape the website
        try:
            print(f"Scraping vector database for URL: {url}")
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            print(f"HTTP Status Code: {response.status_code}")
            print(f"Response Headers: {response.headers}")
            print(f"Fetched Content Length: {len(response.text)}")
        except requests.exceptions.RequestException as e:
            return f"Error fetching website content: {e}"
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()

        # Split the text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        print("Splitting text into chunks...")
        texts = text_splitter.split_text(text)
        print(f"Number of text chunks created: {len(texts)}")
        print(f"Sample text chunk: {texts[0] if texts else 'No text chunks available'}")

        # Create embeddings and vector database
        print("Creating embeddings...")
        print("Inspecting embeddings object...")
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("deployment"),
            api_key=os.getenv("SECRET_KEY"),
            azure_endpoint=os.getenv("ENDPOINT"),
            openai_api_version=os.getenv("api_version"),
            model="text-embedding-3-small"  # Explicit model name
        )
        print(f"Embeddings object type: {type(embeddings)}")
        print("Embeddings created successfully.")
        test_embedding = embeddings.embed_query("test")
        print("Creating vector database...")
        
        print(len(test_embedding))  # Should output a number (e.g., 1536 for text-embedding-ada-002)
        vector_db = FAISS.from_texts(texts, embeddings)
        print("Vector database created successfully.")

        # Save the vector database
        db_path = f"{url.replace('https://', '').replace('http://', '')
                     .replace('/', '').replace('www','').replace('.','')}_vector_db"
        vector_db.save_local(db_path)

        return f"Vector database created and saved at: {db_path}"
    except Exception as e:
        return f"Error building vector database: {e}"

print("Tool 'build_website_vector_db' added.")
# print(f"Registered tools: {mcp.list_tools()}")

# Tool to query the vector database
@mcp.tool()
def query_vector_db(prompt: str, db_path: str) -> str:
    """Queries the vector database and returns the most relevant information."""
    try:
        print(f"Loading vector database from: {db_path}")
        vector_db = FAISS.load_local(
            db_path,
            AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("deployment"),
                api_key=os.getenv("SECRET_KEY"),
                azure_endpoint=os.getenv("ENDPOINT"),
                openai_api_version=os.getenv("api_version"),
                model="text-embedding-3-small"  # Explicit model name
            ),
            allow_dangerous_deserialization=True
        )
        print("Vector database loaded successfully.")
        
        # Perform the query
        print(f"Querying vector database with prompt: {prompt}")
        results = vector_db.similarity_search(prompt, k=3)
        return "\n".join([result.page_content for result in results])
    except Exception as e:
        return f"Error querying vector database: {e}"

print("Tool 'query_vector_db' added.")

if __name__ == "__main__":
    print("\n--- Starting FastMCP Server via __main__ ---")
    # This starts the server, typically using the stdio transport by default
    mcp.run()