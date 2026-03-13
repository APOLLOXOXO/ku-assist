from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "documents")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Model settings
OLLAMA_MODEL = "phi3:mini"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Retrieval settings
TOP_K_RESULTS = 3