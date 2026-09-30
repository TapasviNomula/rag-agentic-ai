"""Environment setup & constants."""
import os
from dotenv import load_dotenv

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# If the best retrieved chunk is less similar than this, the question is
# treated as out-of-scope and the bot refuses (groundedness guard).
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.30"))

PDF_PATH = "data/Ebook-Agentic-AI.pdf"
PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"

REFUSAL = "I cannot answer based on the provided document"
