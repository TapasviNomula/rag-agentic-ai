"""PDF loading, chunking, embedding and Pinecone indexing."""
import os
import time

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src import config


def download_pdf(path: str = config.PDF_PATH, url: str = config.PDF_URL) -> str:
    """Download the eBook if it is not already stored locally."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def ensure_index(pc: Pinecone, name: str, dim: int) -> None:
    """Create the Pinecone serverless index (cosine) if missing."""
    existing = [i["name"] for i in pc.list_indexes()]
    if name in existing:
        return
    pc.create_index(
        name=name,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)


def run_ingestion(pdf_path: str = config.PDF_PATH, index_name: str = config.PINECONE_INDEX_NAME):
    pdf_path = download_pdf(pdf_path)

    # 1. Load pages
    docs = PyPDFLoader(pdf_path).load()
    for d in docs:
        d.metadata["page"] = int(d.metadata.get("page", 0)) + 1  # 1-based page number
        d.metadata["source"] = "Ebook-Agentic-AI.pdf"

    # 2. Chunk (metadata such as page number is inherited by each chunk)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]

    # 3. Embed + upsert into Pinecone (chunk text is stored as metadata "text")
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    embeddings = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    dim = len(embeddings.embed_query("dimension check"))
    ensure_index(pc, index_name, dim)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    # Small batches + pause to stay inside the free-tier rate limits.
    batch = 20
    for i in range(0, len(chunks), batch):
        vectorstore.add_documents(chunks[i : i + batch])
        time.sleep(2)
    print(f"Ingested {len(docs)} pages -> {len(chunks)} chunks into '{index_name}'.")
    return len(chunks)


if __name__ == "__main__":
    run_ingestion()
