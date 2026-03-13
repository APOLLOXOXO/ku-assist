import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from config import DOCUMENTS_DIR, CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


def load_documents():
    documents = []

    for filename in os.listdir(DOCUMENTS_DIR):

        if filename.endswith(".pdf"):

            filepath = os.path.join(DOCUMENTS_DIR, filename)

            print(f"Loading: {filename}")

            loader = PyPDFLoader(filepath)

            documents.extend(loader.load())

    return documents


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks from {len(documents)} pages")

    return chunks


def create_vector_store(chunks):

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Creating vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

    print("Vector store created and saved.")

    return vectorstore


# ⭐ THIS IS WHAT FLASK WILL CALL
def ingest_documents():

    print("--- KU-Assist Ingestion Pipeline ---")

    docs = load_documents()

    print(f"Loaded {len(docs)} pages total")

    if len(docs) == 0:
        print("No PDF documents found.")
        return

    chunks = split_documents(docs)

    create_vector_store(chunks)

    print("--- Ingestion Complete ---")


# Allows running manually from terminal
if __name__ == "__main__":
    ingest_documents()