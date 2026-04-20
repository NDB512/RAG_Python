import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.chunking import build_documents_from_pages

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def ingest_document(file_path: str, store_path: str, doc_type: str = "user"):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    legal_chunks = build_documents_from_pages(docs, doc_type=doc_type)

    embeddings = get_embeddings()

    vector_store = FAISS.from_documents(legal_chunks, embeddings)
    vector_store.save_local(store_path)

    print(f" Ingested {len(legal_chunks)} chunks -> {store_path}")
    return store_path

def ingest_folder(folder_path: str, store_path: str, doc_type: str = "legal"):
    embeddings = get_embeddings()
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            chunks = build_documents_from_pages(docs, doc_type=doc_type)
            all_docs.extend(chunks)

    if not all_docs:
        raise ValueError(f"Không tìm thấy PDF trong folder: {folder_path}")

    vector_store = FAISS.from_documents(all_docs, embeddings)
    vector_store.save_local(store_path)

    print(f" Ingested {len(all_docs)} chunks from folder -> {store_path}")
    return store_path