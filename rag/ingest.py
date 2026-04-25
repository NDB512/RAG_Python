import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.chunking import build_documents_from_pages

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def ingest_document(file_path: str, store_path: str, doc_type: str = "user"):
    """
    Ingest một file PDF vào vector store.
    Nếu store đã tồn tại -> MERGE (không ghi đè) để hỗ trợ nhiều file.
    """
    loader = PyPDFLoader(file_path)
    docs   = loader.load()

    if not docs:
        raise ValueError(f"Không đọc được nội dung từ file: {file_path}")

    chunks = build_documents_from_pages(docs, doc_type=doc_type)

    if not chunks:
        raise ValueError(f"Không tách được chunk từ file: {file_path}")

    embeddings = get_embeddings()

    # Merge vào store hiện có (nếu có) 
    index_file = os.path.join(store_path, "index.faiss")
    if os.path.exists(index_file):
        try:
            existing = FAISS.load_local(
                store_path, embeddings, allow_dangerous_deserialization=True
            )
            new_store = FAISS.from_documents(chunks, embeddings)
            existing.merge_from(new_store)
            existing.save_local(store_path)
            print(f"[ingest] Merged {len(chunks)} chunks vào store -> {store_path}")
        except Exception as e:
            print(f"[ingest] Không merge được ({e}), tạo store mới.")
            _save_new(chunks, embeddings, store_path)
    else:
        _save_new(chunks, embeddings, store_path)

    return store_path


def _save_new(chunks, embeddings, store_path):
    os.makedirs(store_path, exist_ok=True)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(store_path)
    print(f"[ingest] Created store with {len(chunks)} chunks -> {store_path}")


def ingest_folder(folder_path: str, store_path: str, doc_type: str = "legal"):
    """
    Ingest toàn bộ PDF trong một folder vào vector store.
    Ghi đè store cũ (dùng cho legal_docs load batch).
    """
    embeddings = get_embeddings()
    all_docs   = []

    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise ValueError(f"Không tìm thấy PDF trong folder: {folder_path}")

    for filename in pdf_files:
        file_path = os.path.join(folder_path, filename)
        try:
            loader = PyPDFLoader(file_path)
            docs   = loader.load()
            chunks = build_documents_from_pages(docs, doc_type=doc_type)
            all_docs.extend(chunks)
            print(f"[ingest_folder] {filename}: {len(chunks)} chunks")
        except Exception as e:
            print(f"[ingest_folder] Bỏ qua {filename}: {e}")

    if not all_docs:
        raise ValueError("Không có chunk nào được tạo từ folder.")

    os.makedirs(store_path, exist_ok=True)
    vector_store = FAISS.from_documents(all_docs, embeddings)
    vector_store.save_local(store_path)
    print(f"[ingest_folder] Total {len(all_docs)} chunks -> {store_path}")
    return store_path