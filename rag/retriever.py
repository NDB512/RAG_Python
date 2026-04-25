# rag/retriever.py
import os
from langchain_community.vectorstores import FAISS
from rag.ingest import get_embeddings


def load_store(store_path: str):
    embeddings = get_embeddings()
    return FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_docs(store_path: str, query: str, k: int = 6):
    index_file = os.path.join(store_path, "index.faiss")
    if not os.path.exists(index_file):
        print(f"[retriever] Store không tồn tại: {store_path}")
        return []

    try:
        vector_store = load_store(store_path)
        retriever    = vector_store.as_retriever(search_kwargs={"k": k})
        results      = retriever.invoke(query)
        return results
    except Exception as e:
        print(f"[retriever] Lỗi load store {store_path}: {e}")
        return []