import os
from langchain_community.vectorstores import FAISS
from rag.ingest import get_embeddings

# Các hàm liên quan đến truy vấn và reranking tài liệu từ vector store
def load_store(store_path: str):
    embeddings = get_embeddings()
    return FAISS.load_local(
        store_path, embeddings, allow_dangerous_deserialization=True
    )

# Hàm chính để truy vấn và rerank tài liệu
def retrieve_docs(store_path: str, query: str, k: int = 8):
    index_file = os.path.join(store_path, "index.faiss")
    if not os.path.exists(index_file):
        print(f"[retriever] Store không tồn tại: {store_path}")
        return []

    try:
        vector_store = load_store(store_path)
        results = vector_store.similarity_search(query, k=k*2)   # lấy nhiều hơn để rerank

        query_lower = query.lower()
        scored = []

        for doc in results:
            content_lower = doc.page_content.lower()
            meta = doc.metadata
            score = 1.0

            # Reranking theo ngữ nghĩa
            if any(x in query_lower for x in ["nghĩa vụ", "quyền", "trách nhiệm", "phải", "cam kết"]):
                if "quyền và nghĩa vụ" in content_lower or "điều 4" in content_lower:
                    score += 1.5
                if any(x in content_lower for x in ["bên cho thuê", "bên thuê", "bên a", "bên b"]):
                    score += 0.8

            # Ưu tiên theo loại hợp đồng
            contract_type = meta.get("contract_type", "")
            if contract_type == "hop_dong_thue_dat" and "thue" in query_lower:
                score += 1.0
            if contract_type == "giay_muon_tien" and any(x in query_lower for x in ["lãi", "trả nợ", "mượn"]):
                score += 1.0

            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    except Exception as e:
        print(f"[retriever] Lỗi: {e}")
        # Fallback về cách cũ
        vector_store = load_store(store_path)
        return vector_store.as_retriever(search_kwargs={"k": k}).invoke(query)