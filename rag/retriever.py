from langchain_community.vectorstores import FAISS
from rag.ingest import get_embeddings

def load_store(store_path: str):
    embeddings = get_embeddings()
    return FAISS.load_local(
        store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

def retrieve_docs(store_path: str, query: str, k: int = 5):
    vector_store = load_store(store_path)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)