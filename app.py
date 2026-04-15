import os
import streamlit as st
from rag.ingest import ingest_document, ingest_folder
from rag.pipeline import answer_question

st.set_page_config(page_title="Legal RAG Mini", layout="wide")
st.title("⚖️ Legal RAG Mini - Tra cứu hợp đồng, luật, quy định")

USER_STORE = "data/vector_store/user_docs"
LEGAL_STORE = "data/vector_store/legal_docs"
UPLOAD_DIR = "data/uploads"
LEGAL_DOCS_DIR = "legal_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("data/vector_store", exist_ok=True)

st.sidebar.header("⚙️ Quản lý dữ liệu")

# Ingest legal docs
if st.sidebar.button("📚 Ingest kho luật / quy định"):
    if not os.path.exists(LEGAL_DOCS_DIR):
        st.sidebar.error("❌ Chưa có folder legal_docs/")
    else:
        with st.spinner("Đang ingest kho luật / quy định..."):
            ingest_folder(LEGAL_DOCS_DIR, LEGAL_STORE, doc_type="legal")
        st.sidebar.success("✅ Đã ingest legal docs")

# Reset vector DB
if st.sidebar.button("🗑️ Xóa toàn bộ vector store"):
    import shutil
    if os.path.exists("data/vector_store"):
        shutil.rmtree("data/vector_store")
    os.makedirs("data/vector_store", exist_ok=True)
    st.sidebar.success("✅ Đã xóa vector store")

st.subheader("1) Upload tài liệu người dùng")
uploaded_file = st.file_uploader("Upload PDF hợp đồng / giấy tờ", type="pdf")

if uploaded_file:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Đang ingest tài liệu người dùng..."):
        ingest_document(file_path, USER_STORE, doc_type="user")

    st.success("✅ Tài liệu người dùng đã được ingest")

st.subheader("2) Hỏi đáp tài liệu")

query = st.text_input(
    "Nhập câu hỏi",
    placeholder="Ví dụ: Điều 3 là gì? / Lãi suất có hợp pháp không? / Tóm tắt nghĩa vụ của bên B"
)

if st.button("🔍 Tra cứu") and query:
    with st.spinner("Đang phân tích tài liệu..."):
        answer, user_docs, legal_docs = answer_question(query)

    st.markdown("## 📌 Câu trả lời")
    st.write(answer)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 📄 Nguồn từ tài liệu người dùng")
        if user_docs:
            for i, doc in enumerate(user_docs):
                st.write(f"**Nguồn {i+1}:**")
                st.write(doc.page_content[:700] + "...")
                st.caption(f"Metadata: {doc.metadata}")
        else:
            st.info("Không có dữ liệu user_docs")

    with col2:
        st.markdown("## ⚖️ Nguồn từ luật / quy định")
        if legal_docs:
            for i, doc in enumerate(legal_docs):
                st.write(f"**Nguồn {i+1}:**")
                st.write(doc.page_content[:700] + "...")
                st.caption(f"Metadata: {doc.metadata}")
        else:
            st.info("Không có dữ liệu legal_docs")